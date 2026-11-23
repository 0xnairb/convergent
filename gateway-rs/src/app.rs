use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::TryStreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

use crate::admin::{manage_space_grant, memory_history};
use crate::ethical_walls::filter_withheld_blocks;
use crate::inference::{build_blended_payload, forward_completion, get_active_model};
use crate::mcp::handle_mcp;
use crate::oauth::{issue_jwt, verify_jwt, verify_pkce_s256, Claims};
use crate::openfga::OpenFgaClient;
use crate::retrieval::prefiltered_hybrid_search;

#[derive(Clone)]
pub struct PkceAuthCode {
    pub sub: String,
    pub org_id: Uuid,
    pub scope: String,
    pub code_challenge: String,
}

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
    pub valkey: redis::Client,
    pub openfga: OpenFgaClient,
    pub openfga_store_id: String,
    pub nats: async_nats::Client,
    pub http: reqwest::Client,
    pub jwt_secret: Vec<u8>,
    pub master_seed: Vec<u8>,
    pub fallback_vllm_url: String,
    pub fallback_model_id: String,
    pub openbox_url: Option<String>,
    pub auth_codes: Arc<RwLock<HashMap<String, PkceAuthCode>>>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/oauth/authorize", post(oauth_authorize))
        .route("/oauth/token", post(oauth_token))
        .route("/mcp", post(handle_mcp))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/spaces/{space_id}/grants", post(manage_space_grant))
        .route("/v1/memory/history", get(memory_history))
        .with_state(state)
}

async fn healthz() -> Json<Value> {
    Json(json!({ "status": "ok", "service": "convergent-gateway" }))
}

#[derive(Deserialize)]
struct AuthorizeReq {
    sub: String,
    org_id: Uuid,
    scope: String,
    code_challenge: String,
}

async fn oauth_authorize(
    State(state): State<AppState>,
    Json(req): Json<AuthorizeReq>,
) -> impl IntoResponse {
    let code = format!("cvg_code_{}", Uuid::new_v4().simple());
    state.auth_codes.write().unwrap().insert(
        code.clone(),
        PkceAuthCode {
            sub: req.sub,
            org_id: req.org_id,
            scope: req.scope,
            code_challenge: req.code_challenge,
        },
    );
    Json(json!({ "code": code }))
}

#[derive(Deserialize)]
struct TokenReq {
    code: String,
    code_verifier: String,
}

async fn oauth_token(
    State(state): State<AppState>,
    Json(req): Json<TokenReq>,
) -> impl IntoResponse {
    let entry = state.auth_codes.write().unwrap().remove(&req.code);
    let Some(auth) = entry else {
        return (StatusCode::UNAUTHORIZED, Json(json!({ "error": "invalid_grant" }))).into_response();
    };
    if !verify_pkce_s256(&req.code_verifier, &auth.code_challenge) {
        return (StatusCode::UNAUTHORIZED, Json(json!({ "error": "invalid_pkce_verifier" }))).into_response();
    }
    match issue_jwt(&state.jwt_secret, &auth.sub, auth.org_id, &auth.scope, 3600) {
        Ok(token) => (
            StatusCode::OK,
            Json(json!({ "access_token": token, "token_type": "Bearer", "expires_in": 3600 })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub fn extract_claims(headers: &HeaderMap, jwt_secret: &[u8]) -> Result<Claims, String> {
    let raw = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| "missing Authorization header".to_string())?;
    let token = raw
        .strip_prefix("Bearer ")
        .ok_or_else(|| "expected Bearer token".to_string())?;
    verify_jwt(jwt_secret, token).map_err(|e| e.to_string())
}

async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let active = get_active_model(&state.pool, &state.fallback_vllm_url, &state.fallback_model_id).await;
    Json(json!({
        "object": "list",
        "data": [{ "id": active.model_id, "object": "model", "owned_by": "convergent-vpc" }]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let t0 = Instant::now();
    let claims = match extract_claims(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(msg) => return (StatusCode::UNAUTHORIZED, Json(json!({ "error": msg }))).into_response(),
    };

    let space_id = headers
        .get("x-convergent-space")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| Uuid::parse_str(s).ok());

    let mut blocks = vec![];
    if let Some(sid) = space_id {
        let mut valkey = match state.valkey.get_multiplexed_async_connection().await {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_GATEWAY, Json(json!({ "error": e.to_string() }))).into_response(),
        };
        let allowed = match state
            .openfga
            .list_allowed_documents(&state.openfga_store_id, &claims.sub, &mut valkey, 30)
            .await
        {
            Ok(ids) => ids,
            Err(e) => return (StatusCode::BAD_GATEWAY, Json(json!({ "error": e.to_string() }))).into_response(),
        };
        let query = body
            .get("messages")
            .and_then(|m| m.as_array())
            .and_then(|arr| arr.last())
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        if !query.is_empty() {
            if let Ok(found) = prefiltered_hybrid_search(
                &state.pool,
                claims.org_id,
                Some(sid),
                &allowed,
                query,
                None,
                5,
            )
            .await
            {
                blocks = found;
            }
        }
    }

    if let Some(ref openbox_url) = state.openbox_url {
        match filter_withheld_blocks(&state.http, openbox_url, &claims.sub, blocks).await {
            Ok((kept, _withheld)) => blocks = kept,
            Err(e) => return (StatusCode::FORBIDDEN, Json(json!({ "error": e.to_string() }))).into_response(),
        }
    }

    let active = get_active_model(&state.pool, &state.fallback_vllm_url, &state.fallback_model_id).await;
    let (payload, _kv_keys) = build_blended_payload(body, &active.model_id, &state.master_seed, &blocks);
    let proxy_overhead_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let upstream = match forward_completion(&state.http, &active.endpoint_url, &payload).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY, Json(json!({ "error": e.to_string() }))).into_response(),
    };

    let content_type = upstream
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));

    let stream = upstream.bytes_stream().map_err(std::io::Error::other);
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut().insert("content-type", content_type);
    if let Ok(hv) = HeaderValue::from_str(&format!("{proxy_overhead_ms:.2}")) {
        resp.headers_mut().insert("x-convergent-proxy-overhead-ms", hv);
    }
    resp
}
