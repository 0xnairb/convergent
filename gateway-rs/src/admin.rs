use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::app::{extract_claims, AppState};
use crate::openfga::OpenFgaClient;

#[derive(Deserialize)]
pub struct SpaceGrantReq {
    pub user_id: String,
    pub relation: String,
    #[serde(default = "default_grant_action")]
    pub action: String, // "grant" or "revoke"
}

fn default_grant_action() -> String {
    "grant".into()
}

/// Grant or revoke an OpenFGA relation (`reader`, `editor`, `owner`) on a space and invalidate Valkey ACL cache.
pub async fn manage_space_grant(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(space_id): Path<Uuid>,
    Json(req): Json<SpaceGrantReq>,
) -> impl IntoResponse {
    if let Err(msg) = extract_claims(&headers, &state.jwt_secret) {
        return (StatusCode::UNAUTHORIZED, Json(json!({ "error": msg }))).into_response();
    }

    if !matches!(req.relation.as_str(), "viewer" | "editor" | "restricted") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "relation must be viewer, editor, or restricted" })),
        )
            .into_response();
    }

    let user = format!("user:{}", req.user_id);
    let object = format!("space:{space_id}");

    let op = match req.action.as_str() {
        "grant" => {
            state
                .openfga
                .write_tuple(&state.openfga_store_id, &user, &req.relation, &object)
                .await
        }
        "revoke" => {
            state
                .openfga
                .delete_tuple(&state.openfga_store_id, &user, &req.relation, &object)
                .await
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "action must be grant or revoke" })),
            )
                .into_response();
        }
    };

    if let Err(e) = op {
        return (StatusCode::BAD_GATEWAY, Json(json!({ "error": e.to_string() }))).into_response();
    }

    if let Ok(mut valkey) = state.valkey.get_multiplexed_async_connection().await {
        let _ = OpenFgaClient::invalidate_cache(&state.openfga_store_id, &req.user_id, &mut valkey).await;
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "action": req.action,
            "user": user,
            "relation": req.relation,
            "object": object,
        })),
    )
        .into_response()
}

#[derive(Deserialize)]
pub struct MemoryHistoryQuery {
    pub source_uri: String,
    pub as_of: Option<DateTime<Utc>>,
}

/// Bi-temporal memory history lookup: returns authorized blocks for `source_uri`, either active or at `as_of`.
pub async fn memory_history(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<MemoryHistoryQuery>,
) -> impl IntoResponse {
    let claims = match extract_claims(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(msg) => return (StatusCode::UNAUTHORIZED, Json(json!({ "error": msg }))).into_response(),
    };

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

    if allowed.is_empty() {
        return (StatusCode::OK, Json(json!({ "blocks": [] }))).into_response();
    }

    let rows = match q.as_of {
        Some(ts) => {
            sqlx::query(
                r#"
                SELECT kb.id, kb.document_id, kb.clearance_scope, kb.canonical_text,
                       kb.content_blake3, kb.citations, kb.valid_from, kb.valid_to, kb.superseded_by
                FROM knowledge_blocks kb
                JOIN documents d ON d.id = kb.document_id
                WHERE kb.org_id = $1
                  AND d.source_uri = $2
                  AND kb.document_id = ANY($3)
                  AND kb.valid_from <= $4
                  AND (kb.valid_to IS NULL OR kb.valid_to > $4)
                ORDER BY kb.valid_from DESC
                "#,
            )
            .bind(claims.org_id)
            .bind(&q.source_uri)
            .bind(&allowed)
            .bind(ts)
            .fetch_all(&state.pool)
            .await
        }
        None => {
            sqlx::query(
                r#"
                SELECT kb.id, kb.document_id, kb.clearance_scope, kb.canonical_text,
                       kb.content_blake3, kb.citations, kb.valid_from, kb.valid_to, kb.superseded_by
                FROM knowledge_blocks kb
                JOIN documents d ON d.id = kb.document_id
                WHERE kb.org_id = $1
                  AND d.source_uri = $2
                  AND kb.document_id = ANY($3)
                ORDER BY kb.valid_from DESC
                "#,
            )
            .bind(claims.org_id)
            .bind(&q.source_uri)
            .bind(&allowed)
            .fetch_all(&state.pool)
            .await
        }
    };

    let rows = match rows {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response(),
    };

    let blocks: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "id": r.get::<Uuid, _>("id"),
                "document_id": r.get::<Uuid, _>("document_id"),
                "clearance_scope": r.get::<String, _>("clearance_scope"),
                "canonical_text": r.get::<String, _>("canonical_text"),
                "content_blake3": r.get::<String, _>("content_blake3"),
                "citations": r.get::<Value, _>("citations"),
                "valid_from": r.get::<DateTime<Utc>, _>("valid_from"),
                "valid_to": r.get::<Option<DateTime<Utc>>, _>("valid_to"),
                "superseded_by": r.get::<Option<Uuid>, _>("superseded_by"),
            })
        })
        .collect();

    (StatusCode::OK, Json(json!({ "blocks": blocks }))).into_response()
}
