use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    response::Response,
    routing::post,
    Json, Router,
};
use convergent_gateway::{
    app::{build_router, AppState},
    backends::{create_backend, BackendKind},
    config::Config,
    oauth::issue_jwt,
    openfga::OpenFgaClient,
};
use serde_json::{json, Value};
use sqlx::postgres::PgPoolOptions;
use tokio::net::TcpListener;
use tower::ServiceExt;
use uuid::Uuid;

async fn mcp_call(app: &Router, token: &str, tool: &str, arguments: Value) -> Value {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": { "name": tool, "arguments": arguments } })
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    body["result"].clone()
}

#[tokio::test]
async fn mcp_zero_copy_overlays_auto_rebase_conflict_undo_and_chat_header() -> anyhow::Result<()> {
    let cfg = Config::from_env();
    let pool = PgPoolOptions::new().max_connections(5).connect(&cfg.pg_dsn).await?;
    let valkey = redis::Client::open(cfg.valkey_url.as_str())?;
    let openfga = OpenFgaClient::new(&cfg.openfga_url);
    let store_id = openfga.resolve_store_id("convergent").await?;
    let nats = async_nats::connect(&cfg.nats_url).await?;

    let mock_llm = Router::new().route(
        "/v1/chat/completions",
        post(|Json(body): Json<Value>| async move {
            let sys = body["messages"][0]["content"].as_str().unwrap_or("").to_string();
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({ "choices": [{ "message": { "content": sys } }] }).to_string(),
                ))
                .unwrap()
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let llm_url = format!("http://{}", listener.local_addr()?);
    tokio::spawn(async move {
        let _ = axum::serve(listener, mock_llm).await;
    });

    sqlx::query("UPDATE model_registry SET is_active = false").execute(&pool).await?;

    let jwt_secret = b"overlay-test-jwt".to_vec();
    let state = AppState {
        pool: pool.clone(),
        valkey,
        openfga: openfga.clone(),
        openfga_store_id: store_id.clone(),
        nats,
        http: reqwest::Client::new(),
        jwt_secret: jwt_secret.clone(),
        master_seed: b"overlay-test-seed".to_vec(),
        fallback_vllm_url: llm_url,
        fallback_model_id: "granite4.2:3b".into(),
        backend: create_backend(BackendKind::Vllm),
        policy_webhook_url: None,
        auth_codes: Arc::new(RwLock::new(HashMap::new())),
    };
    let app = build_router(state);

    let org_id = Uuid::new_v4();
    let space_id = Uuid::new_v4();
    let alice = format!("alice-{}", &Uuid::new_v4().simple().to_string()[..6]);
    let token = issue_jwt(&jwt_secret, &alice, org_id, "mcp:write", 3600)?;
    openfga
        .write_tuple(&store_id, &format!("user:{alice}"), "viewer", &format!("space:{space_id}"))
        .await?;

    // 1. Seed r1 via open_overlay -> stage_delta -> land_overlay
    let ov_init = mcp_call(&app, &token, "open_overlay", json!({ "space_id": space_id })).await;
    let init_id = ov_init["overlay"]["overlay_id"].as_str().unwrap().to_string();
    mcp_call(
        &app,
        &token,
        "stage_delta",
        json!({ "overlay_id": init_id, "source_uri": "adr://001-auth.md", "content": "Auth uses session cookies." }),
    )
    .await;
    let r1 = mcp_call(&app, &token, "land_overlay", json!({ "overlay_id": init_id, "message": "Init r1" })).await;
    assert_eq!(r1["land"]["rev"], 1);

    // 2. Open two parallel 0-byte overlays at base_rev=1
    let ov_a = mcp_call(&app, &token, "open_overlay", json!({ "space_id": space_id })).await;
    let ov_b = mcp_call(&app, &token, "open_overlay", json!({ "space_id": space_id })).await;
    let a_id = ov_a["overlay"]["overlay_id"].as_str().unwrap().to_string();
    let b_id = ov_b["overlay"]["overlay_id"].as_str().unwrap().to_string();

    mcp_call(
        &app,
        &token,
        "stage_delta",
        json!({ "overlay_id": a_id, "source_uri": "adr://001-auth.md", "content": "Auth uses OAuth 2.1 PKCE." }),
    )
    .await;
    mcp_call(
        &app,
        &token,
        "stage_delta",
        json!({ "overlay_id": b_id, "source_uri": "runbook://deploy.md", "content": "Daily continuous deploys." }),
    )
    .await;

    // 3. Chat with X-Convergent-Overlay: a_id sees OAuth 2.1 PKCE; base space still sees session cookies
    let ov_chat = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {token}"))
                .header("x-convergent-space", space_id.to_string())
                .header("x-convergent-overlay", &a_id)
                .header("content-type", "application/json")
                .body(Body::from(json!({ "messages": [{ "role": "user", "content": "Auth" }] }).to_string()))?,
        )
        .await?;
    assert_eq!(ov_chat.status(), StatusCode::OK);
    assert_eq!(ov_chat.headers().get("x-convergent-overlay").unwrap().to_str()?, a_id);
    let chat_json: Value = serde_json::from_slice(&axum::body::to_bytes(ov_chat.into_body(), usize::MAX).await?)?;
    assert!(chat_json["choices"][0]["message"]["content"].as_str().unwrap().contains("OAuth 2.1 PKCE"));

    let base_search = mcp_call(&app, &token, "search_memory", json!({ "space_id": space_id, "query": "Auth" })).await;
    assert!(base_search["blocks"][0]["canonical_text"].as_str().unwrap().contains("session cookies"));

    // 4. Land A (r2) and auto-rebase non-overlapping B (r3)
    let r2 = mcp_call(&app, &token, "land_overlay", json!({ "overlay_id": a_id, "message": "Upgrade auth" })).await;
    assert_eq!(r2["land"]["status"], "landed");
    assert_eq!(r2["land"]["rev"], 2);

    let r3 = mcp_call(&app, &token, "land_overlay", json!({ "overlay_id": b_id, "message": "Add deploy runbook" })).await;
    assert_eq!(r3["land"]["status"], "landed");
    assert_eq!(r3["land"]["rev"], 3);

    // 5. Stale overlay at base_rev=1 editing adr://001-auth.md -> CONFLICT
    let ov_c = mcp_call(&app, &token, "open_overlay", json!({ "space_id": space_id, "base_rev": 1 })).await;
    let c_id = ov_c["overlay"]["overlay_id"].as_str().unwrap().to_string();
    mcp_call(
        &app,
        &token,
        "stage_delta",
        json!({ "overlay_id": c_id, "source_uri": "adr://001-auth.md", "content": "Auth uses mTLS." }),
    )
    .await;
    let conflict = mcp_call(&app, &token, "land_overlay", json!({ "overlay_id": c_id, "message": "Conflict" })).await;
    assert_eq!(conflict["land"]["status"], "conflict");

    // 6. Undo r2 restores `session cookies`
    mcp_call(&app, &token, "undo_revision", json!({ "space_id": space_id, "rev": 2 })).await;
    let after_undo = mcp_call(&app, &token, "search_memory", json!({ "space_id": space_id, "query": "Auth" })).await;
    assert!(after_undo["blocks"][0]["canonical_text"].as_str().unwrap().contains("session cookies"));

    Ok(())
}
