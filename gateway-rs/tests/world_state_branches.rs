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
async fn mcp_git_world_state_branch_merge_revert_and_chat_header() -> anyhow::Result<()> {
    let cfg = Config::from_env();
    let pool = PgPoolOptions::new().max_connections(5).connect(&cfg.pg_dsn).await?;
    let valkey = redis::Client::open(cfg.valkey_url.as_str())?;
    let openfga = OpenFgaClient::new(&cfg.openfga_url);
    let store_id = openfga.resolve_store_id("convergent").await?;
    let nats = async_nats::connect(&cfg.nats_url).await?;

    // Echo system prompt back from mock inference endpoint so we prove branch context isolation
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

    let jwt_secret = b"branch-test-jwt".to_vec();
    let state = AppState {
        pool: pool.clone(),
        valkey,
        openfga: openfga.clone(),
        openfga_store_id: store_id.clone(),
        nats,
        http: reqwest::Client::new(),
        jwt_secret: jwt_secret.clone(),
        master_seed: b"branch-test-seed".to_vec(),
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

    // 1. Commit v1 to `main` via MCP
    mcp_call(
        &app,
        &token,
        "commit_memory",
        json!({
            "space_id": space_id,
            "branch_name": "main",
            "source_uri": "adr://001-auth.md",
            "content": "Auth uses session cookies.",
            "message": "Initial auth"
        }),
    )
    .await;

    // 2. Branch `alice/oauth` and commit v2 on branch only
    mcp_call(
        &app,
        &token,
        "branch_memory",
        json!({ "space_id": space_id, "new_branch": "alice/oauth", "from_branch": "main" }),
    )
    .await;
    mcp_call(
        &app,
        &token,
        "commit_memory",
        json!({
            "space_id": space_id,
            "branch_name": "alice/oauth",
            "source_uri": "adr://001-auth.md",
            "content": "Auth uses OAuth 2.1 PKCE.",
            "message": "Switch to OAuth 2.1"
        }),
    )
    .await;

    // 3. Chat on `X-Convergent-Branch: alice/oauth` sees OAuth; `main` still sees session cookies
    let branch_chat = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {token}"))
                .header("x-convergent-space", space_id.to_string())
                .header("x-convergent-branch", "alice/oauth")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "messages": [{ "role": "user", "content": "Auth" }] }).to_string(),
                ))?,
        )
        .await?;
    assert_eq!(branch_chat.status(), StatusCode::OK);
    assert_eq!(branch_chat.headers().get("x-convergent-branch").unwrap(), "alice/oauth");
    let chat_json: Value = serde_json::from_slice(&axum::body::to_bytes(branch_chat.into_body(), usize::MAX).await?)?;
    assert!(chat_json["choices"][0]["message"]["content"].as_str().unwrap().contains("OAuth 2.1 PKCE"));

    let main_search = mcp_call(
        &app,
        &token,
        "search_memory",
        json!({ "space_id": space_id, "branch": "main", "query": "Auth" }),
    )
    .await;
    assert!(main_search["blocks"][0]["canonical_text"].as_str().unwrap().contains("session cookies"));

    // 4. Diff & Merge `alice/oauth` -> `main`
    let diff = mcp_call(
        &app,
        &token,
        "diff_memory",
        json!({ "space_id": space_id, "source_branch": "alice/oauth", "target_branch": "main" }),
    )
    .await;
    assert_eq!(diff["diff"]["updated"].as_array().unwrap().len(), 1);

    let merged = mcp_call(
        &app,
        &token,
        "merge_memory",
        json!({ "space_id": space_id, "source_branch": "alice/oauth", "target_branch": "main" }),
    )
    .await;
    let merge_commit_id = merged["merge"]["merge_commit_id"].as_str().unwrap().to_string();

    let main_after_merge = mcp_call(
        &app,
        &token,
        "search_memory",
        json!({ "space_id": space_id, "branch": "main", "query": "Auth" }),
    )
    .await;
    assert!(main_after_merge["blocks"][0]["canonical_text"].as_str().unwrap().contains("OAuth 2.1 PKCE"));

    // 5. Instant Revert on `main` restores `session cookies`
    mcp_call(
        &app,
        &token,
        "revert_memory",
        json!({ "space_id": space_id, "branch_name": "main", "commit_id": merge_commit_id }),
    )
    .await;
    let main_reverted = mcp_call(
        &app,
        &token,
        "search_memory",
        json!({ "space_id": space_id, "branch": "main", "query": "Auth" }),
    )
    .await;
    assert!(main_reverted["blocks"][0]["canonical_text"].as_str().unwrap().contains("session cookies"));

    Ok(())
}
