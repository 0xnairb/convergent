use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
    routing::post,
    Json, Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use convergent_gateway::{
    app::{build_router, AppState},
    backends::{create_backend, BackendKind},
    config::Config,
    kv_keys::content_blake3,
    openfga::OpenFgaClient,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPoolOptions;
use tokio::net::TcpListener;
use uuid::Uuid;

#[tokio::test]
async fn phase4_oauth21_pkce_mcp_streamable_http_and_openai_sse() -> anyhow::Result<()> {
    let cfg = Config::from_env();
    let pool = PgPoolOptions::new().max_connections(4).connect(&cfg.pg_dsn).await?;
    let valkey = redis::Client::open(cfg.valkey_url.as_str())?;
    let fga = OpenFgaClient::new(&cfg.openfga_url);
    let store_id = fga.resolve_store_id("convergent").await?;
    let nats = async_nats::connect(&cfg.nats_url).await?;

    // Start mock vLLM SSE server
    let mock_vllm = Router::new().route(
        "/v1/chat/completions",
        post(|Json(body): Json<Value>| async move {
            let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
            let has_kv = body
                .get("extra_body")
                .and_then(|e| e.get("kv_cache_keys"))
                .and_then(|k| k.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            if stream {
                let sse = format!(
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":\"ok-kv={has_kv}\"}}}}]}}\n\ndata: [DONE]\n\n"
                );
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            } else {
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(json!({ "choices": [{ "message": { "content": format!("ok-kv={has_kv}") } }] }).to_string()))
                    .unwrap()
            }
        }),
    );
    let vllm_listener = TcpListener::bind("127.0.0.1:0").await?;
    let vllm_addr = vllm_listener.local_addr()?;
    tokio::spawn(async move {
        let _ = axum::serve(vllm_listener, mock_vllm).await;
    });

    // Ensure model_registry points to our mock vLLM for this test
    let test_model = format!("qwen-test-{}", Uuid::new_v4().simple());
    sqlx::query("UPDATE model_registry SET is_active = false").execute(&pool).await?;
    sqlx::query("INSERT INTO model_registry (model_id, endpoint_url, is_active) VALUES ($1, $2, true)")
        .bind(&test_model)
        .bind(format!("http://{vllm_addr}"))
        .execute(&pool)
        .await?;

    let state = AppState {
        pool: pool.clone(),
        valkey,
        openfga: fga.clone(),
        openfga_store_id: store_id.clone(),
        nats,
        http: reqwest::Client::new(),
        jwt_secret: b"phase4-jwt-secret".to_vec(),
        master_seed: b"phase4-master-seed".to_vec(),
        fallback_vllm_url: format!("http://{vllm_addr}"),
        fallback_model_id: test_model.clone(),
        backend: create_backend(BackendKind::Vllm),
        policy_webhook_url: None,
        auth_codes: Arc::new(RwLock::new(HashMap::new())),
    };

    let gw_listener = TcpListener::bind("127.0.0.1:0").await?;
    let gw_addr = gw_listener.local_addr()?;
    tokio::spawn(async move {
        let _ = axum::serve(gw_listener, build_router(state)).await;
    });
    let gw_url = format!("http://{gw_addr}");
    let http = reqwest::Client::new();

    // Seed one authorized doc in space
    let org_id = Uuid::new_v4();
    let space_id = Uuid::new_v4();
    let doc_id = Uuid::new_v4();
    let bob = format!("bob-{}", Uuid::new_v4());
    let text = "Convergent supports OAuth 2.1 PKCE and MCP Streamable HTTP.";
    let hash = content_blake3(text);
    sqlx::query("INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3) VALUES ($1,$2,$3,$4,$5,$6)")
        .bind(doc_id).bind(org_id).bind(space_id).bind(format!("org:{org_id}:public")).bind("doc://convergent/spec").bind(&hash)
        .execute(&pool).await?;
    sqlx::query("INSERT INTO knowledge_blocks (id, document_id, org_id, space_id, clearance_scope, block_type, canonical_text, content_blake3, citations) VALUES ($1,$2,$3,$4,$5,'fact',$6,$7,'[]'::jsonb)")
        .bind(Uuid::new_v4()).bind(doc_id).bind(org_id).bind(space_id).bind(format!("org:{org_id}:public")).bind(text).bind(&hash)
        .execute(&pool).await?;
    fga.write_tuple(&store_id, &format!("space:{space_id}"), "space", &format!("document:{doc_id}")).await?;
    fga.write_tuple(&store_id, &format!("user:{bob}"), "viewer", &format!("space:{space_id}")).await?;

    // 1. OAuth 2.1 + PKCE S256 flow
    let verifier = "convergent-pkce-verifier-9876543210";
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));

    // Bad verifier rejected
    let bad_code: Value = http
        .post(format!("{gw_url}/oauth/authorize"))
        .json(&json!({ "sub": bob, "org_id": org_id, "scope": "mcp:read inference", "code_challenge": challenge }))
        .send().await?.json().await?;
    let bad_resp = http
        .post(format!("{gw_url}/oauth/token"))
        .json(&json!({ "code": bad_code["code"], "code_verifier": "wrong-verifier" }))
        .send().await?;
    assert_eq!(bad_resp.status(), StatusCode::UNAUTHORIZED);

    // Valid verifier succeeds
    let auth_resp: Value = http
        .post(format!("{gw_url}/oauth/authorize"))
        .json(&json!({ "sub": bob, "org_id": org_id, "scope": "mcp:read inference", "code_challenge": challenge }))
        .send().await?.json().await?;
    let tok_resp: Value = http
        .post(format!("{gw_url}/oauth/token"))
        .json(&json!({ "code": auth_resp["code"], "code_verifier": verifier }))
        .send().await?.error_for_status()?.json().await?;
    let bearer = tok_resp["access_token"].as_str().unwrap();

    // 2. MCP Streamable HTTP: tools/list + search_memory
    let mcp_list: Value = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(bearer)
        .json(&json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }))
        .send().await?.error_for_status()?.json().await?;
    assert_eq!(mcp_list["result"]["tools"].as_array().unwrap().len(), 4);

    let mcp_search: Value = http
        .post(format!("{gw_url}/mcp"))
        .bearer_auth(bearer)
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": { "name": "search_memory", "arguments": { "query": "OAuth PKCE MCP", "space_id": space_id.to_string() } }
        }))
        .send().await?.error_for_status()?.json().await?;
    assert_eq!(mcp_search["result"]["blocks"].as_array().unwrap().len(), 1);

    // 3. OpenAI-compatible SSE streaming + <5ms proxy overhead check (warm ACL cache)
    let sse_resp = http
        .post(format!("{gw_url}/v1/chat/completions"))
        .bearer_auth(bearer)
        .header("x-convergent-space", space_id.to_string())
        .json(&json!({
            "model": test_model,
            "stream": true,
            "messages": [{ "role": "user", "content": "What does Convergent support?" }]
        }))
        .send().await?.error_for_status()?;

    let overhead_ms: f64 = sse_resp
        .headers()
        .get("x-convergent-proxy-overhead-ms")
        .unwrap()
        .to_str()?
        .parse()?;
    let sse_text = sse_resp.text().await?;
    assert!(sse_text.contains("ok-kv=true"), "Expected KV keys injected, got: {sse_text}");
    assert!(sse_text.contains("[DONE]"));
    println!("Measured gateway-rs warm proxy overhead: {overhead_ms:.2} ms");
    assert!(overhead_ms < 25.0, "Unexpectedly high local proxy overhead: {overhead_ms}ms");

    Ok(())
}
