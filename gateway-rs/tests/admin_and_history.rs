use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use chrono::{Duration, Utc};
use convergent_gateway::{
    app::{build_router, AppState},
    config::Config,
    oauth::issue_jwt,
    openfga::OpenFgaClient,
};
use serde_json::{json, Value};
use sqlx::postgres::PgPoolOptions;
use tower::ServiceExt;
use uuid::Uuid;

#[tokio::test]
async fn admin_space_grants_and_bitemporal_memory_history() -> anyhow::Result<()> {
    let cfg = Config::from_env();
    let pool = PgPoolOptions::new().max_connections(5).connect(&cfg.pg_dsn).await?;
    let valkey = redis::Client::open(cfg.valkey_url.as_str())?;
    let openfga = OpenFgaClient::new(&cfg.openfga_url);
    let store_id = openfga.resolve_store_id("convergent").await?;
    let nats = async_nats::connect(&cfg.nats_url).await?;

    let jwt_secret = b"test-admin-jwt-secret".to_vec();
    let state = AppState {
        pool: pool.clone(),
        valkey,
        openfga: openfga.clone(),
        openfga_store_id: store_id.clone(),
        nats,
        http: reqwest::Client::new(),
        jwt_secret: jwt_secret.clone(),
        master_seed: b"test-admin-seed".to_vec(),
        fallback_vllm_url: "http://127.0.0.1:8000".into(),
        fallback_model_id: "Qwen/Qwen3-32B-FP8".into(),
        openbox_url: None,
        auth_codes: Arc::new(RwLock::new(HashMap::new())),
    };
    let app = build_router(state);

    let org_id = Uuid::new_v4();
    let space_id = Uuid::new_v4();
    let doc_v1 = Uuid::new_v4();
    let doc_v2 = Uuid::new_v4();
    let block_v1 = Uuid::new_v4();
    let block_v2 = Uuid::new_v4();
    let user_id = format!("bob-{}", &Uuid::new_v4().simple().to_string()[..8]);
    let source_uri = format!("docs://admin-test/{}.md", Uuid::new_v4().simple());

    let t_v1 = Utc::now() - Duration::minutes(10);
    let t_cutover = Utc::now() - Duration::minutes(5);
    let t_as_of_v1 = Utc::now() - Duration::minutes(7);
    let zero_vec = format!("[{}]", vec!["0"; 64].join(","));

    for (doc_id, hash) in [(doc_v1, "hash-v1"), (doc_v2, "hash-v2")] {
        sqlx::query(
            "INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3) VALUES ($1,$2,$3,$4,$5,$6)",
        )
        .bind(doc_id)
        .bind(org_id)
        .bind(space_id)
        .bind(format!("space:{space_id}"))
        .bind(&source_uri)
        .bind(hash)
        .execute(&pool)
        .await?;
        openfga
            .write_tuple(&store_id, &format!("space:{space_id}"), "space", &format!("document:{doc_id}"))
            .await?;
    }

    sqlx::query(
        "INSERT INTO knowledge_blocks (id, document_id, org_id, space_id, clearance_scope, block_type, canonical_text, content_blake3, citations, embedding, valid_from) VALUES ($1,$2,$3,$4,$5,'fact',$6,'bhash-v2','[]'::jsonb,$7::vector,$8)",
    )
    .bind(block_v2)
    .bind(doc_v2)
    .bind(org_id)
    .bind(space_id)
    .bind(format!("space:{space_id}"))
    .bind("Version 2: updated SLA is 4 hours.")
    .bind(&zero_vec)
    .bind(t_cutover)
    .execute(&pool)
    .await?;

    sqlx::query(
        "INSERT INTO knowledge_blocks (id, document_id, org_id, space_id, clearance_scope, block_type, canonical_text, content_blake3, citations, embedding, valid_from, valid_to, superseded_by) VALUES ($1,$2,$3,$4,$5,'fact',$6,'bhash-v1','[]'::jsonb,$7::vector,$8,$9,$10)",
    )
    .bind(block_v1)
    .bind(doc_v1)
    .bind(org_id)
    .bind(space_id)
    .bind(format!("space:{space_id}"))
    .bind("Version 1: legacy SLA is 24 hours.")
    .bind(&zero_vec)
    .bind(t_v1)
    .bind(t_cutover)
    .bind(block_v2)
    .execute(&pool)
    .await?;

    let token = issue_jwt(&jwt_secret, &user_id, org_id, "admin", 3600)?;

    // 1. Grant viewer via HTTP API
    let grant_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/spaces/{space_id}/grants"))
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "user_id": user_id, "relation": "viewer", "action": "grant" }).to_string(),
                ))?,
        )
        .await?;
    assert_eq!(grant_resp.status(), StatusCode::OK);

    // 2. Query historical as_of=t_as_of_v1 -> must return only v1
    let encoded_uri = source_uri.replace(':', "%3A").replace('/', "%2F");
    let encoded_ts = t_as_of_v1.to_rfc3339().replace('+', "%2B").replace(':', "%3A");
    let hist_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/memory/history?source_uri={encoded_uri}&as_of={encoded_ts}"))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(hist_resp.status(), StatusCode::OK);
    let hist_bytes = axum::body::to_bytes(hist_resp.into_body(), usize::MAX).await?;
    let hist_json: Value = serde_json::from_slice(&hist_bytes)?;
    let blocks = hist_json["blocks"].as_array().unwrap();
    assert_eq!(blocks.len(), 1);
    assert!(blocks[0]["canonical_text"].as_str().unwrap().contains("Version 1"));

    // 3. Revoke viewer via HTTP API -> immediate Valkey ACL invalidation -> empty result
    let revoke_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/spaces/{space_id}/grants"))
                .header("authorization", format!("Bearer {token}"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "user_id": user_id, "relation": "viewer", "action": "revoke" }).to_string(),
                ))?,
        )
        .await?;
    assert_eq!(revoke_resp.status(), StatusCode::OK);

    let after_revoke = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/memory/history?source_uri={encoded_uri}"))
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(after_revoke.status(), StatusCode::OK);
    let after_bytes = axum::body::to_bytes(after_revoke.into_body(), usize::MAX).await?;
    let after_json: Value = serde_json::from_slice(&after_bytes)?;
    assert!(after_json["blocks"].as_array().unwrap().is_empty());

    Ok(())
}
