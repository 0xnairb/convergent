use axum::{routing::post, Json, Router};
use convergent_gateway::{
    config::Config,
    ethical_walls::filter_withheld_blocks,
    kv_keys::{content_blake3, derive_kv_key},
    openfga::OpenFgaClient,
    retrieval::prefiltered_hybrid_search,
};
use serde_json::{json, Value};
use sqlx::postgres::PgPoolOptions;
use tokio::net::TcpListener;
use uuid::Uuid;

#[tokio::test]
async fn phase2_openfga_prefilter_revocation_hmac_and_ethical_walls() -> anyhow::Result<()> {
    let cfg = Config::from_env();
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&cfg.pg_dsn)
        .await?;
    let valkey_client = redis::Client::open(cfg.valkey_url.as_str())?;
    let mut valkey = valkey_client.get_multiplexed_async_connection().await?;
    let fga = OpenFgaClient::new(&cfg.openfga_url);
    let store_id = fga.resolve_store_id("convergent").await?;

    let org_id = Uuid::new_v4();
    let space_alpha = Uuid::new_v4();
    let space_beta = Uuid::new_v4();
    let doc_alpha = Uuid::new_v4();
    let doc_beta = Uuid::new_v4();
    let alice = format!("alice-{}", Uuid::new_v4());

    // Insert 1 document + 1 block in space_alpha and 1 in space_beta
    let alpha_text = "Confidential Project Alpha acquisition price is $42M.";
    let beta_text = "Confidential Project Beta acquisition price is $99M.";
    let alpha_hash = content_blake3(alpha_text);
    let beta_hash = content_blake3(beta_text);

    for (doc_id, space_id, scope, uri, text, hash) in [
        (
            doc_alpha,
            space_alpha,
            format!("space:{space_alpha}:matter:alpha"),
            "doc://alpha/deal",
            alpha_text,
            &alpha_hash,
        ),
        (
            doc_beta,
            space_beta,
            format!("space:{space_beta}:matter:beta"),
            "doc://beta/deal",
            beta_text,
            &beta_hash,
        ),
    ] {
        sqlx::query(
            "INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3) VALUES ($1,$2,$3,$4,$5,$6)",
        )
        .bind(doc_id)
        .bind(org_id)
        .bind(space_id)
        .bind(&scope)
        .bind(uri)
        .bind(hash)
        .execute(&pool)
        .await?;

        sqlx::query(
            "INSERT INTO knowledge_blocks (id, document_id, org_id, space_id, clearance_scope, block_type, canonical_text, content_blake3, citations) VALUES ($1,$2,$3,$4,$5,'fact',$6,$7,'[]'::jsonb)",
        )
        .bind(Uuid::new_v4())
        .bind(doc_id)
        .bind(org_id)
        .bind(space_id)
        .bind(&scope)
        .bind(text)
        .bind(hash)
        .execute(&pool)
        .await?;
    }

    // Bind documents to spaces in OpenFGA, grant alice viewer ONLY on space_alpha
    fga.write_tuple(&store_id, &format!("space:{space_alpha}"), "space", &format!("document:{doc_alpha}")).await?;
    fga.write_tuple(&store_id, &format!("space:{space_beta}"), "space", &format!("document:{doc_beta}")).await?;
    fga.write_tuple(&store_id, &format!("user:{alice}"), "viewer", &format!("space:{space_alpha}")).await?;

    // 2.1a: Pre-filtered search returns ONLY alpha, zero beta
    let allowed = fga.list_allowed_documents(&store_id, &alice, &mut valkey, 30).await?;
    assert!(allowed.contains(&doc_alpha));
    assert!(!allowed.contains(&doc_beta));

    let hits = prefiltered_hybrid_search(&pool, org_id, None, &allowed, "acquisition price", None, 10).await?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document_id, doc_alpha);

    // 2.1b: Revoke tuple + invalidate Valkey cache -> zero documents returned
    fga.delete_tuple(&store_id, &format!("user:{alice}"), "viewer", &format!("space:{space_alpha}")).await?;
    OpenFgaClient::invalidate_cache(&store_id, &alice, &mut valkey).await?;
    let allowed_after = fga.list_allowed_documents(&store_id, &alice, &mut valkey, 30).await?;
    assert!(!allowed_after.contains(&doc_alpha));

    // 2.2a: Per-clearance HMAC-SHA256 KV key isolation
    let master_seed = b"super-secret-vpc-seed-2026";
    let identical_hash = content_blake3("Same confidential clause text.");
    let key_alpha = derive_kv_key(master_seed, "space:alpha:matter:coke", "qwen-72b", &identical_hash);
    let key_beta = derive_kv_key(master_seed, "space:beta:matter:pepsi", "qwen-72b", &identical_hash);
    let key_org_user1 = derive_kv_key(master_seed, "org:acme:public", "qwen-72b", &identical_hash);
    let key_org_user2 = derive_kv_key(master_seed, "org:acme:public", "qwen-72b", &identical_hash);
    assert_ne!(key_alpha, key_beta, "Different clearances must produce disjoint KV keys");
    assert_eq!(key_org_user1, key_org_user2, "Same org clearance must share warm KV keys");

    // 2.2b: Opt-in OpenBox Ethical Walls webhook drops Withheld[] block before inference
    let withheld_target = alpha_hash.clone();
    let app = Router::new().route(
        "/api/v1/governance/evaluate",
        post(move |_body: Json<Value>| {
            let w = withheld_target.clone();
            async move {
                Json(json!({
                    "verdict": "CONSTRAIN",
                    "withheld_handles": [w]
                }))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let http = reqwest::Client::new();
    let (kept, withheld) = filter_withheld_blocks(&http, &format!("http://{addr}"), &alice, hits).await?;
    assert_eq!(withheld, vec![alpha_hash]);
    assert!(kept.is_empty(), "Withheld block must be removed before inference");

    Ok(())
}
