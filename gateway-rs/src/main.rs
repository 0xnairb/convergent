use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use convergent_gateway::{
    app::{build_router, AppState},
    backends::create_backend_from_str,
    config::Config,
    openfga::OpenFgaClient,
};
use sqlx::postgres::PgPoolOptions;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cfg = Config::from_env();

    let pool = PgPoolOptions::new()
        .max_connections(16)
        .connect(&cfg.pg_dsn)
        .await?;
    let valkey = redis::Client::open(cfg.valkey_url.as_str())?;
    let openfga = OpenFgaClient::new(&cfg.openfga_url);
    let openfga_store_id = openfga.resolve_store_id(&cfg.openfga_store_name).await?;
    let nats = async_nats::connect(&cfg.nats_url).await?;

    let backend = create_backend_from_str(&cfg.inference_backend)?;

    let state = AppState {
        pool,
        valkey,
        openfga,
        openfga_store_id,
        nats,
        http: reqwest::Client::new(),
        jwt_secret: cfg.jwt_secret.into_bytes(),
        master_seed: cfg.master_seed.into_bytes(),
        fallback_vllm_url: cfg.fallback_vllm_url,
        fallback_model_id: cfg.fallback_model_id,
        backend: backend.clone(),
        policy_webhook_url: cfg.policy_webhook_url,
        auth_codes: Arc::new(RwLock::new(HashMap::new())),
    };

    let listener = TcpListener::bind(&cfg.bind_addr).await?;
    tracing::info!(
        "convergent-gateway listening on {} (backend={}, hardware={})",
        cfg.bind_addr,
        backend.name(),
        backend.hardware_target()
    );
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}
