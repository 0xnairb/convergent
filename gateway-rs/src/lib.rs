pub mod admin;
pub mod app;
pub mod backends;
pub mod config;
pub mod ethical_walls;
pub mod inference;
pub mod kv_keys;
pub mod mcp;
pub mod oauth;
pub mod openfga;
pub mod overlays;
pub mod retrieval;

use axum::{routing::get, Json, Router};
use serde_json::{json, Value};

pub fn health_router() -> Router {
    Router::new().route("/healthz", get(healthz))
}

async fn healthz() -> Json<Value> {
    Json(json!({ "status": "ok", "service": "convergent-gateway" }))
}
