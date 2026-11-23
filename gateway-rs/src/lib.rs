pub mod config;

use axum::{routing::get, Json, Router};
use serde_json::{json, Value};

pub fn health_router() -> Router {
    Router::new().route("/healthz", get(healthz))
}

async fn healthz() -> Json<Value> {
    Json(json!({ "status": "ok", "service": "convergent-gateway" }))
}
