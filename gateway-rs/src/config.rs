use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub pg_dsn: String,
    pub valkey_url: String,
    pub openfga_url: String,
    pub openfga_store_name: String,
    pub nats_url: String,
    pub bind_addr: String,
    pub jwt_secret: String,
    pub master_seed: String,
    pub fallback_vllm_url: String,
    pub fallback_model_id: String,
    pub inference_backend: String,
    pub policy_webhook_url: Option<String>,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            pg_dsn: env::var("CONVERGENT_PG_DSN").unwrap_or_else(|_| {
                "postgres://convergent:convergent_dev_password@127.0.0.1:55432/convergent".into()
            }),
            valkey_url: env::var("CONVERGENT_VALKEY_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:56379/0".into()),
            openfga_url: env::var("CONVERGENT_OPENFGA_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:58080".into()),
            openfga_store_name: env::var("CONVERGENT_OPENFGA_STORE")
                .unwrap_or_else(|_| "convergent".into()),
            nats_url: env::var("CONVERGENT_NATS_URL")
                .unwrap_or_else(|_| "nats://127.0.0.1:54222".into()),
            bind_addr: env::var("CONVERGENT_BIND_ADDR")
                .unwrap_or_else(|_| "127.0.0.1:8090".into()),
            jwt_secret: env::var("CONVERGENT_JWT_SECRET")
                .unwrap_or_else(|_| "convergent-dev-jwt-secret-change-in-prod".into()),
            master_seed: env::var("CONVERGENT_MASTER_SEED")
                .unwrap_or_else(|_| "convergent-dev-kv-master-seed-change-in-prod".into()),
            fallback_vllm_url: env::var("CONVERGENT_VLLM_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8000".into()),
            fallback_model_id: env::var("CONVERGENT_MODEL_ID")
                .unwrap_or_else(|_| "Qwen/Qwen3-32B-FP8".into()),
            inference_backend: env::var("CONVERGENT_INFERENCE_BACKEND")
                .unwrap_or_else(|_| "vllm".into()),
            policy_webhook_url: env::var("CONVERGENT_POLICY_WEBHOOK_URL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
        }
    }
}
