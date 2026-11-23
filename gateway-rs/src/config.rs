use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub pg_dsn: String,
    pub valkey_url: String,
    pub openfga_url: String,
    pub nats_url: String,
    pub bind_addr: String,
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
            nats_url: env::var("CONVERGENT_NATS_URL")
                .unwrap_or_else(|_| "nats://127.0.0.1:54222".into()),
            bind_addr: env::var("CONVERGENT_BIND_ADDR")
                .unwrap_or_else(|_| "127.0.0.1:8090".into()),
        }
    }
}
