use convergent_gateway::config::Config;
use redis::AsyncCommands;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

#[tokio::test]
async fn smoke_postgres_valkey_openfga_nats() -> anyhow::Result<()> {
    let cfg = Config::from_env();

    // 1. Postgres round-trip
    let pool = PgPoolOptions::new()
        .max_connections(2)
        .connect(&cfg.pg_dsn)
        .await?;
    let val: (i32,) = sqlx::query_as("SELECT 1").fetch_one(&pool).await?;
    assert_eq!(val.0, 1);

    // 2. Valkey round-trip
    let client = redis::Client::open(cfg.valkey_url.as_str())?;
    let mut con = client.get_multiplexed_async_connection().await?;
    let key = format!("cvg:rs-smoke:{}", Uuid::new_v4());
    let _: () = con.set_ex(&key, "ok", 30).await?;
    let got: String = con.get(&key).await?;
    assert_eq!(got, "ok");
    let _: () = con.del(&key).await?;

    // 3. OpenFGA reachable
    let http = reqwest::Client::new();
    let resp = http
        .get(format!("{}/stores", cfg.openfga_url))
        .send()
        .await?
        .error_for_status()?;
    let body: serde_json::Value = resp.json().await?;
    assert!(body["stores"].is_array());

    // 4. NATS JetStream reachable
    let nc = async_nats::connect(&cfg.nats_url).await?;
    let js = async_nats::jetstream::new(nc);
    let stream = js.get_stream("CONVERGENT_MEMORY").await?;
    assert_eq!(stream.cached_info().config.name, "CONVERGENT_MEMORY");

    Ok(())
}
