use convergent_gateway::{config::Config, health_router};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cfg = Config::from_env();
    let listener = TcpListener::bind(&cfg.bind_addr).await?;
    tracing::info!("convergent-gateway listening on {}", cfg.bind_addr);
    axum::serve(listener, health_router()).await?;
    Ok(())
}
