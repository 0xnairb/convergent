FROM rust:1.85-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --bin convergent-gateway

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/convergent-gateway /usr/local/bin/convergent-gateway
EXPOSE 8090
ENV CONVERGENT_BIND_ADDR=0.0.0.0:8090
CMD ["/usr/local/bin/convergent-gateway"]
