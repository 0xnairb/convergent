use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};

use crate::kv_keys::derive_kv_key;
use crate::retrieval::KnowledgeBlock;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveModel {
    pub model_id: String,
    pub endpoint_url: String,
}

pub async fn get_active_model(pool: &PgPool, fallback_url: &str, fallback_model: &str) -> ActiveModel {
    if let Ok(Some(row)) = sqlx::query(
        "SELECT model_id, endpoint_url FROM model_registry WHERE is_active = true LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    {
        return ActiveModel {
            model_id: row.get("model_id"),
            endpoint_url: row.get("endpoint_url"),
        };
    }
    ActiveModel {
        model_id: fallback_model.to_string(),
        endpoint_url: fallback_url.to_string(),
    }
}

/// Build the OpenAI-compatible payload with scoped context blocks and per-clearance HMAC KV keys.
pub fn build_blended_payload(
    mut body: Value,
    model_id: &str,
    master_seed: &[u8],
    blocks: &[KnowledgeBlock],
) -> (Value, Vec<String>) {
    let mut kv_keys = Vec::with_capacity(blocks.len());
    if !blocks.is_empty() {
        let mut context_str = String::from("Authorized Context Blocks:\n");
        for b in blocks {
            let key = derive_kv_key(master_seed, &b.clearance_scope, model_id, &b.content_blake3);
            kv_keys.push(key);
            context_str.push_str(&format!("\n[block:{}]\n{}\n", b.content_blake3, b.canonical_text));
        }

        if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
            messages.insert(
                0,
                json!({
                    "role": "system",
                    "content": context_str
                }),
            );
        }
        body["extra_body"] = json!({
            "lmcache_blend": true,
            "kv_cache_keys": kv_keys
        });
    }
    if body.get("model").is_none() {
        body["model"] = json!(model_id);
    }
    (body, kv_keys)
}

pub async fn forward_completion(
    http: &reqwest::Client,
    endpoint_url: &str,
    payload: &Value,
) -> Result<reqwest::Response> {
    let url = format!("{}/v1/chat/completions", endpoint_url.trim_end_matches('/'));
    let resp = http.post(url).json(payload).send().await?.error_for_status()?;
    Ok(resp)
}
