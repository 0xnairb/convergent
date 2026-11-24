use std::sync::Arc;

use anyhow::{bail, Result};
use serde_json::{json, Value};

use crate::kv_keys::derive_kv_key;
use crate::retrieval::KnowledgeBlock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Vllm,
    Sglang,
    Mlx,
    Ollama,
    OpenAi,
    Anthropic,
    Dev,
}

impl BackendKind {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "vllm" | "cuda" | "vllm-cuda" => Ok(Self::Vllm),
            "sglang" | "sglang-cuda" => Ok(Self::Sglang),
            "mlx" | "apple-silicon" | "metal" => Ok(Self::Mlx),
            "ollama" => Ok(Self::Ollama),
            "openai" | "openai-proxy" => Ok(Self::OpenAi),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "dev" | "simulator" => Ok(Self::Dev),
            other => bail!(
                "unsupported CONVERGENT_INFERENCE_BACKEND '{other}'; expected one of: vllm (cuda), sglang, mlx, ollama, openai, anthropic, dev"
            ),
        }
    }
}

/// Factory-created inference backend adapting Project Convergent's context & KV keys
/// to CUDA (vLLM / SGLang), Apple Silicon (MLX / Ollama), or local dev engines.
pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn hardware_target(&self) -> &'static str;
    fn supports_native_kv_keys(&self) -> bool;

    fn chat_url(&self, endpoint_url: &str) -> String {
        format!("{}/v1/chat/completions", endpoint_url.trim_end_matches('/'))
    }

    fn extra_headers(&self) -> Vec<(String, String)> {
        vec![]
    }

    fn prepare_chat_payload(
        &self,
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
        }
        if body.get("model").is_none() {
            body["model"] = json!(model_id);
        }
        self.apply_backend_options(&mut body, &kv_keys);
        (body, kv_keys)
    }

    fn apply_backend_options(&self, body: &mut Value, kv_keys: &[String]);
}

pub struct VllmBackend;
impl InferenceBackend for VllmBackend {
    fn name(&self) -> &'static str { "vllm" }
    fn hardware_target(&self) -> &'static str { "cuda" }
    fn supports_native_kv_keys(&self) -> bool { true }
    fn apply_backend_options(&self, body: &mut Value, kv_keys: &[String]) {
        if !kv_keys.is_empty() {
            body["extra_body"] = json!({
                "lmcache_blend": true,
                "kv_cache_keys": kv_keys
            });
        }
    }
}

pub struct SglangBackend;
impl InferenceBackend for SglangBackend {
    fn name(&self) -> &'static str { "sglang" }
    fn hardware_target(&self) -> &'static str { "cuda" }
    fn supports_native_kv_keys(&self) -> bool { true }
    fn apply_backend_options(&self, body: &mut Value, kv_keys: &[String]) {
        if !kv_keys.is_empty() {
            body["extra_body"] = json!({
                "radix_cache_keys": kv_keys,
                "kv_cache_keys": kv_keys
            });
        }
    }
}

pub struct MlxBackend;
impl InferenceBackend for MlxBackend {
    fn name(&self) -> &'static str { "mlx" }
    fn hardware_target(&self) -> &'static str { "apple-silicon-metal" }
    fn supports_native_kv_keys(&self) -> bool { false }
    fn apply_backend_options(&self, body: &mut Value, _kv_keys: &[String]) {
        // mlx_lm.server accepts standard OpenAI chat payloads; strip vLLM-only extra_body.
        if let Some(obj) = body.as_object_mut() {
            obj.remove("extra_body");
        }
    }
}

pub struct OllamaBackend;
impl InferenceBackend for OllamaBackend {
    fn name(&self) -> &'static str { "ollama" }
    fn hardware_target(&self) -> &'static str { "metal-or-cuda" }
    fn supports_native_kv_keys(&self) -> bool { false }
    fn apply_backend_options(&self, body: &mut Value, _kv_keys: &[String]) {
        // Ollama's OpenAI-compatible /v1/chat/completions endpoint rejects unknown extra_body keys.
        if let Some(obj) = body.as_object_mut() {
            obj.remove("extra_body");
        }
    }
}

pub struct OpenAiBackend;
impl InferenceBackend for OpenAiBackend {
    fn name(&self) -> &'static str { "openai" }
    fn hardware_target(&self) -> &'static str { "cloud-proxy" }
    fn supports_native_kv_keys(&self) -> bool { false }
    fn extra_headers(&self) -> Vec<(String, String)> {
        match std::env::var("OPENAI_API_KEY") {
            Ok(k) if !k.trim().is_empty() => vec![("authorization".into(), format!("Bearer {k}"))],
            _ => vec![],
        }
    }
    fn apply_backend_options(&self, body: &mut Value, kv_keys: &[String]) {
        if let Some(obj) = body.as_object_mut() {
            obj.remove("extra_body");
        }
        // Map branch/space HMAC KV key to OpenAI's prompt_cache_key so merges/reverts auto-bust cache.
        if let Some(first_key) = kv_keys.first() {
            body["prompt_cache_key"] = json!(first_key);
        }
    }
}

pub struct AnthropicBackend;
impl InferenceBackend for AnthropicBackend {
    fn name(&self) -> &'static str { "anthropic" }
    fn hardware_target(&self) -> &'static str { "cloud-proxy" }
    fn supports_native_kv_keys(&self) -> bool { false }
    fn chat_url(&self, endpoint_url: &str) -> String {
        format!("{}/v1/messages", endpoint_url.trim_end_matches('/'))
    }
    fn extra_headers(&self) -> Vec<(String, String)> {
        let mut hdrs = vec![("anthropic-version".into(), "2023-06-01".into())];
        if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
            if !k.trim().is_empty() {
                hdrs.push(("x-api-key".into(), k));
            }
        }
        hdrs
    }
    fn apply_backend_options(&self, body: &mut Value, _kv_keys: &[String]) {
        if let Some(obj) = body.as_object_mut() {
            obj.remove("extra_body");
        }
        // Convert leading system message into Anthropic's cache_control: ephemeral block.
        if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
            if let Some(first) = messages.first() {
                if first.get("role").and_then(|r| r.as_str()) == Some("system") {
                    let sys_text = first.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
                    messages.remove(0);
                    body["system"] = json!([{
                        "type": "text",
                        "text": sys_text,
                        "cache_control": { "type": "ephemeral" }
                    }]);
                }
            }
        }
        if body.get("max_tokens").is_none() {
            body["max_tokens"] = json!(1024);
        }
    }
}

pub struct DevBackend;
impl InferenceBackend for DevBackend {
    fn name(&self) -> &'static str { "dev" }
    fn hardware_target(&self) -> &'static str { "local-simulator" }
    fn supports_native_kv_keys(&self) -> bool { true }
    fn apply_backend_options(&self, body: &mut Value, kv_keys: &[String]) {
        if !kv_keys.is_empty() {
            body["extra_body"] = json!({
                "lmcache_blend": true,
                "kv_cache_keys": kv_keys
            });
        }
    }
}

pub fn create_backend(kind: BackendKind) -> Arc<dyn InferenceBackend> {
    match kind {
        BackendKind::Vllm => Arc::new(VllmBackend),
        BackendKind::Sglang => Arc::new(SglangBackend),
        BackendKind::Mlx => Arc::new(MlxBackend),
        BackendKind::Ollama => Arc::new(OllamaBackend),
        BackendKind::OpenAi => Arc::new(OpenAiBackend),
        BackendKind::Anthropic => Arc::new(AnthropicBackend),
        BackendKind::Dev => Arc::new(DevBackend),
    }
}

pub fn create_backend_from_str(raw: &str) -> Result<Arc<dyn InferenceBackend>> {
    Ok(create_backend(BackendKind::parse(raw)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn sample_block() -> KnowledgeBlock {
        KnowledgeBlock {
            id: Uuid::new_v4(),
            document_id: Uuid::new_v4(),
            clearance_scope: "space:alpha".into(),
            canonical_text: "Travel cap is $1200.".into(),
            content_blake3: "abc123".into(),
            citations: json!([]),
            score: 1.0,
        }
    }

    #[test]
    fn factory_resolves_all_supported_backends() {
        assert_eq!(create_backend_from_str("cuda").unwrap().name(), "vllm");
        assert_eq!(create_backend_from_str("vllm").unwrap().name(), "vllm");
        assert_eq!(create_backend_from_str("sglang").unwrap().name(), "sglang");
        assert_eq!(create_backend_from_str("mlx").unwrap().name(), "mlx");
        assert_eq!(create_backend_from_str("ollama").unwrap().name(), "ollama");
        assert_eq!(create_backend_from_str("openai").unwrap().name(), "openai");
        assert_eq!(create_backend_from_str("anthropic").unwrap().name(), "anthropic");
        assert_eq!(create_backend_from_str("dev").unwrap().name(), "dev");
        assert!(create_backend_from_str("unknown-engine").is_err());
    }

    #[test]
    fn payload_adaptation_matches_backend_capabilities() {
        let blocks = vec![sample_block()];
        let base = json!({ "messages": [{ "role": "user", "content": "hi" }] });

        let (vllm_payload, vllm_keys) = create_backend(BackendKind::Vllm)
            .prepare_chat_payload(base.clone(), "Qwen3-32B", b"seed", &blocks);
        assert_eq!(vllm_keys.len(), 1);
        assert!(vllm_payload["extra_body"]["lmcache_blend"].as_bool().unwrap());
        assert_eq!(vllm_payload["extra_body"]["kv_cache_keys"][0], vllm_keys[0]);

        let (mlx_payload, mlx_keys) = create_backend(BackendKind::Mlx)
            .prepare_chat_payload(base.clone(), "mlx-community/Qwen3-8B-4bit", b"seed", &blocks);
        assert_eq!(mlx_keys.len(), 1);
        assert!(mlx_payload.get("extra_body").is_none());
        assert_eq!(mlx_payload["messages"][0]["role"], "system");

        let (openai_payload, openai_keys) = create_backend(BackendKind::OpenAi)
            .prepare_chat_payload(base.clone(), "gpt-5", b"seed", &blocks);
        assert_eq!(openai_payload["prompt_cache_key"], openai_keys[0]);
        assert!(openai_payload.get("extra_body").is_none());

        let (anthropic_payload, _) = create_backend(BackendKind::Anthropic)
            .prepare_chat_payload(base.clone(), "claude-sonnet-4", b"seed", &blocks);
        assert_eq!(anthropic_payload["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(anthropic_payload["messages"][0]["role"], "user");

        let (ollama_payload, ollama_keys) = create_backend(BackendKind::Ollama)
            .prepare_chat_payload(base, "qwen3:8b", b"seed", &blocks);
        assert_eq!(ollama_keys.len(), 1);
        assert!(ollama_payload.get("extra_body").is_none());
    }
}
