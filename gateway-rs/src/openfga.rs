use anyhow::{anyhow, Context, Result};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

#[derive(Clone)]
pub struct OpenFgaClient {
    base_url: String,
    http: reqwest::Client,
}

#[derive(Deserialize)]
struct ListStoresResp {
    stores: Vec<StoreItem>,
}

#[derive(Deserialize)]
struct StoreItem {
    id: String,
    name: String,
}

#[derive(Deserialize)]
struct ListObjectsResp {
    objects: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct CachedDocIds {
    doc_ids: Vec<Uuid>,
}

impl OpenFgaClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub async fn resolve_store_id(&self, store_name: &str) -> Result<String> {
        let resp: ListStoresResp = self
            .http
            .get(format!("{}/stores", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        resp.stores
            .into_iter()
            .find(|s| s.name == store_name)
            .map(|s| s.id)
            .ok_or_else(|| anyhow!("OpenFGA store '{store_name}' not found"))
    }

    pub async fn write_tuple(
        &self,
        store_id: &str,
        user: &str,
        relation: &str,
        object: &str,
    ) -> Result<()> {
        self.http
            .post(format!("{}/stores/{store_id}/write", self.base_url))
            .json(&json!({
                "writes": {
                    "tuple_keys": [{ "user": user, "relation": relation, "object": object }]
                }
            }))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn delete_tuple(
        &self,
        store_id: &str,
        user: &str,
        relation: &str,
        object: &str,
    ) -> Result<()> {
        self.http
            .post(format!("{}/stores/{store_id}/write", self.base_url))
            .json(&json!({
                "deletes": {
                    "tuple_keys": [{ "user": user, "relation": relation, "object": object }]
                }
            }))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Resolve allowed document UUIDs for `user_id` (`user:<id>`), with short-TTL Valkey caching.
    pub async fn list_allowed_documents(
        &self,
        store_id: &str,
        user_id: &str,
        valkey: &mut redis::aio::MultiplexedConnection,
        ttl_secs: u64,
    ) -> Result<Vec<Uuid>> {
        let cache_key = format!("cvg:acl:{store_id}:{user_id}");
        if ttl_secs > 0 {
            if let Ok(Some(raw)) = valkey.get::<_, Option<String>>(&cache_key).await {
                if let Ok(parsed) = serde_json::from_str::<CachedDocIds>(&raw) {
                    return Ok(parsed.doc_ids);
                }
            }
        }

        let resp: ListObjectsResp = self
            .http
            .post(format!("{}/stores/{store_id}/list-objects", self.base_url))
            .json(&json!({
                "user": format!("user:{user_id}"),
                "relation": "viewer",
                "type": "document"
            }))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
            .context("parse OpenFGA list-objects response")?;

        let mut doc_ids = Vec::with_capacity(resp.objects.len());
        for obj in resp.objects {
            if let Some(id_str) = obj.strip_prefix("document:") {
                if let Ok(uid) = Uuid::parse_str(id_str) {
                    doc_ids.push(uid);
                }
            }
        }

        if ttl_secs > 0 {
            let encoded = serde_json::to_string(&CachedDocIds {
                doc_ids: doc_ids.clone(),
            })?;
            let _: () = valkey.set_ex(&cache_key, encoded, ttl_secs).await?;
        }
        Ok(doc_ids)
    }

    pub async fn invalidate_cache(
        store_id: &str,
        user_id: &str,
        valkey: &mut redis::aio::MultiplexedConnection,
    ) -> Result<()> {
        let cache_key = format!("cvg:acl:{store_id}:{user_id}");
        let _: () = valkey.del(&cache_key).await?;
        Ok(())
    }
}
