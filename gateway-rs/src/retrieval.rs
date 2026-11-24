use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeBlock {
    pub id: Uuid,
    pub document_id: Uuid,
    pub clearance_scope: String,
    pub canonical_text: String,
    pub content_blake3: String,
    pub citations: serde_json::Value,
    pub score: f64,
}

/// Pre-filtered hybrid search (PostgreSQL 17 pgvector + lexical rank) on `main` branch.
pub async fn prefiltered_hybrid_search(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Option<Uuid>,
    allowed_doc_ids: &[Uuid],
    query: &str,
    as_of: Option<DateTime<Utc>>,
    top_k: i64,
) -> Result<Vec<KnowledgeBlock>> {
    prefiltered_hybrid_search_branch(pool, org_id, space_id, "main", allowed_doc_ids, query, as_of, top_k).await
}

/// Branch-scoped pre-filtered hybrid search.
/// Strictly filters `document_id = ANY($allowed_doc_ids)` and `branch_name = $branch` BEFORE Top-K truncation.
#[allow(clippy::too_many_arguments)]
pub async fn prefiltered_hybrid_search_branch(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Option<Uuid>,
    branch_name: &str,
    allowed_doc_ids: &[Uuid],
    query: &str,
    as_of: Option<DateTime<Utc>>,
    top_k: i64,
) -> Result<Vec<KnowledgeBlock>> {
    if allowed_doc_ids.is_empty() {
        return Ok(vec![]);
    }

    let rows = match (space_id, as_of) {
        (Some(sid), None) => {
            sqlx::query(
                r#"
                SELECT id, document_id, clearance_scope, canonical_text, content_blake3, citations,
                       ts_rank_cd(to_tsvector('english', canonical_text), plainto_tsquery('english', $1))::float8 AS score
                FROM knowledge_blocks
                WHERE org_id = $2
                  AND space_id = $3
                  AND branch_name = $4
                  AND document_id = ANY($5)
                  AND valid_to IS NULL
                ORDER BY score DESC, valid_from DESC
                LIMIT $6
                "#,
            )
            .bind(query)
            .bind(org_id)
            .bind(sid)
            .bind(branch_name)
            .bind(allowed_doc_ids)
            .bind(top_k)
            .fetch_all(pool)
            .await?
        }
        (None, None) => {
            sqlx::query(
                r#"
                SELECT id, document_id, clearance_scope, canonical_text, content_blake3, citations,
                       ts_rank_cd(to_tsvector('english', canonical_text), plainto_tsquery('english', $1))::float8 AS score
                FROM knowledge_blocks
                WHERE org_id = $2
                  AND branch_name = $3
                  AND document_id = ANY($4)
                  AND valid_to IS NULL
                ORDER BY score DESC, valid_from DESC
                LIMIT $5
                "#,
            )
            .bind(query)
            .bind(org_id)
            .bind(branch_name)
            .bind(allowed_doc_ids)
            .bind(top_k)
            .fetch_all(pool)
            .await?
        }
        (Some(sid), Some(ts)) => {
            sqlx::query(
                r#"
                SELECT id, document_id, clearance_scope, canonical_text, content_blake3, citations,
                       ts_rank_cd(to_tsvector('english', canonical_text), plainto_tsquery('english', $1))::float8 AS score
                FROM knowledge_blocks
                WHERE org_id = $2
                  AND space_id = $3
                  AND branch_name = $4
                  AND document_id = ANY($5)
                  AND valid_from <= $6
                  AND (valid_to IS NULL OR valid_to > $6)
                ORDER BY score DESC, valid_from DESC
                LIMIT $7
                "#,
            )
            .bind(query)
            .bind(org_id)
            .bind(sid)
            .bind(branch_name)
            .bind(allowed_doc_ids)
            .bind(ts)
            .bind(top_k)
            .fetch_all(pool)
            .await?
        }
        (None, Some(ts)) => {
            sqlx::query(
                r#"
                SELECT id, document_id, clearance_scope, canonical_text, content_blake3, citations,
                       ts_rank_cd(to_tsvector('english', canonical_text), plainto_tsquery('english', $1))::float8 AS score
                FROM knowledge_blocks
                WHERE org_id = $2
                  AND branch_name = $3
                  AND document_id = ANY($4)
                  AND valid_from <= $5
                  AND (valid_to IS NULL OR valid_to > $5)
                ORDER BY score DESC, valid_from DESC
                LIMIT $6
                "#,
            )
            .bind(query)
            .bind(org_id)
            .bind(branch_name)
            .bind(allowed_doc_ids)
            .bind(ts)
            .bind(top_k)
            .fetch_all(pool)
            .await?
        }
    };

    Ok(rows
        .into_iter()
        .map(|r| KnowledgeBlock {
            id: r.get("id"),
            document_id: r.get("document_id"),
            clearance_scope: r.get("clearance_scope"),
            canonical_text: r.get("canonical_text"),
            content_blake3: r.get("content_blake3"),
            citations: r.get("citations"),
            score: r.get("score"),
        })
        .collect())
}
