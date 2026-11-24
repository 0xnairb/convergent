use anyhow::Result;
use serde_json::{json, Value};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::kv_keys::content_blake3;
use crate::openfga::OpenFgaClient;
use crate::retrieval::KnowledgeBlock;

pub async fn current_head_rev(pool: &PgPool, org_id: Uuid, space_id: Uuid) -> Result<i64> {
    let row = sqlx::query(
        "SELECT COALESCE(MAX(rev), 0)::bigint AS rev FROM space_revisions WHERE org_id = $1 AND space_id = $2",
    )
    .bind(org_id)
    .bind(space_id)
    .fetch_one(pool)
    .await?;
    Ok(row.get("rev"))
}

/// Create a 0-byte virtual overlay pinned to `base_rev` (copies ZERO knowledge blocks).
pub async fn open_overlay(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Uuid,
    owner: &str,
    base_rev: Option<i64>,
) -> Result<Value> {
    let rev = match base_rev {
        Some(r) => r,
        None => current_head_rev(pool, org_id, space_id).await?,
    };
    let overlay_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO overlays (id, org_id, space_id, base_rev, owner, status) VALUES ($1,$2,$3,$4,$5,'open')",
    )
    .bind(overlay_id)
    .bind(org_id)
    .bind(space_id)
    .bind(rev)
    .bind(owner)
    .execute(pool)
    .await?;

    Ok(json!({
        "overlay_id": overlay_id,
        "space_id": space_id,
        "base_rev": rev,
        "owner": owner,
        "status": "open"
    }))
}

pub async fn stage_delta(
    pool: &PgPool,
    overlay_id: Uuid,
    source_uri: &str,
    text: &str,
    op: &str,
) -> Result<Value> {
    if op != "upsert" && op != "retract" {
        anyhow::bail!("op must be 'upsert' or 'retract'");
    }
    let trimmed = if op == "upsert" { text.trim() } else { "" };
    let bhash = if trimmed.is_empty() { String::new() } else { content_blake3(trimmed) };
    let zero_vec = format!("[{}]", vec!["0"; 64].join(","));
    let citations = if trimmed.is_empty() {
        json!([])
    } else {
        json!([{
            "source_uri": source_uri,
            "offset": 0,
            "length": trimmed.len(),
            "quote_hash": &bhash[..16.min(bhash.len())]
        }])
    };
    let delta_id = Uuid::new_v4();

    sqlx::query(
        r#"
        INSERT INTO overlay_deltas (id, overlay_id, op, source_uri, block_type, canonical_text, content_blake3, citations, embedding)
        VALUES ($1,$2,$3,$4,'fact',$5,$6,$7,$8::vector)
        ON CONFLICT (overlay_id, source_uri) DO UPDATE
          SET op = EXCLUDED.op,
              canonical_text = EXCLUDED.canonical_text,
              content_blake3 = EXCLUDED.content_blake3,
              citations = EXCLUDED.citations,
              embedding = EXCLUDED.embedding,
              updated_at = now()
        "#,
    )
    .bind(delta_id)
    .bind(overlay_id)
    .bind(op)
    .bind(source_uri)
    .bind(trimmed)
    .bind(&bhash)
    .bind(&citations)
    .bind(&zero_vec)
    .execute(pool)
    .await?;

    Ok(json!({
        "delta_id": delta_id,
        "overlay_id": overlay_id,
        "op": op,
        "source_uri": source_uri
    }))
}

pub async fn inspect_overlay(pool: &PgPool, overlay_id: Uuid) -> Result<Value> {
    let ov = sqlx::query("SELECT space_id, base_rev, owner, status, landed_rev FROM overlays WHERE id = $1")
        .bind(overlay_id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| anyhow::anyhow!("overlay {overlay_id} not found"))?;

    let rows = sqlx::query(
        "SELECT id, op, source_uri, canonical_text, content_blake3 FROM overlay_deltas WHERE overlay_id = $1 ORDER BY source_uri ASC",
    )
    .bind(overlay_id)
    .fetch_all(pool)
    .await?;

    let deltas: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "id": r.get::<Uuid, _>("id"),
                "op": r.get::<String, _>("op"),
                "source_uri": r.get::<String, _>("source_uri"),
                "canonical_text": r.get::<String, _>("canonical_text"),
                "content_blake3": r.get::<String, _>("content_blake3"),
            })
        })
        .collect();

    Ok(json!({
        "overlay_id": overlay_id,
        "space_id": ov.get::<Uuid, _>("space_id"),
        "base_rev": ov.get::<i64, _>("base_rev"),
        "owner": ov.get::<String, _>("owner"),
        "status": ov.get::<String, _>("status"),
        "landed_rev": ov.get::<Option<i64>, _>("landed_rev"),
        "deltas": deltas
    }))
}

/// Zero-copy Mount Stack composition: merges base authorized blocks with an open overlay's delta.
/// Untouched base blocks keep their shared `space:{id}` clearance_scope for 100% KV cache reuse.
pub async fn apply_overlay_to_blocks(
    pool: &PgPool,
    overlay_id: Uuid,
    base_blocks: Vec<KnowledgeBlock>,
) -> Result<Vec<KnowledgeBlock>> {
    let delta_rows = sqlx::query(
        "SELECT id, op, source_uri, canonical_text, content_blake3, citations FROM overlay_deltas WHERE overlay_id = $1",
    )
    .bind(overlay_id)
    .fetch_all(pool)
    .await?;

    if delta_rows.is_empty() {
        return Ok(base_blocks);
    }

    let mut overridden_uris = std::collections::HashSet::<String>::new();
    let mut overlay_upserts = Vec::new();
    for r in delta_rows {
        let op: String = r.get("op");
        let uri: String = r.get("source_uri");
        overridden_uris.insert(uri);
        if op == "upsert" {
            overlay_upserts.push(KnowledgeBlock {
                id: r.get("id"),
                document_id: overlay_id,
                clearance_scope: format!("overlay:{overlay_id}"),
                canonical_text: r.get("canonical_text"),
                content_blake3: r.get("content_blake3"),
                citations: r.get("citations"),
                score: 10.0,
            });
        }
    }

    // Filter out base blocks whose source_uri is overridden/retracted by the overlay
    let mut composed = Vec::with_capacity(base_blocks.len() + overlay_upserts.len());
    composed.extend(overlay_upserts);
    for blk in base_blocks {
        let blk_uri = blk
            .citations
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("source_uri"))
            .and_then(|u| u.as_str())
            .unwrap_or("");
        if !blk_uri.is_empty() && overridden_uris.contains(blk_uri) {
            continue;
        }
        composed.push(blk);
    }
    Ok(composed)
}

#[allow(clippy::too_many_arguments)]
pub async fn land_overlay(
    pool: &PgPool,
    openfga: &OpenFgaClient,
    openfga_store_id: &str,
    valkey: &mut redis::aio::MultiplexedConnection,
    overlay_id: Uuid,
    author: &str,
    message: &str,
    force: bool,
) -> Result<Value> {
    let mut tx = pool.begin().await?;
    let ov = sqlx::query("SELECT org_id, space_id, base_rev, status FROM overlays WHERE id = $1 FOR UPDATE")
        .bind(overlay_id)
        .fetch_optional(&mut *tx)
        .await?
        .ok_or_else(|| anyhow::anyhow!("overlay {overlay_id} not found"))?;
    let status: String = ov.get("status");
    if status != "open" {
        anyhow::bail!("overlay {overlay_id} is not open (status={status})");
    }
    let org_id: Uuid = ov.get("org_id");
    let space_id: Uuid = ov.get("space_id");
    let base_rev: i64 = ov.get("base_rev");

    sqlx::query("SELECT pg_advisory_xact_lock(hashtext($1))")
        .bind(format!("{org_id}:{space_id}"))
        .execute(&mut *tx)
        .await?;

    let deltas = sqlx::query(
        "SELECT op, source_uri, block_type, canonical_text, content_blake3, citations, embedding::text AS emb_text FROM overlay_deltas WHERE overlay_id = $1 ORDER BY source_uri ASC",
    )
    .bind(overlay_id)
    .fetch_all(&mut *tx)
    .await?;

    if !force && !deltas.is_empty() {
        let uris: Vec<String> = deltas.iter().map(|d| d.get("source_uri")).collect();
        let conflict_rows = sqlx::query(
            r#"
            SELECT d.source_uri, kb.landed_rev, kb.canonical_text
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = $1 AND kb.space_id = $2
              AND d.source_uri = ANY($3)
              AND kb.landed_rev > $4
              AND kb.valid_to IS NULL
            "#,
        )
        .bind(org_id)
        .bind(space_id)
        .bind(&uris)
        .bind(base_rev)
        .fetch_all(&mut *tx)
        .await?;

        if !conflict_rows.is_empty() {
            let conflicts: Vec<Value> = conflict_rows
                .into_iter()
                .map(|r| {
                    json!({
                        "source_uri": r.get::<String, _>("source_uri"),
                        "current_rev": r.get::<i64, _>("landed_rev"),
                        "current_text": r.get::<String, _>("canonical_text"),
                    })
                })
                .collect();
            tx.rollback().await?;
            return Ok(json!({ "status": "conflict", "conflicts": conflicts }));
        }
    }

    let head_row = sqlx::query(
        "SELECT COALESCE(MAX(rev), 0)::bigint AS rev FROM space_revisions WHERE org_id = $1 AND space_id = $2",
    )
    .bind(org_id)
    .bind(space_id)
    .fetch_one(&mut *tx)
    .await?;
    let new_rev = head_row.get::<i64, _>("rev") + 1;

    sqlx::query(
        "INSERT INTO space_revisions (org_id, space_id, rev, author, message, action) VALUES ($1,$2,$3,$4,$5,'land')",
    )
    .bind(org_id)
    .bind(space_id)
    .bind(new_rev)
    .bind(author)
    .bind(message)
    .execute(&mut *tx)
    .await?;

    let space_scope = format!("space:{space_id}");
    let mut new_doc_ids = Vec::new();
    let mut applied = 0usize;
    let mut superseded = 0usize;

    for d in deltas {
        let op: String = d.get("op");
        let uri: String = d.get("source_uri");
        let btype: String = d.get("block_type");
        let text: String = d.get("canonical_text");
        let bhash: String = d.get("content_blake3");
        let cits: Value = d.get("citations");
        let emb_text: Option<String> = d.get("emb_text");

        let prior = sqlx::query(
            r#"
            SELECT kb.id, kb.content_blake3, kb.clearance_scope
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = $1 AND kb.space_id = $2 AND d.source_uri = $3 AND kb.valid_to IS NULL
            FOR UPDATE OF kb
            "#,
        )
        .bind(org_id)
        .bind(space_id)
        .bind(&uri)
        .fetch_all(&mut *tx)
        .await?;

        let mut new_bid: Option<Uuid> = None;
        if op == "upsert" {
            let doc_id = Uuid::new_v4();
            let bid = Uuid::new_v4();
            let vec_lit = emb_text.unwrap_or_else(|| format!("[{}]", vec!["0"; 64].join(",")));
            sqlx::query(
                "INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3) VALUES ($1,$2,$3,$4,$5,$6)",
            )
            .bind(doc_id)
            .bind(org_id)
            .bind(space_id)
            .bind(&space_scope)
            .bind(&uri)
            .bind(&bhash)
            .execute(&mut *tx)
            .await?;

            sqlx::query(
                r#"
                INSERT INTO knowledge_blocks (
                  id, document_id, org_id, space_id, clearance_scope, landed_rev,
                  block_type, canonical_text, content_blake3, citations, embedding
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::vector)
                "#,
            )
            .bind(bid)
            .bind(doc_id)
            .bind(org_id)
            .bind(space_id)
            .bind(&space_scope)
            .bind(new_rev)
            .bind(&btype)
            .bind(&text)
            .bind(&bhash)
            .bind(&cits)
            .bind(&vec_lit)
            .execute(&mut *tx)
            .await?;

            new_doc_ids.push(doc_id);
            new_bid = Some(bid);
            applied += 1;
        }

        for p in prior {
            let old_id: Uuid = p.get("id");
            let old_hash: String = p.get("content_blake3");
            let old_scope: String = p.get("clearance_scope");
            sqlx::query("UPDATE knowledge_blocks SET valid_to = now(), superseded_by = $1 WHERE id = $2")
                .bind(new_bid)
                .bind(old_id)
                .execute(&mut *tx)
                .await?;
            superseded += 1;
            emit_eviction_outbox(&mut tx, org_id, space_id, &old_scope, old_id, &old_hash, new_bid, &bhash, &format!("land:r{new_rev}")).await?;
        }
    }

    sqlx::query("UPDATE overlays SET status = 'landed', landed_rev = $1, updated_at = now() WHERE id = $2")
        .bind(new_rev)
        .bind(overlay_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;

    // Register OpenFGA document -> space tuples and clear ACL cache for immediate visibility
    for doc_id in new_doc_ids {
        let _ = openfga
            .write_tuple(openfga_store_id, &format!("space:{space_id}"), "space", &format!("document:{doc_id}"))
            .await;
    }
    let _ = OpenFgaClient::invalidate_cache(openfga_store_id, author, valkey).await;

    Ok(json!({
        "status": "landed",
        "rev": new_rev,
        "applied_blocks": applied,
        "superseded_blocks": superseded
    }))
}

pub async fn undo_revision(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Uuid,
    rev: i64,
    author: &str,
) -> Result<Value> {
    let mut tx = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock(hashtext($1))")
        .bind(format!("{org_id}:{space_id}"))
        .execute(&mut *tx)
        .await?;

    let head_row = sqlx::query(
        "SELECT COALESCE(MAX(rev), 0)::bigint AS rev FROM space_revisions WHERE org_id = $1 AND space_id = $2",
    )
    .bind(org_id)
    .bind(space_id)
    .fetch_one(&mut *tx)
    .await?;
    let undo_rev = head_row.get::<i64, _>("rev") + 1;

    sqlx::query(
        "INSERT INTO space_revisions (org_id, space_id, rev, author, message, action) VALUES ($1,$2,$3,$4,$5,'undo')",
    )
    .bind(org_id)
    .bind(space_id)
    .bind(undo_rev)
    .bind(author)
    .bind(format!("Undo revision r{rev}"))
    .execute(&mut *tx)
    .await?;

    let bad_blocks = sqlx::query(
        "SELECT id, content_blake3, clearance_scope FROM knowledge_blocks WHERE org_id = $1 AND space_id = $2 AND landed_rev = $3 AND valid_to IS NULL FOR UPDATE",
    )
    .bind(org_id)
    .bind(space_id)
    .bind(rev)
    .fetch_all(&mut *tx)
    .await?;

    let mut deactivated = 0usize;
    let mut restored = 0usize;
    for b in bad_blocks {
        let bad_id: Uuid = b.get("id");
        let bad_hash: String = b.get("content_blake3");
        let bad_scope: String = b.get("clearance_scope");
        sqlx::query("UPDATE knowledge_blocks SET valid_to = now() WHERE id = $1")
            .bind(bad_id)
            .execute(&mut *tx)
            .await?;
        deactivated += 1;

        let prev_rows = sqlx::query(
            "SELECT id, content_blake3 FROM knowledge_blocks WHERE org_id = $1 AND space_id = $2 AND superseded_by = $3 FOR UPDATE",
        )
        .bind(org_id)
        .bind(space_id)
        .bind(bad_id)
        .fetch_all(&mut *tx)
        .await?;

        for p in prev_rows {
            let prev_id: Uuid = p.get("id");
            let prev_hash: String = p.get("content_blake3");
            sqlx::query("UPDATE knowledge_blocks SET valid_to = NULL, superseded_by = NULL WHERE id = $1")
                .bind(prev_id)
                .execute(&mut *tx)
                .await?;
            restored += 1;
            emit_eviction_outbox(&mut tx, org_id, space_id, &bad_scope, bad_id, &bad_hash, Some(prev_id), &prev_hash, &format!("undo:r{rev}")).await?;
        }
    }

    tx.commit().await?;
    Ok(json!({
        "undo_rev": undo_rev,
        "undone_rev": rev,
        "deactivated_blocks": deactivated,
        "restored_blocks": restored
    }))
}

#[allow(clippy::too_many_arguments)]
async fn emit_eviction_outbox(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    org_id: Uuid,
    space_id: Uuid,
    clearance_scope: &str,
    old_id: Uuid,
    old_hash: &str,
    new_id: Option<Uuid>,
    new_hash: &str,
    reason: &str,
) -> Result<()> {
    let tag = new_id.map(|u| u.to_string()).unwrap_or_else(|| "retracted".into());
    let msg_id = format!("superseded:{old_id}:{tag}:{}", &Uuid::new_v4().simple().to_string()[..8]);
    let payload = json!({
        "tenant_id": org_id,
        "space_id": space_id,
        "clearance_scope": clearance_scope,
        "old_block_id": old_id,
        "old_block_hash": old_hash,
        "new_block_id": new_id,
        "new_block_hash": new_hash,
        "reason": reason
    });
    sqlx::query(
        "INSERT INTO outbox_events (msg_id, subject, payload) VALUES ($1, 'convergent.memory.superseded', $2) ON CONFLICT (msg_id) DO NOTHING",
    )
    .bind(msg_id)
    .bind(payload)
    .execute(&mut **tx)
    .await?;
    Ok(())
}
