use anyhow::Result;
use serde_json::{json, Value};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::kv_keys::content_blake3;
use crate::openfga::OpenFgaClient;

pub fn branch_clearance_scope(space_id: Uuid, branch_name: &str) -> String {
    if branch_name == "main" {
        format!("space:{space_id}")
    } else {
        format!("space:{space_id}:branch:{branch_name}")
    }
}

pub async fn ensure_branch(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Uuid,
    branch_name: &str,
    author: &str,
) -> Result<Uuid> {
    if let Some(row) = sqlx::query(
        "SELECT head_commit_id FROM memory_branches WHERE org_id = $1 AND space_id = $2 AND name = $3",
    )
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .fetch_optional(pool)
    .await?
    {
        if let Some(head) = row.get::<Option<Uuid>, _>("head_commit_id") {
            return Ok(head);
        }
    }

    let init_id = Uuid::new_v4();
    let mut tx = pool.begin().await?;
    sqlx::query(
        "INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action) VALUES ($1,$2,$3,$4,NULL,$5,$6,'init')",
    )
    .bind(init_id)
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .bind(author)
    .bind(format!("Initialize branch {branch_name}"))
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        "INSERT INTO memory_branches (id, org_id, space_id, name, head_commit_id, created_by) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (org_id, space_id, name) DO UPDATE SET head_commit_id = COALESCE(memory_branches.head_commit_id, EXCLUDED.head_commit_id)",
    )
    .bind(Uuid::new_v4())
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .bind(init_id)
    .bind(author)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(init_id)
}

pub async fn create_branch(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Uuid,
    new_branch: &str,
    from_branch: &str,
    author: &str,
) -> Result<Uuid> {
    let parent_head = ensure_branch(pool, org_id, space_id, from_branch, author).await?;
    let branch_commit_id = Uuid::new_v4();
    let new_scope = branch_clearance_scope(space_id, new_branch);

    let mut tx = pool.begin().await?;
    sqlx::query(
        "INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action) VALUES ($1,$2,$3,$4,$5,$6,$7,'branch')",
    )
    .bind(branch_commit_id)
    .bind(org_id)
    .bind(space_id)
    .bind(new_branch)
    .bind(parent_head)
    .bind(author)
    .bind(format!("Branch {new_branch} from {from_branch}"))
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        "INSERT INTO memory_branches (id, org_id, space_id, name, head_commit_id, created_by) VALUES ($1,$2,$3,$4,$5,$6)",
    )
    .bind(Uuid::new_v4())
    .bind(org_id)
    .bind(space_id)
    .bind(new_branch)
    .bind(branch_commit_id)
    .bind(author)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        r#"
        INSERT INTO knowledge_blocks (
          id, document_id, org_id, space_id, clearance_scope, branch_name, commit_id,
          block_type, canonical_text, content_blake3, citations, embedding
        )
        SELECT gen_random_uuid(), document_id, org_id, space_id, $1, $2, $3,
               block_type, canonical_text, content_blake3, citations, embedding
        FROM knowledge_blocks
        WHERE org_id = $4 AND space_id = $5 AND branch_name = $6 AND valid_to IS NULL
        "#,
    )
    .bind(&new_scope)
    .bind(new_branch)
    .bind(branch_commit_id)
    .bind(org_id)
    .bind(space_id)
    .bind(from_branch)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(branch_commit_id)
}

#[allow(clippy::too_many_arguments)]
pub async fn commit_to_branch(
    pool: &PgPool,
    openfga: &OpenFgaClient,
    openfga_store_id: &str,
    valkey: &mut redis::aio::MultiplexedConnection,
    org_id: Uuid,
    space_id: Uuid,
    branch_name: &str,
    source_uri: &str,
    text: &str,
    author: &str,
    message: &str,
) -> Result<Value> {
    let parent_head = ensure_branch(pool, org_id, space_id, branch_name, author).await?;
    let commit_id = Uuid::new_v4();
    let doc_id = Uuid::new_v4();
    let block_id = Uuid::new_v4();
    let bhash = content_blake3(text);
    let scope = branch_clearance_scope(space_id, branch_name);
    let zero_vec = format!("[{}]", vec!["0"; 64].join(","));
    let citations = json!([{
        "source_uri": source_uri,
        "offset": 0,
        "length": text.len(),
        "quote_hash": &bhash[..16.min(bhash.len())]
    }]);

    let mut tx = pool.begin().await?;
    sqlx::query(
        "INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action) VALUES ($1,$2,$3,$4,$5,$6,$7,'commit')",
    )
    .bind(commit_id)
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .bind(parent_head)
    .bind(author)
    .bind(message)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        "UPDATE memory_branches SET head_commit_id = $1 WHERE org_id = $2 AND space_id = $3 AND name = $4",
    )
    .bind(commit_id)
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        "INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3) VALUES ($1,$2,$3,$4,$5,$6)",
    )
    .bind(doc_id)
    .bind(org_id)
    .bind(space_id)
    .bind(&scope)
    .bind(source_uri)
    .bind(&bhash)
    .execute(&mut *tx)
    .await?;

    let prior = sqlx::query(
        r#"
        SELECT kb.id, kb.content_blake3, kb.clearance_scope
        FROM knowledge_blocks kb
        JOIN documents d ON d.id = kb.document_id
        WHERE kb.org_id = $1 AND kb.space_id = $2 AND kb.branch_name = $3
          AND d.source_uri = $4 AND kb.valid_to IS NULL
        FOR UPDATE OF kb
        "#,
    )
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .bind(source_uri)
    .fetch_all(&mut *tx)
    .await?;

    sqlx::query(
        r#"
        INSERT INTO knowledge_blocks (
          id, document_id, org_id, space_id, clearance_scope, branch_name, commit_id,
          block_type, canonical_text, content_blake3, citations, embedding
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,'decision',$8,$9,$10,$11::vector)
        "#,
    )
    .bind(block_id)
    .bind(doc_id)
    .bind(org_id)
    .bind(space_id)
    .bind(&scope)
    .bind(branch_name)
    .bind(commit_id)
    .bind(text)
    .bind(&bhash)
    .bind(&citations)
    .bind(&zero_vec)
    .execute(&mut *tx)
    .await?;

    let mut superseded = Vec::new();
    for row in prior {
        let old_id: Uuid = row.get("id");
        let old_hash: String = row.get("content_blake3");
        let old_scope: String = row.get("clearance_scope");
        sqlx::query("UPDATE knowledge_blocks SET valid_to = now(), superseded_by = $1 WHERE id = $2")
            .bind(block_id)
            .bind(old_id)
            .execute(&mut *tx)
            .await?;
        superseded.push(old_id);
        emit_superseded_outbox(&mut tx, org_id, space_id, &old_scope, old_id, &old_hash, block_id, &bhash, &format!("branch_commit:{branch_name}")).await?;
    }
    tx.commit().await?;

    // Register OpenFGA document -> space tuple and invalidate caller ACL cache
    let _ = openfga
        .write_tuple(openfga_store_id, &format!("space:{space_id}"), "space", &format!("document:{doc_id}"))
        .await;
    let _ = OpenFgaClient::invalidate_cache(openfga_store_id, author, valkey).await;

    Ok(json!({
        "commit_id": commit_id,
        "branch": branch_name,
        "document_id": doc_id,
        "block_id": block_id,
        "superseded_count": superseded.len()
    }))
}

pub async fn diff_branches(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Uuid,
    source_branch: &str,
    target_branch: &str,
) -> Result<Value> {
    let load = |bname: String| async move {
        sqlx::query(
            r#"
            SELECT DISTINCT ON (d.source_uri) d.source_uri, kb.content_blake3, kb.canonical_text
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = $1 AND kb.space_id = $2 AND kb.branch_name = $3 AND kb.valid_to IS NULL
            ORDER BY d.source_uri, kb.valid_from DESC
            "#,
        )
        .bind(org_id)
        .bind(space_id)
        .bind(bname)
        .fetch_all(pool)
        .await
    };

    let src_rows = load(source_branch.to_string()).await?;
    let tgt_rows = load(target_branch.to_string()).await?;
    let mut tgt_map = std::collections::HashMap::<String, (String, String)>::new();
    for r in tgt_rows {
        tgt_map.insert(r.get("source_uri"), (r.get("content_blake3"), r.get("canonical_text")));
    }

    let mut added = Vec::new();
    let mut updated = Vec::new();
    let mut unchanged = 0usize;
    for r in src_rows {
        let uri: String = r.get("source_uri");
        let hash: String = r.get("content_blake3");
        let text: String = r.get("canonical_text");
        match tgt_map.get(&uri) {
            None => added.push(json!({ "source_uri": uri, "canonical_text": text })),
            Some((old_hash, old_text)) if old_hash != &hash => {
                updated.push(json!({ "source_uri": uri, "old_text": old_text, "new_text": text }));
            }
            _ => unchanged += 1,
        }
    }

    Ok(json!({
        "source_branch": source_branch,
        "target_branch": target_branch,
        "added": added,
        "updated": updated,
        "unchanged_count": unchanged
    }))
}

#[allow(clippy::too_many_arguments)]
pub async fn merge_branch(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Uuid,
    source_branch: &str,
    target_branch: &str,
    author: &str,
    message: Option<&str>,
) -> Result<Value> {
    let target_head = ensure_branch(pool, org_id, space_id, target_branch, author).await?;
    let target_scope = branch_clearance_scope(space_id, target_branch);
    let merge_commit_id = Uuid::new_v4();
    let msg = message
        .map(str::to_string)
        .unwrap_or_else(|| format!("Merge branch '{source_branch}' into '{target_branch}'"));

    let mut tx = pool.begin().await?;
    sqlx::query(
        "INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action) VALUES ($1,$2,$3,$4,$5,$6,$7,'merge')",
    )
    .bind(merge_commit_id)
    .bind(org_id)
    .bind(space_id)
    .bind(target_branch)
    .bind(target_head)
    .bind(author)
    .bind(&msg)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        "UPDATE memory_branches SET head_commit_id = $1 WHERE org_id = $2 AND space_id = $3 AND name = $4",
    )
    .bind(merge_commit_id)
    .bind(org_id)
    .bind(space_id)
    .bind(target_branch)
    .execute(&mut *tx)
    .await?;

    let src_rows = sqlx::query(
        r#"
        SELECT d.source_uri, kb.document_id, kb.block_type, kb.canonical_text, kb.content_blake3, kb.citations, kb.embedding::text AS emb_text
        FROM knowledge_blocks kb
        JOIN documents d ON d.id = kb.document_id
        WHERE kb.org_id = $1 AND kb.space_id = $2 AND kb.branch_name = $3 AND kb.valid_to IS NULL
        "#,
    )
    .bind(org_id)
    .bind(space_id)
    .bind(source_branch)
    .fetch_all(&mut *tx)
    .await?;

    let mut applied = 0usize;
    let mut superseded = 0usize;
    for r in src_rows {
        let uri: String = r.get("source_uri");
        let doc_id: Uuid = r.get("document_id");
        let btype: String = r.get("block_type");
        let text: String = r.get("canonical_text");
        let bhash: String = r.get("content_blake3");
        let citations: Value = r.get("citations");
        let emb_text: String = r.get("emb_text");

        let existing = sqlx::query(
            r#"
            SELECT kb.id, kb.content_blake3, kb.clearance_scope
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = $1 AND kb.space_id = $2 AND kb.branch_name = $3
              AND d.source_uri = $4 AND kb.valid_to IS NULL
            FOR UPDATE OF kb
            "#,
        )
        .bind(org_id)
        .bind(space_id)
        .bind(target_branch)
        .bind(&uri)
        .fetch_all(&mut *tx)
        .await?;

        if !existing.is_empty() && existing.iter().all(|e| e.get::<String, _>("content_blake3") == bhash) {
            continue;
        }

        let new_bid = Uuid::new_v4();
        sqlx::query(
            r#"
            INSERT INTO knowledge_blocks (
              id, document_id, org_id, space_id, clearance_scope, branch_name, commit_id,
              block_type, canonical_text, content_blake3, citations, embedding
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::vector)
            "#,
        )
        .bind(new_bid)
        .bind(doc_id)
        .bind(org_id)
        .bind(space_id)
        .bind(&target_scope)
        .bind(target_branch)
        .bind(merge_commit_id)
        .bind(&btype)
        .bind(&text)
        .bind(&bhash)
        .bind(&citations)
        .bind(&emb_text)
        .execute(&mut *tx)
        .await?;
        applied += 1;

        for e in existing {
            let old_id: Uuid = e.get("id");
            let old_hash: String = e.get("content_blake3");
            let old_scope: String = e.get("clearance_scope");
            sqlx::query("UPDATE knowledge_blocks SET valid_to = now(), superseded_by = $1 WHERE id = $2")
                .bind(new_bid)
                .bind(old_id)
                .execute(&mut *tx)
                .await?;
            superseded += 1;
            emit_superseded_outbox(&mut tx, org_id, space_id, &old_scope, old_id, &old_hash, new_bid, &bhash, &format!("merge:{source_branch}->{target_branch}")).await?;
        }
    }

    tx.commit().await?;
    Ok(json!({
        "merge_commit_id": merge_commit_id,
        "source_branch": source_branch,
        "target_branch": target_branch,
        "applied_blocks": applied,
        "superseded_blocks": superseded
    }))
}

pub async fn revert_commit(
    pool: &PgPool,
    org_id: Uuid,
    space_id: Uuid,
    branch_name: &str,
    commit_id: Uuid,
    author: &str,
) -> Result<Value> {
    let head = ensure_branch(pool, org_id, space_id, branch_name, author).await?;
    let revert_id = Uuid::new_v4();

    let mut tx = pool.begin().await?;
    sqlx::query(
        "INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action) VALUES ($1,$2,$3,$4,$5,$6,$7,'revert')",
    )
    .bind(revert_id)
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .bind(head)
    .bind(author)
    .bind(format!("Revert commit {commit_id}"))
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        "UPDATE memory_branches SET head_commit_id = $1 WHERE org_id = $2 AND space_id = $3 AND name = $4",
    )
    .bind(revert_id)
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .execute(&mut *tx)
    .await?;

    let bad_blocks = sqlx::query(
        "SELECT id, content_blake3, clearance_scope FROM knowledge_blocks WHERE org_id = $1 AND space_id = $2 AND branch_name = $3 AND commit_id = $4 AND valid_to IS NULL FOR UPDATE",
    )
    .bind(org_id)
    .bind(space_id)
    .bind(branch_name)
    .bind(commit_id)
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
            "SELECT id, content_blake3 FROM knowledge_blocks WHERE org_id = $1 AND space_id = $2 AND branch_name = $3 AND superseded_by = $4 FOR UPDATE",
        )
        .bind(org_id)
        .bind(space_id)
        .bind(branch_name)
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
            emit_superseded_outbox(&mut tx, org_id, space_id, &bad_scope, bad_id, &bad_hash, prev_id, &prev_hash, &format!("revert:{commit_id}")).await?;
        }
    }

    tx.commit().await?;
    Ok(json!({
        "revert_commit_id": revert_id,
        "reverted_commit_id": commit_id,
        "deactivated_blocks": deactivated,
        "restored_blocks": restored
    }))
}

#[allow(clippy::too_many_arguments)]
async fn emit_superseded_outbox(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    org_id: Uuid,
    space_id: Uuid,
    clearance_scope: &str,
    old_id: Uuid,
    old_hash: &str,
    new_id: Uuid,
    new_hash: &str,
    reason: &str,
) -> Result<()> {
    let msg_id = format!("superseded:{old_id}:{new_id}:{}", &Uuid::new_v4().simple().to_string()[..8]);
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
