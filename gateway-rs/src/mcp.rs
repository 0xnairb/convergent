use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::app::{extract_claims, AppState};
use crate::branches::{commit_to_branch, create_branch, diff_branches, merge_branch, revert_commit};
use crate::retrieval::prefiltered_hybrid_search_branch;

/// Standard MCP Streamable HTTP JSON-RPC 2.0 endpoint (`POST /mcp`).
pub async fn handle_mcp(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let claims = match extract_claims(&headers, &state.jwt_secret) {
        Ok(c) => c,
        Err(msg) => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "jsonrpc": "2.0", "id": req.get("id"), "error": { "code": -32001, "message": msg } })),
            )
                .into_response()
        }
    };

    let id = req.get("id").cloned().unwrap_or(json!(null));
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2025-03-26",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "convergent-mcp", "version": "0.1.0" }
        })),
        "tools/list" => Ok(json!({
            "tools": [
                { "name": "list_spaces", "description": "List authorized Context Spaces", "inputSchema": { "type": "object", "properties": {} } },
                { "name": "search_memory", "description": "Pre-filtered hybrid search over authorized Knowledge Blocks (branch-aware)", "inputSchema": { "type": "object", "properties": { "query": { "type": "string" }, "space_id": { "type": "string" }, "branch": { "type": "string" }, "top_k": { "type": "integer" } }, "required": ["query"] } },
                { "name": "get_block", "description": "Fetch a single authorized Knowledge Block by ID", "inputSchema": { "type": "object", "properties": { "block_id": { "type": "string" } }, "required": ["block_id"] } },
                { "name": "ingest_note", "description": "Enqueue a new note/document for Smart Memory distillation", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "source_uri": { "type": "string" }, "content": { "type": "string" } }, "required": ["space_id", "source_uri", "content"] } },
                { "name": "branch_memory", "description": "Create an isolated world-state memory branch from an existing branch", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "new_branch": { "type": "string" }, "from_branch": { "type": "string" } }, "required": ["space_id", "new_branch"] } },
                { "name": "commit_memory", "description": "Commit primary-source content to a specific memory branch", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "branch_name": { "type": "string" }, "source_uri": { "type": "string" }, "content": { "type": "string" }, "message": { "type": "string" } }, "required": ["space_id", "source_uri", "content"] } },
                { "name": "diff_memory", "description": "Semantic diff between two world-state branches", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "source_branch": { "type": "string" }, "target_branch": { "type": "string" } }, "required": ["space_id", "source_branch"] } },
                { "name": "merge_memory", "description": "Atomically merge a memory branch into target (default main) and evict stale KV", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "source_branch": { "type": "string" }, "target_branch": { "type": "string" }, "message": { "type": "string" } }, "required": ["space_id", "source_branch"] } },
                { "name": "revert_memory", "description": "Instantly revert a memory commit and restore superseded blocks", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "branch_name": { "type": "string" }, "commit_id": { "type": "string" } }, "required": ["space_id", "commit_id"] } }
            ]
        })),
        "tools/call" => {
            let params = req.get("params").cloned().unwrap_or(json!({}));
            let tool = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            dispatch_tool(&state, &claims, tool, &args).await
        }
        other => Err(format!("unsupported MCP method: {other}")),
    };

    match result {
        Ok(val) => (
            StatusCode::OK,
            Json(json!({ "jsonrpc": "2.0", "id": id, "result": val })),
        )
            .into_response(),
        Err(msg) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32602, "message": msg } })),
        )
            .into_response(),
    }
}

async fn dispatch_tool(
    state: &AppState,
    claims: &crate::oauth::Claims,
    tool: &str,
    args: &Value,
) -> Result<Value, String> {
    let mut valkey = state
        .valkey
        .get_multiplexed_async_connection()
        .await
        .map_err(|e| e.to_string())?;

    let allowed_docs = state
        .openfga
        .list_allowed_documents(&state.openfga_store_id, &claims.sub, &mut valkey, 30)
        .await
        .map_err(|e| e.to_string())?;

    match tool {
        "list_spaces" => {
            let rows = sqlx::query(
                "SELECT DISTINCT space_id FROM documents WHERE org_id = $1 AND id = ANY($2)",
            )
            .bind(claims.org_id)
            .bind(&allowed_docs)
            .fetch_all(&state.pool)
            .await
            .map_err(|e| e.to_string())?;
            let spaces: Vec<String> = rows
                .into_iter()
                .map(|r| r.get::<Uuid, _>("space_id").to_string())
                .collect();
            Ok(json!({ "content": [{ "type": "text", "text": serde_json::to_string(&spaces).unwrap() }], "spaces": spaces }))
        }
        "search_memory" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or("missing query")?;
            let space_id = args
                .get("space_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok());
            let branch = args.get("branch").and_then(|v| v.as_str()).unwrap_or("main");
            let top_k = args.get("top_k").and_then(|v| v.as_i64()).unwrap_or(5);

            let blocks = prefiltered_hybrid_search_branch(
                &state.pool,
                claims.org_id,
                space_id,
                branch,
                &allowed_docs,
                query,
                None,
                top_k,
            )
            .await
            .map_err(|e| e.to_string())?;

            Ok(json!({
                "content": [{ "type": "text", "text": serde_json::to_string(&blocks).unwrap() }],
                "blocks": blocks
            }))
        }
        "get_block" => {
            let block_id_str = args
                .get("block_id")
                .and_then(|v| v.as_str())
                .ok_or("missing block_id")?;
            let block_id = Uuid::parse_str(block_id_str).map_err(|_| "invalid block_id")?;
            let row = sqlx::query(
                "SELECT id, document_id, clearance_scope, canonical_text, content_blake3, citations FROM knowledge_blocks WHERE id = $1 AND org_id = $2 AND document_id = ANY($3)",
            )
            .bind(block_id)
            .bind(claims.org_id)
            .bind(&allowed_docs)
            .fetch_optional(&state.pool)
            .await
            .map_err(|e| e.to_string())?;

            match row {
                Some(r) => {
                    let block = json!({
                        "id": r.get::<Uuid, _>("id"),
                        "document_id": r.get::<Uuid, _>("document_id"),
                        "clearance_scope": r.get::<String, _>("clearance_scope"),
                        "canonical_text": r.get::<String, _>("canonical_text"),
                        "content_blake3": r.get::<String, _>("content_blake3"),
                        "citations": r.get::<Value, _>("citations"),
                    });
                    Ok(json!({ "content": [{ "type": "text", "text": block.to_string() }], "block": block }))
                }
                None => Err("block not found or unauthorized".to_string()),
            }
        }
        "ingest_note" => {
            let space_id = args.get("space_id").and_then(|v| v.as_str()).ok_or("missing space_id")?;
            let source_uri = args.get("source_uri").and_then(|v| v.as_str()).ok_or("missing source_uri")?;
            let content = args.get("content").and_then(|v| v.as_str()).ok_or("missing content")?;
            let payload = json!({
                "org_id": claims.org_id,
                "space_id": space_id,
                "source_uri": source_uri,
                "content": content,
                "principal_id": claims.sub
            });
            let js = async_nats::jetstream::new(state.nats.clone());
            js.publish(
                "convergent.ingest.requested",
                serde_json::to_vec(&payload).unwrap().into(),
            )
            .await
            .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": "enqueued" }], "status": "enqueued" }))
        }
        "branch_memory" => {
            let sid = parse_uuid_arg(args, "space_id")?;
            let new_branch = args.get("new_branch").and_then(|v| v.as_str()).ok_or("missing new_branch")?;
            let from_branch = args.get("from_branch").and_then(|v| v.as_str()).unwrap_or("main");
            let cid = create_branch(&state.pool, claims.org_id, sid, new_branch, from_branch, &claims.sub)
                .await
                .map_err(|e| e.to_string())?;
            let res = json!({ "branch": new_branch, "from_branch": from_branch, "commit_id": cid });
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "result": res }))
        }
        "commit_memory" => {
            let sid = parse_uuid_arg(args, "space_id")?;
            let branch = args.get("branch_name").and_then(|v| v.as_str()).unwrap_or("main");
            let source_uri = args.get("source_uri").and_then(|v| v.as_str()).ok_or("missing source_uri")?;
            let content = args.get("content").and_then(|v| v.as_str()).ok_or("missing content")?;
            let message = args.get("message").and_then(|v| v.as_str()).unwrap_or("MCP commit");
            let res = commit_to_branch(
                &state.pool,
                &state.openfga,
                &state.openfga_store_id,
                &mut valkey,
                claims.org_id,
                sid,
                branch,
                source_uri,
                content,
                &claims.sub,
                message,
            )
            .await
            .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "result": res }))
        }
        "diff_memory" => {
            let sid = parse_uuid_arg(args, "space_id")?;
            let src = args.get("source_branch").and_then(|v| v.as_str()).ok_or("missing source_branch")?;
            let tgt = args.get("target_branch").and_then(|v| v.as_str()).unwrap_or("main");
            let res = diff_branches(&state.pool, claims.org_id, sid, src, tgt)
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "diff": res }))
        }
        "merge_memory" => {
            let sid = parse_uuid_arg(args, "space_id")?;
            let src = args.get("source_branch").and_then(|v| v.as_str()).ok_or("missing source_branch")?;
            let tgt = args.get("target_branch").and_then(|v| v.as_str()).unwrap_or("main");
            let msg = args.get("message").and_then(|v| v.as_str());
            let res = merge_branch(&state.pool, claims.org_id, sid, src, tgt, &claims.sub, msg)
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "merge": res }))
        }
        "revert_memory" => {
            let sid = parse_uuid_arg(args, "space_id")?;
            let branch = args.get("branch_name").and_then(|v| v.as_str()).unwrap_or("main");
            let cid = parse_uuid_arg(args, "commit_id")?;
            let res = revert_commit(&state.pool, claims.org_id, sid, branch, cid, &claims.sub)
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "revert": res }))
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

fn parse_uuid_arg(args: &Value, key: &str) -> Result<Uuid, String> {
    let raw = args.get(key).and_then(|v| v.as_str()).ok_or_else(|| format!("missing {key}"))?;
    Uuid::parse_str(raw).map_err(|_| format!("invalid {key}"))
}
