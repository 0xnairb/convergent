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
use crate::overlays::{apply_overlay_to_blocks, inspect_overlay, land_overlay, open_overlay, stage_delta, undo_revision};
use crate::retrieval::prefiltered_hybrid_search;

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
                { "name": "search_memory", "description": "Pre-filtered hybrid search over authorized Knowledge Blocks (supports zero-copy overlay_id)", "inputSchema": { "type": "object", "properties": { "query": { "type": "string" }, "space_id": { "type": "string" }, "overlay_id": { "type": "string" }, "top_k": { "type": "integer" } }, "required": ["query"] } },
                { "name": "get_block", "description": "Fetch a single authorized Knowledge Block by ID", "inputSchema": { "type": "object", "properties": { "block_id": { "type": "string" } }, "required": ["block_id"] } },
                { "name": "ingest_note", "description": "Enqueue a new note/document for Smart Memory distillation", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "source_uri": { "type": "string" }, "content": { "type": "string" } }, "required": ["space_id", "source_uri", "content"] } },
                { "name": "open_overlay", "description": "Open a 0-byte virtual overlay pinned to a Space revision", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "base_rev": { "type": "integer" } }, "required": ["space_id"] } },
                { "name": "stage_delta", "description": "Stage an upsert or retract delta inside an open overlay", "inputSchema": { "type": "object", "properties": { "overlay_id": { "type": "string" }, "source_uri": { "type": "string" }, "content": { "type": "string" }, "op": { "type": "string" } }, "required": ["overlay_id", "source_uri"] } },
                { "name": "inspect_overlay", "description": "Inspect staged deltas inside an overlay", "inputSchema": { "type": "object", "properties": { "overlay_id": { "type": "string" } }, "required": ["overlay_id"] } },
                { "name": "land_overlay", "description": "Atomically land an overlay changeset with optimistic concurrency auto-rebase & KV eviction", "inputSchema": { "type": "object", "properties": { "overlay_id": { "type": "string" }, "message": { "type": "string" }, "force": { "type": "boolean" } }, "required": ["overlay_id"] } },
                { "name": "undo_revision", "description": "Atomically undo a landed Space revision and restore superseded blocks", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "rev": { "type": "integer" } }, "required": ["space_id", "rev"] } }
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
            let overlay_id = args
                .get("overlay_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok());
            let top_k = args.get("top_k").and_then(|v| v.as_i64()).unwrap_or(5);

            let mut blocks = prefiltered_hybrid_search(
                &state.pool,
                claims.org_id,
                space_id,
                &allowed_docs,
                query,
                None,
                top_k,
            )
            .await
            .map_err(|e| e.to_string())?;
            if let Some(oid) = overlay_id {
                blocks = apply_overlay_to_blocks(&state.pool, oid, blocks)
                    .await
                    .map_err(|e| e.to_string())?;
            }

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
        "open_overlay" => {
            let sid = parse_uuid_arg(args, "space_id")?;
            let base_rev = args.get("base_rev").and_then(|v| v.as_i64());
            let res = open_overlay(&state.pool, claims.org_id, sid, &claims.sub, base_rev)
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "overlay": res }))
        }
        "stage_delta" => {
            let oid = parse_uuid_arg(args, "overlay_id")?;
            let source_uri = args.get("source_uri").and_then(|v| v.as_str()).ok_or("missing source_uri")?;
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let op = args.get("op").and_then(|v| v.as_str()).unwrap_or("upsert");
            let res = stage_delta(&state.pool, oid, source_uri, content, op)
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "delta": res }))
        }
        "inspect_overlay" => {
            let oid = parse_uuid_arg(args, "overlay_id")?;
            let res = inspect_overlay(&state.pool, oid).await.map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "overlay": res }))
        }
        "land_overlay" => {
            let oid = parse_uuid_arg(args, "overlay_id")?;
            let message = args.get("message").and_then(|v| v.as_str()).unwrap_or("Land overlay");
            let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
            let res = land_overlay(
                &state.pool,
                &state.openfga,
                &state.openfga_store_id,
                &mut valkey,
                oid,
                &claims.sub,
                message,
                force,
            )
            .await
            .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "land": res }))
        }
        "undo_revision" => {
            let sid = parse_uuid_arg(args, "space_id")?;
            let rev = args.get("rev").and_then(|v| v.as_i64()).ok_or("missing rev")?;
            let res = undo_revision(&state.pool, claims.org_id, sid, rev, &claims.sub)
                .await
                .map_err(|e| e.to_string())?;
            Ok(json!({ "content": [{ "type": "text", "text": res.to_string() }], "undo": res }))
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

fn parse_uuid_arg(args: &Value, key: &str) -> Result<Uuid, String> {
    let raw = args.get(key).and_then(|v| v.as_str()).ok_or_else(|| format!("missing {key}"))?;
    Uuid::parse_str(raw).map_err(|_| format!("invalid {key}"))
}
