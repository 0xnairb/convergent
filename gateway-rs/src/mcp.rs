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
                { "name": "search_memory", "description": "Pre-filtered hybrid search over authorized Knowledge Blocks", "inputSchema": { "type": "object", "properties": { "query": { "type": "string" }, "space_id": { "type": "string" }, "top_k": { "type": "integer" } }, "required": ["query"] } },
                { "name": "get_block", "description": "Fetch a single authorized Knowledge Block by ID", "inputSchema": { "type": "object", "properties": { "block_id": { "type": "string" } }, "required": ["block_id"] } },
                { "name": "ingest_note", "description": "Enqueue a new note/document for Layer-3 distillation", "inputSchema": { "type": "object", "properties": { "space_id": { "type": "string" }, "source_uri": { "type": "string" }, "content": { "type": "string" } }, "required": ["space_id", "source_uri", "content"] } }
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
            let top_k = args.get("top_k").and_then(|v| v.as_i64()).unwrap_or(5);

            let blocks = prefiltered_hybrid_search(
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
        other => Err(format!("unknown tool: {other}")),
    }
}
