use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::retrieval::KnowledgeBlock;

#[derive(Serialize)]
struct EvaluateReq<'a> {
    principal_id: &'a str,
    action: &'a str,
    blocks: Vec<BlockRef<'a>>,
}

#[derive(Serialize)]
struct BlockRef<'a> {
    block_hash: &'a str,
    clearance_scope: &'a str,
}

#[derive(Deserialize)]
struct EvaluateResp {
    verdict: String,
    #[serde(default)]
    withheld_handles: Vec<String>,
}

/// Optional external Policy / Ethical Walls webhook.
/// Fails closed (returns Err) if the webhook is unreachable or returns BLOCK/HALT.
pub async fn filter_withheld_blocks(
    http: &reqwest::Client,
    policy_webhook_url: &str,
    principal_id: &str,
    blocks: Vec<KnowledgeBlock>,
) -> Result<(Vec<KnowledgeBlock>, Vec<String>)> {
    if blocks.is_empty() {
        return Ok((blocks, vec![]));
    }

    let req = EvaluateReq {
        principal_id,
        action: "convergent.context.compose",
        blocks: blocks
            .iter()
            .map(|b| BlockRef {
                block_hash: &b.content_blake3,
                clearance_scope: &b.clearance_scope,
            })
            .collect(),
    };

    let url = format!(
        "{}/api/v1/governance/evaluate",
        policy_webhook_url.trim_end_matches('/')
    );
    let resp: EvaluateResp = http
        .post(url)
        .json(&req)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    match resp.verdict.as_str() {
        "ALLOW" | "CONSTRAIN" => {
            let withheld: HashSet<String> = resp.withheld_handles.iter().cloned().collect();
            let kept = blocks
                .into_iter()
                .filter(|b| !withheld.contains(&b.content_blake3))
                .collect();
            Ok((kept, resp.withheld_handles))
        }
        other => bail!("Policy webhook denied context composition: {other}"),
    }
}
