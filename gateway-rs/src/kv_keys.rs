use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Derive a clearance-scoped secret from the deployment master seed so
/// different clearance scopes produce cryptographically disjoint KV keys.
fn derive_scope_secret(master_seed: &[u8], clearance_scope: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(master_seed).expect("HMAC accepts any key size");
    mac.update(b"cvg:scope:v1:");
    mac.update(clearance_scope.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

/// Compute `cvg:kv:v1:<hex(HMAC-SHA256(secret(clearance_scope), model_id || ":" || block_blake3))>`.
pub fn derive_kv_key(
    master_seed: &[u8],
    clearance_scope: &str,
    model_id: &str,
    block_blake3: &str,
) -> String {
    let scope_secret = derive_scope_secret(master_seed, clearance_scope);
    let mut mac = HmacSha256::new_from_slice(&scope_secret).expect("HMAC accepts any key size");
    mac.update(model_id.as_bytes());
    mac.update(b":");
    mac.update(block_blake3.as_bytes());
    format!("cvg:kv:v1:{}", hex::encode(mac.finalize().into_bytes()))
}

pub fn content_blake3(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}
