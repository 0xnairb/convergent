use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::Utc;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub org_id: Uuid,
    pub scope: String,
    pub exp: usize,
}

/// Verify OAuth 2.1 PKCE S256 (`BASE64URL(SHA256(code_verifier)) == code_challenge`).
pub fn verify_pkce_s256(code_verifier: &str, code_challenge: &str) -> bool {
    let digest = Sha256::digest(code_verifier.as_bytes());
    let computed = URL_SAFE_NO_PAD.encode(digest);
    computed == code_challenge
}

pub fn issue_jwt(secret: &[u8], sub: &str, org_id: Uuid, scope: &str, ttl_secs: i64) -> Result<String> {
    let exp = (Utc::now().timestamp() + ttl_secs) as usize;
    let claims = Claims {
        sub: sub.to_string(),
        org_id,
        scope: scope.to_string(),
        exp,
    };
    let token = encode(&Header::default(), &claims, &EncodingKey::from_secret(secret))?;
    Ok(token)
}

pub fn verify_jwt(secret: &[u8], token: &str) -> Result<Claims> {
    let data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret),
        &Validation::default(),
    )
    .map_err(|e| anyhow!("invalid bearer token: {e}"))?;
    Ok(data.claims)
}
