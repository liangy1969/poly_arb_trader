//! Kalshi WS auth — RSA-PSS request signing (DESIGN_KALSHI_VENUE §2).
//!
//! Each authenticated request signs `timestamp_ms + METHOD + path` with the
//! account's RSA private key using **RSA-PSS / SHA-256 / MGF1-SHA256** and a salt
//! length equal to the digest length (32). The base64 signature + key id + the
//! same timestamp go in the `KALSHI-ACCESS-*` headers.

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::pss::SigningKey;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::RsaPrivateKey;
use sha2::Sha256;

pub struct Signer {
    key_id: String,
    signing: SigningKey<Sha256>,
}

impl Signer {
    /// Load the access-key id + an RSA private key (PKCS#8 or PKCS#1 PEM file).
    pub fn load(key_id: &str, pem_path: &str) -> Result<Self> {
        let pem = std::fs::read_to_string(pem_path)
            .with_context(|| format!("reading Kalshi private key {pem_path}"))?;
        let key = RsaPrivateKey::from_pkcs8_pem(&pem)
            .or_else(|_| RsaPrivateKey::from_pkcs1_pem(&pem))
            .context("parsing Kalshi RSA private key (expected PKCS#8 or PKCS#1 PEM)")?;
        Ok(Signer {
            key_id: key_id.to_string(),
            // SigningKey::new uses salt length = digest size (32), which is what
            // Kalshi's PSS verification expects.
            signing: SigningKey::<Sha256>::new(key),
        })
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Sign `{ts_ms}{method}{path}`; returns `(timestamp_string, base64_signature)`.
    pub fn sign(&self, method: &str, path: &str, ts_ms: i64) -> Result<(String, String)> {
        let ts = ts_ms.to_string();
        let msg = format!("{ts}{method}{path}");
        let mut rng = rand::rngs::OsRng;
        let sig = self
            .signing
            .try_sign_with_rng(&mut rng, msg.as_bytes())
            .context("RSA-PSS signing failed")?;
        Ok((ts, STANDARD.encode(sig.to_bytes())))
    }
}
