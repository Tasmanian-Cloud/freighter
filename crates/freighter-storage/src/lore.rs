//! Crate storage backed by Lore's sangdb-native `ImmutableStore`
//! (`lore_storage::sangdb_store::SangdbImmutableStore`) — consolidating what was a standalone
//! `iroh-blobs` `FsStore` (this crate's own, separate content-addressed store) onto the SAME
//! durable, hashtrinity-native backend `lore-storage`'s own storage layer now stands on.
//! `iroh-blobs`' storage is closed (no pluggable trait for a third-party backend — verified
//! against its own source, `store/fs.rs`/`store/mem.rs` are concrete structs, not impls of a
//! generic `Store` trait), so rather than maintain two independent content-addressed stores that
//! would need to be kept in sync by hand, freighter's crate bytes now live in the SAME store
//! Lore's revision/branch model reads.
//!
//! `(name, version)` maps onto Lore's `(Partition, Address)` scheme: one fixed partition for
//! freighter's crates, `address.hash = Hash::hash_buffer("{name}-{version}")` — the same
//! content-addressing idea the original iroh-blobs plan used (a `"{name}-{version}"` tag), just
//! keyed into Lore's own address space instead of an iroh-blobs tag string.

use async_trait::async_trait;
use bytes::Bytes;
use freighter_api_types::storage::{FileResponse, StorageError, StorageProvider, StorageResult};
use lore_storage::immutable_store::{ImmutableStore, StoreError as LoreStoreError};
use lore_storage::sangdb_store::SangdbImmutableStore;
use lore_storage::store_types::{StoreMatch, StoreObliterateStats};
use lore_storage::{Address, Context, Fragment, Hash, Partition};
use std::sync::Arc;

fn crate_address(name: &str, version: &str) -> Address {
    let hash = Hash::hash_buffer(format!("{name}-{version}").as_bytes());
    Address { hash, context: Context::default() }
}

fn map_store_error(e: LoreStoreError) -> StorageError {
    match e {
        LoreStoreError::AddressNotFound(_) | LoreStoreError::PayloadNotFound(_) => StorageError::NotFound,
        other => StorageError::ServiceError(anyhow::anyhow!("{other}")),
    }
}

/// Crate blob storage backed by Lore's sangdb-native `ImmutableStore` — durable across restarts
/// via sangdb's own `.seed` snapshot mechanism, the exact same store Lore's revision model reads.
pub struct LoreStorageProvider {
    store: Arc<SangdbImmutableStore>,
    partition: Partition,
}

impl LoreStorageProvider {
    #[must_use]
    pub fn new() -> Self {
        LoreStorageProvider { store: Arc::new(SangdbImmutableStore::new()), partition: Partition::default() }
    }

    /// Share an existing store (e.g. one Lore's own revision/branch machinery is also reading) —
    /// crate bytes and repository content living in the SAME store, not two.
    #[must_use]
    pub fn with_store(store: Arc<SangdbImmutableStore>, partition: Partition) -> Self {
        LoreStorageProvider { store, partition }
    }
}

impl Default for LoreStorageProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StorageProvider for LoreStorageProvider {
    async fn pull_crate(&self, name: &str, version: &str, _tarball_checksum: [u8; 32]) -> StorageResult<FileResponse> {
        let address = crate_address(name, version);
        let (_, data) =
            self.store.clone().get(self.partition, address, StoreMatch::MatchFull).await.map_err(map_store_error)?;
        Ok(FileResponse { last_modified: None, data })
    }

    async fn put_crate(&self, name: &str, version: &str, crate_bytes: Bytes, _sha256: [u8; 32]) -> StorageResult<()> {
        let address = crate_address(name, version);
        let fragment =
            Fragment { flags: 0, size_payload: crate_bytes.len() as u32, size_content: crate_bytes.len() as u64 };
        self.store.clone().put(self.partition, address, fragment, Some(crate_bytes), false).await.map_err(map_store_error)
    }

    async fn delete_crate(&self, name: &str, version: &str, _tarball_checksum: [u8; 32]) -> StorageResult<()> {
        let address = crate_address(name, version);
        let stats = Arc::new(StoreObliterateStats::default());
        self.store.clone().obliterate(self.partition, address, stats).await.map_err(map_store_error)
    }

    async fn healthcheck(&self) -> anyhow::Result<()> {
        // A store-level existence check on a definitely-absent address proves the store answers
        // queries at all -- there's no separate network endpoint to ping for an embedded store.
        let probe = crate_address("__freighter_healthcheck__", "0.0.0");
        self.store.clone().exist(self.partition, probe, StoreMatch::MatchFull).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn put_then_pull_round_trips_bit_exact() {
        let provider = LoreStorageProvider::new();
        let bytes = Bytes::from_static(b"a fake .crate tarball's bytes, for real this time");
        provider.put_crate("demo-crate", "0.1.0", bytes.clone(), [0u8; 32]).await.unwrap();

        let fetched = provider.pull_crate("demo-crate", "0.1.0", [0u8; 32]).await.unwrap();
        assert_eq!(fetched.data, bytes);
    }

    #[tokio::test]
    async fn pulling_an_unknown_crate_is_not_found() {
        let provider = LoreStorageProvider::new();
        // `FileResponse` (the Ok side) has no `Debug` impl, so `unwrap_err` (which requires Ok:
        // Debug to format a would-be panic message) doesn't typecheck here -- match explicitly.
        match provider.pull_crate("nope", "0.0.0", [0u8; 32]).await {
            Err(StorageError::NotFound) => {}
            other => panic!("expected StorageError::NotFound for an unknown crate, got Ok={}", other.is_ok()),
        }
    }

    #[tokio::test]
    async fn delete_then_pull_is_not_found() {
        let provider = LoreStorageProvider::new();
        let bytes = Bytes::from_static(b"some crate bytes");
        provider.put_crate("demo", "1.0.0", bytes, [0u8; 32]).await.unwrap();
        provider.delete_crate("demo", "1.0.0", [0u8; 32]).await.unwrap();
        match provider.pull_crate("demo", "1.0.0", [0u8; 32]).await {
            Err(StorageError::NotFound) => {}
            other => panic!("expected StorageError::NotFound after delete, got Ok={}", other.is_ok()),
        }
    }

    #[tokio::test]
    async fn different_versions_of_the_same_crate_are_independent() {
        let provider = LoreStorageProvider::new();
        provider.put_crate("demo", "1.0.0", Bytes::from_static(b"v1 bytes"), [0u8; 32]).await.unwrap();
        provider.put_crate("demo", "2.0.0", Bytes::from_static(b"v2 bytes"), [0u8; 32]).await.unwrap();

        let v1 = provider.pull_crate("demo", "1.0.0", [0u8; 32]).await.unwrap();
        let v2 = provider.pull_crate("demo", "2.0.0", [0u8; 32]).await.unwrap();
        assert_eq!(v1.data, Bytes::from_static(b"v1 bytes"));
        assert_eq!(v2.data, Bytes::from_static(b"v2 bytes"));
    }

    #[tokio::test]
    async fn healthcheck_succeeds_on_a_fresh_store() {
        let provider = LoreStorageProvider::new();
        provider.healthcheck().await.unwrap();
    }

    #[tokio::test]
    async fn shared_store_lets_two_providers_see_the_same_crates() {
        let store = Arc::new(SangdbImmutableStore::new());
        let partition = Partition::default();
        let writer = LoreStorageProvider::with_store(store.clone(), partition);
        let reader = LoreStorageProvider::with_store(store, partition);

        writer.put_crate("shared", "1.0.0", Bytes::from_static(b"shared bytes"), [0u8; 32]).await.unwrap();
        let fetched = reader.pull_crate("shared", "1.0.0", [0u8; 32]).await.unwrap();
        assert_eq!(fetched.data, Bytes::from_static(b"shared bytes"));
    }
}
