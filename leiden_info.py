import anndata

adata = anndata.read_h5ad("data/adata_headBody_S_v1.0.h5ad", backed="r")

print("Leiden clustering distribution:")
leiden_counts = adata.obs['leiden'].value_counts().sort_index()
print(f"Number of clusters: {len(leiden_counts)}")
print(f"\nCluster ID -> Cell Count:")
for idx, count in leiden_counts.items():
    print(f"  {idx:3s}: {count:7d}")
    
print(f"\nSummary stats:")
print(f"  Min cluster size: {leiden_counts.min()}")
print(f"  Max cluster size: {leiden_counts.max()}")
print(f"  Mean cluster size: {leiden_counts.mean():.1f}")
print(f"  Median cluster size: {leiden_counts.median():.1f}")

