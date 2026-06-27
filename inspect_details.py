import anndata
import numpy as np

adata = anndata.read_h5ad("data/adata_headBody_S_v1.0.h5ad", backed="r")

print("=" * 80)
print("DETAILED UNS INSPECTION")
print("=" * 80)

# PCA details
print("\n1. PCA (uns['pca']):")
pca_dict = adata.uns['pca']
for key, val in pca_dict.items():
    if isinstance(val, np.ndarray):
        print(f"  {key:25s} shape: {val.shape}, dtype: {val.dtype}")
        if key == 'variance':
            print(f"    First 10 PCs variance: {val[:10]}")
            print(f"    Total variance: {val.sum():.4f}")
            print(f"    Cumulative var (first 10): {val[:10].sum():.4f} ({100*val[:10].sum()/val.sum():.2f}%)")
            print(f"    Cumulative var (first 50): {val[:50].sum():.4f} ({100*val[:50].sum()/val.sum():.2f}%)")
            print(f"    Cumulative var (first 100): {val[:100].sum():.4f} ({100*val[:100].sum()/val.sum():.2f}%)")
        elif key == 'variance_ratio':
            print(f"    First 10 PCs variance ratio: {val[:10]}")
            print(f"    First 50 PCs variance ratio sum: {val[:50].sum():.4f}")
    else:
        print(f"  {key:25s} type: {type(val).__name__}")

# HVG details
print("\n2. HVG (uns['hvg']):")
hvg_dict = adata.uns['hvg']
for key, val in hvg_dict.items():
    print(f"  {key:25s}: {val}")

# Leiden details
print("\n3. Leiden (uns['leiden']):")
leiden_dict = adata.uns['leiden']
for key, val in leiden_dict.items():
    if isinstance(val, dict):
        print(f"  {key:25s} (dict): {val}")
    elif isinstance(val, np.ndarray):
        print(f"  {key:25s} shape: {val.shape}, dtype: {val.dtype}")
    else:
        print(f"  {key:25s}: {val}")

# Neighbors details
print("\n4. Neighbors (uns['neighbors']):")
neighbors_dict = adata.uns['neighbors']
for key, val in neighbors_dict.items():
    print(f"  {key:25s}: {val}")

# Leiden clustering stats
print("\n5. Leiden clustering (from obs):")
if 'leiden' in adata.obs.columns:
    leiden_clusters = adata.obs['leiden'].value_counts().sort_index()
    print(f"  Number of clusters: {len(leiden_clusters)}")
    print(f"  Cluster sizes:")
    print(leiden_clusters.to_string())
    print(f"\n  Min cluster size: {leiden_clusters.min()}")
    print(f"  Max cluster size: {leiden_clusters.max()}")
    print(f"  Mean cluster size: {leiden_clusters.mean():.1f}")
    print(f"  Median cluster size: {leiden_clusters.median():.1f}")

# Check tissue/sex distribution
print("\n6. Metadata distribution:")
print("\nTissue:")
print(adata.obs['tissue'].value_counts())
print("\nSex:")
print(adata.obs['sex'].value_counts())
print("\nAge:")
print(adata.obs['age'].value_counts())
print("\nDataset:")
print(adata.obs['dataset'].value_counts())

