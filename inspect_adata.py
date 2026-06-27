import anndata
import numpy as np
import pandas as pd
from scipy import sparse

# Load in backed mode to avoid loading full matrix
adata = anndata.read_h5ad("data/adata_headBody_S_v1.0.h5ad", backed="r")

print("=" * 80)
print("ANNDATA OBJECT OVERVIEW")
print("=" * 80)
print(f"Shape: {adata.shape}")
print(f"Object exists: {adata is not None}")

print("\n" + "=" * 80)
print("1. OBS (OBSERVATIONS / CELLS)")
print("=" * 80)
print(f"Number of obs: {len(adata.obs)}")
print("\nObs columns and dtypes:")
for col in adata.obs.columns:
    print(f"  {col:30s} {adata.obs[col].dtype}")

print("\nObs head (first 5 rows):")
print(adata.obs.head())

print("\n" + "=" * 80)
print("2. VAR (FEATURES / GENES)")
print("=" * 80)
print(f"Number of var: {len(adata.var)}")
print("\nVar columns and dtypes:")
for col in adata.var.columns:
    print(f"  {col:30s} {adata.var[col].dtype}")

print("\nVar head (first 5 rows):")
print(adata.var.head())

# Check highly_variable statistics
if 'highly_variable' in adata.var.columns:
    n_hv = adata.var['highly_variable'].sum()
    print(f"\nHighly variable genes: {n_hv} / {len(adata.var)}")

if 'means' in adata.var.columns:
    print(f"\nMeans statistics:")
    print(f"  min: {adata.var['means'].min():.6f}")
    print(f"  max: {adata.var['means'].max():.6f}")
    print(f"  mean: {adata.var['means'].mean():.6f}")
    print(f"  median: {adata.var['means'].median():.6f}")

if 'dispersions' in adata.var.columns:
    print(f"\ndispersions statistics:")
    print(f"  min: {adata.var['dispersions'].min():.6f}")
    print(f"  max: {adata.var['dispersions'].max():.6f}")
    print(f"  mean: {adata.var['dispersions'].mean():.6f}")
    print(f"  median: {adata.var['dispersions'].median():.6f}")

if 'dispersions_norm' in adata.var.columns:
    print(f"\ndispersions_norm statistics:")
    print(f"  min: {adata.var['dispersions_norm'].min():.6f}")
    print(f"  max: {adata.var['dispersions_norm'].max():.6f}")
    print(f"  mean: {adata.var['dispersions_norm'].mean():.6f}")
    print(f"  median: {adata.var['dispersions_norm'].median():.6f}")

print("\n" + "=" * 80)
print("3. X MATRIX INSPECTION (reading first 500 cells)")
print("=" * 80)
print(f"X type: {type(adata.X)}")
print(f"X dtype: {adata.X.dtype}")

# Read first 500 cells - adata.X returns the actual CSR matrix when backed
X_sample = adata.X[:500]
print(f"X_sample type: {type(X_sample)}")
print(f"X_sample shape: {X_sample.shape}")
print(f"X_sample is sparse: {sparse.issparse(X_sample)}")

# Convert to dense for analysis
if sparse.issparse(X_sample):
    X_dense = X_sample.toarray()
else:
    X_dense = np.asarray(X_sample)

X_flat = X_dense.flatten()

print(f"\nX value statistics (first 500 cells, all genes):")
print(f"  min: {X_flat.min():.10f}")
print(f"  max: {X_flat.max():.10f}")
print(f"  mean: {X_flat.mean():.10f}")
print(f"  median: {np.median(X_flat):.10f}")
print(f"  0th percentile (min): {np.percentile(X_flat, 0):.10f}")
print(f"  25th percentile: {np.percentile(X_flat, 25):.10f}")
print(f"  50th percentile: {np.percentile(X_flat, 50):.10f}")
print(f"  75th percentile: {np.percentile(X_flat, 75):.10f}")
print(f"  99th percentile: {np.percentile(X_flat, 99):.10f}")
print(f"  # zeros in sample: {(X_flat == 0).sum()} / {len(X_flat)} ({100.0 * (X_flat == 0).sum() / len(X_flat):.2f}%)")

# Sample some specific values
print(f"\nSample X values (first 10x10 block):")
print(X_dense[:10, :10])

# Check per-cell sums
cell_sums = X_dense.sum(axis=1)

print(f"\nPer-cell sums (first 500 cells):")
print(f"  min: {cell_sums.min():.6f}")
print(f"  max: {cell_sums.max():.6f}")
print(f"  mean: {cell_sums.mean():.6f}")
print(f"  median: {np.median(cell_sums):.6f}")

# Check if expm1(X).sum is ~1e4 (would indicate log1p(normalize_total))
print(f"\nExpm1-transformed per-cell sums (checking if ~1e4):")
X_expm1 = np.expm1(X_dense)
cell_sums_expm1 = X_expm1.sum(axis=1)
print(f"  min: {cell_sums_expm1.min():.2f}")
print(f"  max: {cell_sums_expm1.max():.2f}")
print(f"  mean: {cell_sums_expm1.mean():.2f}")
print(f"  median: {np.median(cell_sums_expm1):.2f}")
print(f"  std: {cell_sums_expm1.std():.2f}")

print("\n" + "=" * 80)
print("4. OBSM (EMBEDDINGS)")
print("=" * 80)
print(f"obsm keys: {list(adata.obsm.keys())}")
for key in adata.obsm.keys():
    print(f"  {key:20s} shape: {adata.obsm[key].shape}")

print("\n" + "=" * 80)
print("5. OBSP (CELL-CELL PAIRWISE)")
print("=" * 80)
print(f"obsp keys: {list(adata.obsp.keys())}")
for key in adata.obsp.keys():
    print(f"  {key:20s} shape: {adata.obsp[key].shape}")

print("\n" + "=" * 80)
print("6. LAYERS")
print("=" * 80)
print(f"Layer keys: {list(adata.layers.keys())}")

print("\n" + "=" * 80)
print("7. UNS (UNSTRUCTURED METADATA)")
print("=" * 80)
print(f"uns keys: {list(adata.uns.keys())}")
for key in adata.uns.keys():
    val = adata.uns[key]
    if isinstance(val, dict):
        print(f"  {key:30s} (dict with keys: {list(val.keys())[:5]}...)")
    elif isinstance(val, (list, tuple)):
        print(f"  {key:30s} (list/tuple of length {len(val)})")
    elif isinstance(val, np.ndarray):
        print(f"  {key:30s} (array shape: {val.shape}, dtype: {val.dtype})")
    else:
        print(f"  {key:30s} (type: {type(val).__name__})")

print("\n" + "=" * 80)
print("8. RAW ATTRIBUTE")
print("=" * 80)
print(f"adata.raw exists: {adata.raw is not None}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
