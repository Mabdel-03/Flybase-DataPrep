import anndata

adata = anndata.read_h5ad("data/adata_headBody_S_v1.0.h5ad", backed="r")

print("All obs columns:")
print(adata.obs.columns.tolist())

print("\nAll var columns:")
print(adata.var.columns.tolist())

