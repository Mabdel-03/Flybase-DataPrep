import anndata as ad, numpy as np, pandas as pd
from sklearn.metrics import silhouette_score

A = ad.read_h5ad('outputs/03_Integrated/sweep/scale_dataset/fly_annotated.h5ad', backed='r')
obs = A.obs
print("N cells:", A.n_obs)
print("obs cols:", list(obs.columns))

tissue = obs['tissue'].astype(str)
print("\n=== tissue counts ===")
print(tissue.value_counts())

# ---- 1. broad x tissue crosstab ----
print("\n=== 1. afca_annotation_broad x tissue ===")
ct = pd.crosstab(obs['afca_annotation_broad'].astype(str), tissue)
ct['total'] = ct.sum(axis=1)
for c in ['head', 'body']:
    if c not in ct.columns:
        ct[c] = 0
ct['head_frac'] = ct['head'] / ct['total']
ct['body_frac'] = ct['body'] / ct['total']
ct = ct.sort_values('total', ascending=False)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
print(ct.head(20).to_string())

# classify
both = ct[(ct['head_frac'] >= 0.1) & (ct['body_frac'] >= 0.1)]
head_excl = ct[ct['head_frac'] >= 0.9]
body_excl = ct[ct['body_frac'] >= 0.9]
print("\n# broad classes shared (both>=10%):", len(both))
print("# head-exclusive (>=90% head):", len(head_excl))
print("# body-exclusive (>=90% body):", len(body_excl))

# ---- 2. leiden_res1 tissue purity ----
print("\n=== 2. leiden_res1 tissue purity ===")
lc = obs['leiden_res1'].astype(str)
g = pd.crosstab(lc, tissue)
for c in ['head', 'body']:
    if c not in g.columns:
        g[c] = 0
g['total'] = g['head'] + g['body']
g['head_frac'] = g['head'] / g['total']
g['purity'] = g[['head_frac']].assign(body_frac=g['body']/g['total']).max(axis=1)
n_clusters = len(g)
n_pure90 = (g['purity'] > 0.90).sum()
n_mixed = ((g['head_frac'] >= 0.40) & (g['head_frac'] <= 0.60)).sum()
n_mixed_loose = ((g['purity'] >= 0.40) & (g['purity'] <= 0.60)).sum()
print("total leiden_res1 clusters:", n_clusters)
print("clusters >90%% one tissue:", n_pure90)
print("clusters mixed (head_frac 40-60%%):", n_mixed)
print("purity distribution describe:")
print(g['purity'].describe().to_string())
# bins
bins = pd.cut(g['purity'], [0.4,0.6,0.7,0.8,0.9,0.95,1.0001], include_lowest=True)
print("\npurity bins (count of clusters):")
print(bins.value_counts().sort_index().to_string())

# For mixed clusters, how many cell types do they span?
print("\n=== mixed clusters: cell-type span (broad) ===")
mixed_clusters = g[(g['head_frac'] >= 0.40) & (g['head_frac'] <= 0.60)].index.tolist()
print("mixed cluster ids:", mixed_clusters)
broad = obs['afca_annotation_broad'].astype(str)
for cl in mixed_clusters[:15]:
    mask = (lc == cl).values
    sub_broad = broad[mask]
    # number of broad types covering 90% of cluster
    vc = sub_broad.value_counts(normalize=True)
    cum = vc.cumsum()
    n90 = (cum < 0.9).sum() + 1
    print(f"  cluster {cl}: n={mask.sum()}, head_frac={g.loc[cl,'head_frac']:.2f}, "
          f"#broad_types_to_90%={n90}, top3={list(vc.head(3).round(2).items())}")

# ---- 3. silhouette of tissue on X_harmony ----
print("\n=== 3. silhouette of tissue on X_harmony ===")
rng = np.random.default_rng(0)
n = A.n_obs
sub_idx = np.sort(rng.choice(n, size=min(40000, n), replace=False))
X = A.obsm['X_harmony'][sub_idx]
labels = tissue.values[sub_idx]
print("subsample size:", len(sub_idx), "harmony dim:", X.shape[1])
print("subsample tissue counts:", pd.Series(labels).value_counts().to_dict())
sil = silhouette_score(X, labels, metric='euclidean', sample_size=10000, random_state=0)
print("tissue silhouette on X_harmony:", round(float(sil), 4))
