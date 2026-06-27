import anndata as ad, numpy as np, pandas as pd

A = ad.read_h5ad('outputs/03_Integrated/sweep/scale_dataset/fly_annotated.h5ad', backed='r')
obs = A.obs
tissue = obs['tissue'].astype(str)
lc = obs['leiden_res1'].astype(str)
broad = obs['afca_annotation_broad'].astype(str)

g = pd.crosstab(lc, tissue)
for c in ['head','body']:
    if c not in g.columns: g[c]=0
g['total']=g['head']+g['body']
g['head_frac']=g['head']/g['total']
g['purity']=np.maximum(g['head_frac'], g['body']/g['total'])

# tissue-mixed clusters = purity < 0.90
mixed = g[g['purity'] < 0.90].sort_values('purity')
print("=== tissue-mixed clusters (purity<0.90):", len(mixed), "===")
print("of these, how many ALSO span multiple cell types (need >=2 broad types to reach 90%)")
multi = 0
for cl in mixed.index:
    mask=(lc==cl).values
    vc=broad[mask].value_counts(normalize=True)
    n90=(vc.cumsum()<0.9).sum()+1
    if n90>=2: multi+=1
    print(f"  cl {cl}: n={int(mixed.loc[cl,'total'])}, purity={mixed.loc[cl,'purity']:.2f}, "
          f"head_frac={mixed.loc[cl,'head_frac']:.2f}, #broad_to90={n90}, top={list(vc.head(2).round(2).items())}")
print("\n# tissue-mixed clusters that are ALSO multi-celltype (true smearing):", multi)
