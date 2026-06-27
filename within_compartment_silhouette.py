import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

RNG = 0
MAX_CELLS = 20000      # subsample cap per compartment
SIL_SAMPLE = 8000      # silhouette sample_size

A = ad.read_h5ad('outputs/03_Integrated/sweep/scale_dataset/fly_annotated.h5ad')

compartments = [
    'CNS neuron', 'muscle cell', 'epithelial cell',
    'fat cell', 'sensory neuron', 'glial cell',
]
# also include any other large compartments present (top by size) for completeness
extra = (A.obs['afca_annotation_broad'].value_counts()
         .index.tolist())
for c in extra:
    if c not in compartments and A.obs['afca_annotation_broad'].value_counts()[c] >= 5000:
        compartments.append(c)

broad = A.obs['afca_annotation_broad'].astype(str).values
fine = A.obs['afca_annotation'].astype(str).values
Xh = np.asarray(A.obsm['X_harmony'])
Xp = np.asarray(A.obsm['X_pca'])

rng = np.random.default_rng(RNG)
rows = []

for comp in compartments:
    mask = broad == comp
    n_total = int(mask.sum())
    if n_total == 0:
        continue
    idx = np.where(mask)[0]

    # subsample to <=MAX_CELLS
    if idx.size > MAX_CELLS:
        idx = rng.choice(idx, size=MAX_CELLS, replace=False)

    labels = fine[idx]
    # silhouette needs >=2 labels and each label needs >=1; drop singleton-only situations
    uniq, counts = np.unique(labels, return_counts=True)
    # keep only labels with >=2 cells (silhouette requires it for the sampled set; we filter to be safe)
    keep_labels = set(uniq[counts >= 2])
    sub = np.array([l in keep_labels for l in labels])
    idx = idx[sub]
    labels = labels[sub]
    n_used = idx.size
    n_fine = int(np.unique(labels).size)

    def sil(X):
        if n_fine < 2 or n_used < 3:
            return float('nan')
        ss = min(SIL_SAMPLE, n_used)
        return float(silhouette_score(X[idx], labels, metric='euclidean',
                                      sample_size=ss, random_state=RNG))

    s_h = sil(Xh)
    s_p = sil(Xp)
    rows.append(dict(compartment=comp, n_cells=n_total, n_fine_types=n_fine,
                     silhouette_fine_on_harmony=s_h, silhouette_fine_on_pca=s_p))

df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda v: f'{v:.4f}')
print(df.to_string(index=False))
