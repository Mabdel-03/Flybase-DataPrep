# Resources

Optional reference files for the Flybase pipeline.

## Fly marker set (optional)

`integrate_annotate.py` can run an **ORA cross-check overlay** if you supply a
Drosophila cell-type marker set as an RDS file (a named R `list`: cell-type →
character vector of gene symbols), pointed to by `FLY_MARKERS_RDS` (see
`config/paths.sh`) or `--markers-rds`.

This is **not on the critical path**. The AFCA atlas already ships peer-reviewed
cell-type labels (`afca_annotation`), which the pipeline copies into
`obs["cell_type"]` and treats as authoritative. The ORA overlay, when a marker
set is present, only writes a *separate* `obs["cell_type_ora"]` column and a
ranking CSV as a sanity cross-check — it never overwrites the atlas labels.

If no marker RDS is present here, ORA is silently skipped.
