# load_brain_vascular.py
# Quick-start Scanpy script for hackathon participants

import scanpy as sc

adata = sc.read_h5ad("brain_vascular_subsampled.h5ad")

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

sc.pl.umap(adata, color=["cell_type", "disease", "sex"])
