# --- Load and preprocess ---
import scanpy as sc
import pandas as pd
import numpy as np

adata = sc.read_h5ad("brain_vascular_subsampled.h5ad")

# Normalize and log-transform
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']].copy()

# Set processed data as raw to avoid Scanpy warning
adata.raw = adata

# --- Filter to Alzheimer disease only (optional) ---
# Since you want sex differences in Alzheimer, filter for patients
adata_filtered = adata[adata.obs['disease'] == 'Alzheimer disease'].copy()

# --- Get all cell types ---
cell_types = adata_filtered.obs['cell_type'].unique()
print("Cell types found:", cell_types)

# --- Perform DE analysis for each cell type (Male vs Female) ---
all_de_results = []

for cell_type in cell_types:
    print(f"\nAnalyzing {cell_type}...")

    # Subset to current cell type
    adata_cell = adata_filtered[adata_filtered.obs['cell_type'] == cell_type].copy()
    
    # Check if both sexes are present and have enough cells
    condition_counts = adata_cell.obs['sex'].value_counts()
    if 'male' in condition_counts and 'female' in condition_counts:
        if condition_counts['male'] > 3 and condition_counts['female'] > 3:
            
            # Differential expression
            sc.tl.rank_genes_groups(
                adata_cell,
                groupby='sex',
                method='wilcoxon',
                reference='male',  # Compare female vs male
                use_raw=True
            )
            
            # Get results for female vs male
            de_df = sc.get.rank_genes_groups_df(adata_cell, group='female')
            de_df['cell_type'] = cell_type
            
            # Additional metrics
            de_df['log2_fold_change'] = np.log2(np.exp(de_df['logfoldchanges']))
            de_df['-log10_padj'] = -np.log10(de_df['pvals_adj'])
            
            all_de_results.append(de_df)
        else:
            print(f"  Skipping {cell_type}: insufficient cells per sex")
    else:
        print(f"  Skipping {cell_type}: missing one sex")

# --- Combine all results ---
if all_de_results:
    combined_de = pd.concat(all_de_results, ignore_index=True)
    
    # Filter for significant genes
    significant_genes = combined_de[
        (combined_de['pvals_adj'] < 0.05) &
        (abs(combined_de['logfoldchanges']) > 0.5)
    ].sort_values(['cell_type', 'pvals_adj'])
    
    print(f"\nFound {len(significant_genes)} significant DE genes (Female vs Male)")
    
    # Display top genes per cell type
    for cell_type in significant_genes['cell_type'].unique():
        cell_type_genes = significant_genes[significant_genes['cell_type'] == cell_type]
        print(f"\n--- {cell_type} ---")
        print("Top 10 DE genes (Female vs Male):")
        print(cell_type_genes.head(10)[['names', 'log2_fold_change', 'pvals_adj', 'scores']])
