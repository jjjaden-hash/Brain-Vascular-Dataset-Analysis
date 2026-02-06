# Load and preprocess
import scanpy as sc
import pandas as pd
import numpy as np

adata = sc.read_h5ad("brain_vascular_subsampled.h5ad")

# Preprocess
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']].copy()

# Set processed data as raw
adata.raw = adata

# Filter for conditions
adata_filtered = adata[adata.obs['disease'].isin(['Alzheimer disease', 'normal'])].copy()

# Get all cell types
cell_types = adata_filtered.obs['cell_type'].unique()
print("Cell types found:", cell_types)

# Perform DE analysis for each cell type
all_de_results = []

for cell_type in cell_types:
    print(f"Analyzing {cell_type}...")
    
    # Subset to current cell type
    adata_cell = adata_filtered[adata_filtered.obs['cell_type'] == cell_type].copy()
    
    # Check if we have both conditions in this cell type
    condition_counts = adata_cell.obs['disease'].value_counts()
    if 'normal' in condition_counts and 'Alzheimer disease' in condition_counts:
        if condition_counts['normal'] > 3 and condition_counts['Alzheimer disease'] > 3:
            
            # Perform differential expression
            sc.tl.rank_genes_groups(
                adata_cell, 
                groupby='disease', 
                method='wilcoxon',
                reference='normal'
            )
            
            # Get results
            de_df = sc.get.rank_genes_groups_df(adata_cell, group='Alzheimer disease')
            de_df['cell_type'] = cell_type
            
            # Add additional metrics
            de_df['log2_fold_change'] = np.log2(np.exp(de_df['logfoldchanges']))
            de_df['-log10_padj'] = -np.log10(de_df['pvals_adj'])
            
            all_de_results.append(de_df)
        else:
            print(f"  Skipping {cell_type}: insufficient cells per condition")
    else:
        print(f"  Skipping {cell_type}: missing one condition")

# Combine all results
if all_de_results:
    combined_de = pd.concat(all_de_results, ignore_index=True)
    
    # Filter for significant genes (adjust threshold as needed)
    significant_genes = combined_de[
        (combined_de['pvals_adj'] < 0.05) & 
        (abs(combined_de['logfoldchanges']) > 0.5)  # Fold change filter
    ].sort_values(['cell_type', 'pvals_adj'])
    
    print(f"Found {len(significant_genes)} significant DE genes")
    
    # Display top genes per cell type
    for cell_type in significant_genes['cell_type'].unique():
        cell_type_genes = significant_genes[significant_genes['cell_type'] == cell_type]
        print(f"\n--- {cell_type} ---")
        print(f"Top 10 DE genes (Alzheimer vs Normal):")
        print(cell_type_genes.head(10)[['names', 'logfoldchanges', 'pvals_adj', 'scores']])