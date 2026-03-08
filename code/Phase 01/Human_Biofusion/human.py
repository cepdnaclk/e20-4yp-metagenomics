import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import umap
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("METAGENOMICS DIMENSIONALITY REDUCTION PIPELINE")
print("="*70)

# Step 1: Load and separate metadata from abundance data
print("\nStep 1: Loading data and separating metadata...")
df = pd.read_csv('abundance.csv', index_col=0)
print(f"Original shape: {df.shape}")

# Identify taxonomic abundance columns (start with 'k__')
abundance_cols = [col for col in df.columns if col.startswith('k__')]
metadata_cols = [col for col in df.columns if not col.startswith('k__')]

print(f"  - Metadata columns: {len(metadata_cols)}")
print(f"  - Abundance columns: {len(abundance_cols)}")

# Separate metadata and abundance data
metadata_df = df[metadata_cols].copy()
abundance_df = df[abundance_cols].copy()

# Convert abundance to numeric (should already be, but just in case)
abundance_df = abundance_df.apply(pd.to_numeric, errors='coerce')

print(f"\nAbundance data shape: {abundance_df.shape}")
print(f"Value range: [{abundance_df.min().min():.4f}, {abundance_df.max().max():.4f}]")

# Step 2: Filter low-prevalence features (optional but recommended)
print("\nStep 2: Filtering low-prevalence features...")
prevalence_threshold = 0.05  # Keep features present in ≥5% of samples
min_samples = int(prevalence_threshold * len(abundance_df))

# Count non-zero occurrences per feature
feature_prevalence = (abundance_df > 0).sum()
features_to_keep = feature_prevalence >= min_samples

print(f"  - Prevalence threshold: {prevalence_threshold*100}% ({min_samples} samples)")
print(f"  - Features before filtering: {len(abundance_df.columns)}")
print(f"  - Features after filtering: {features_to_keep.sum()}")

abundance_filtered = abundance_df.loc[:, features_to_keep].copy()

# Step 3: Replace zeros with NaN (treat as missing)
print("\nStep 3: Replacing zeros with NaN...")
data_matrix = abundance_filtered.values.copy()
data_matrix[data_matrix == 0] = np.nan

zero_percent = np.sum(np.isnan(data_matrix)) / data_matrix.size * 100
print(f"  - Missing/zero values: {zero_percent:.2f}%")

# Step 4: RCLR transformation
print("\nStep 4: Applying RCLR transformation...")

def rclr_transform(X):
    """
    Robust Centered Log Ratio transformation for RELATIVE ABUNDANCE data
    Handles the data correctly since it's already proportions (0-100)
    """
    X_rclr = np.zeros_like(X)
    
    # Convert percentages to proportions if needed (0-1 range)
    # If max > 1, assume it's 0-100 scale
    if np.nanmax(X) > 1:
        X = X / 100.0
    
    # For each sample (row)
    for i in range(X.shape[0]):
        row = X[i, :]
        observed = row[~np.isnan(row)]
        
        if len(observed) > 0 and np.all(observed > 0):
            # Geometric mean of observed values
            geom_mean_row = np.exp(np.mean(np.log(observed)))
            
            # Log transform and center by row
            for j in range(X.shape[1]):
                if not np.isnan(row[j]) and row[j] > 0:
                    X_rclr[i, j] = np.log(row[j]) - np.log(geom_mean_row)
                else:
                    X_rclr[i, j] = np.nan
    
    # Center by column means (computed on non-NaN values)
    for j in range(X_rclr.shape[1]):
        col = X_rclr[:, j]
        observed_col = col[~np.isnan(col)]
        if len(observed_col) > 0:
            col_mean = np.mean(observed_col)
            X_rclr[:, j] = np.where(~np.isnan(col), col - col_mean, np.nan)
    
    return X_rclr

rclr_data = rclr_transform(data_matrix)
print(f"  - RCLR shape: {rclr_data.shape}")
print(f"  - RCLR range: [{np.nanmin(rclr_data):.2f}, {np.nanmax(rclr_data):.2f}]")
print(f"  - RCLR mean: {np.nanmean(rclr_data):.4f}")

# Step 5: Matrix completion
print("\nStep 5: Matrix completion using iterative imputation...")

imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    verbose=0,
    tol=0.001
)

completed_data = imputer.fit_transform(rclr_data)
print(f"  - Completed shape: {completed_data.shape}")
print(f"  - Completed range: [{completed_data.min():.2f}, {completed_data.max():.2f}]")
print(f"  - NaN remaining: {np.isnan(completed_data).any()}")

# Step 6: Dimensionality reduction
print("\nStep 6: Running dimensionality reduction...")

# UMAP
print("  - Running UMAP...")
umap_reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42,
    verbose=False
)
umap_embedding = umap_reducer.fit_transform(completed_data)

# t-SNE
print("  - Running t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    n_iter=1000,
    verbose=0
)
tsne_embedding = tsne.fit_transform(completed_data)

print("  ✓ Dimensionality reduction complete")

# Step 7: Create visualizations
print("\nStep 7: Creating visualizations...")

# Color by disease column
color_by = 'disease'
print(f"  - Coloring by: {color_by}")

if color_by in metadata_df.columns:
    # Clean disease labels (replace 'nd', 'na', etc. with 'unknown')
    disease_labels = metadata_df[color_by].copy()
    disease_labels = disease_labels.replace(['nd', 'na', '-', 'unknown'], 'unknown')
    
    # Get unique diseases
    unique_diseases = disease_labels.unique()
    n_diseases = len(unique_diseases)
    print(f"  - Number of disease categories: {n_diseases}")
    print(f"  - Disease categories: {sorted([str(d) for d in unique_diseases[:10]])}")
    
    # Convert to categorical codes for coloring
    color_values = pd.Categorical(disease_labels).codes
    color_label = color_by
else:
    print(f"  - Warning: '{color_by}' column not found, using sample index")
    color_values = np.arange(len(umap_embedding))
    color_label = 'Sample Index'

# Create plots with disease coloring
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Choose colormap based on number of categories
if n_diseases <= 10:
    cmap = 'tab10'
elif n_diseases <= 20:
    cmap = 'tab20'
else:
    cmap = 'viridis'

# UMAP plot
scatter1 = axes[0].scatter(
    umap_embedding[:, 0], 
    umap_embedding[:, 1], 
    c=color_values,
    cmap=cmap,
    alpha=0.7, 
    s=40,
    edgecolors='black',
    linewidth=0.5
)
axes[0].set_xlabel('UMAP 1', fontsize=13, fontweight='bold')
axes[0].set_ylabel('UMAP 2', fontsize=13, fontweight='bold')
axes[0].set_title(f'UMAP Projection (colored by {color_by})\n{abundance_filtered.shape[0]} samples × {abundance_filtered.shape[1]} features', 
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# t-SNE plot
scatter2 = axes[1].scatter(
    tsne_embedding[:, 0], 
    tsne_embedding[:, 1], 
    c=color_values,
    cmap=cmap,
    alpha=0.7, 
    s=40,
    edgecolors='black',
    linewidth=0.5
)
axes[1].set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
axes[1].set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
axes[1].set_title(f't-SNE Projection (colored by {color_by})\n{abundance_filtered.shape[0]} samples × {abundance_filtered.shape[1]} features',
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Create custom legend for diseases
if color_by == 'disease' and n_diseases <= 30:
    from matplotlib.patches import Patch
    import matplotlib.cm as cm
    
    # Get colormap
    cmap_obj = cm.get_cmap(cmap)
    
    # Create legend handles
    legend_elements = []
    for i, disease in enumerate(sorted(unique_diseases)):
        if n_diseases <= 20:
            color = cmap_obj(i / max(n_diseases-1, 1))
        else:
            # For continuous colormap
            code = pd.Categorical(disease_labels).categories.get_loc(disease)
            color = cmap_obj(code / max(n_diseases-1, 1))
        
        # Count samples per disease
        count = (disease_labels == disease).sum()
        legend_elements.append(Patch(facecolor=color, edgecolor='black', 
                                     label=f'{disease} (n={count})'))
    
    # Add legend below the plots
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=min(5, n_diseases), fontsize=9,
              frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, -0.05))
    
    plt.subplots_adjust(bottom=0.15)
else:
    plt.colorbar(scatter2, ax=axes[1], label=color_label)

plt.tight_layout()
plt.savefig('dimensionality_reduction_results.png', dpi=300, bbox_inches='tight')
print("  - Saved: dimensionality_reduction_results.png")
plt.show()

# Step 8: Save results
print("\nStep 8: Saving results...")

# Save embeddings with disease labels
umap_df = pd.DataFrame(
    umap_embedding,
    index=abundance_filtered.index,
    columns=['UMAP1', 'UMAP2']
)
umap_df['disease'] = disease_labels.values

tsne_df = pd.DataFrame(
    tsne_embedding,
    index=abundance_filtered.index,
    columns=['tSNE1', 'tSNE2']
)
tsne_df['disease'] = disease_labels.values

# Merge with metadata for easier analysis
results_df = pd.concat([metadata_df, umap_df, tsne_df], axis=1)
results_df.to_csv('embeddings_with_metadata.csv')
print("  - Saved: embeddings_with_metadata.csv")

# Save just embeddings
umap_df.to_csv('umap_embeddings.csv')
tsne_df.to_csv('tsne_embeddings.csv')
print("  - Saved: umap_embeddings.csv")
print("  - Saved: tsne_embeddings.csv")

# Save preprocessed data
preprocessed_df = pd.DataFrame(
    completed_data,
    index=abundance_filtered.index,
    columns=abundance_filtered.columns
)
preprocessed_df.to_csv('rclr_completed_data.csv')
print("  - Saved: rclr_completed_data.csv")

# Save filtered feature list
feature_info = pd.DataFrame({
    'feature': abundance_filtered.columns,
    'prevalence': (abundance_filtered > 0).sum().values,
    'mean_abundance': abundance_filtered.mean().values,
    'median_abundance': abundance_filtered.median().values
})
feature_info.to_csv('filtered_features_info.csv', index=False)
print("  - Saved: filtered_features_info.csv")

# Summary
print("\n" + "="*70)
print("PIPELINE SUMMARY")
print("="*70)
print(f"Original samples:           {df.shape[0]}")
print(f"Original features:          {len(abundance_cols)}")
print(f"Filtered features:          {abundance_filtered.shape[1]}")
print(f"Missing data (original):    {zero_percent:.2f}%")
print(f"Metadata columns:           {len(metadata_cols)}")
print(f"UMAP components:            2")
print(f"t-SNE components:           2")
print(f"Visualization colored by:   {color_by} ({n_diseases} categories)")
print("="*70)

print("\n✓ Pipeline completed successfully!")
print("\nOutput files:")
print("  1. embeddings_with_metadata.csv - Full results with metadata")
print("  2. umap_embeddings.csv - UMAP coordinates only")
print("  3. tsne_embeddings.csv - t-SNE coordinates only")
print("  4. rclr_completed_data.csv - Preprocessed abundance data")
print("  5. filtered_features_info.csv - Information about retained features")
print("  6. dimensionality_reduction_results.png - Visualization")