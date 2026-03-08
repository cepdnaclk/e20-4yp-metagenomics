"""
Marine Microbiome Dimensionality Reduction Comparison
Includes: PCA, PCoA, MDS, t-SNE, UMAP, PaCMAP, PHATE, SONG
With CLR preprocessing and Temperature-based coloring for oceanographic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, gmean
import warnings
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

print("=" * 70)
print("INSTALLING REQUIRED PACKAGES")
print("=" * 70)


# Import after installation
import umap
import pacmap
import phate
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix
from song.song import SONG

print("✓ All packages imported successfully!\n")

# ==================== NON-ZERO CLR TRANSFORMATION ====================
def nonzero_clr_transform(data):
    """
    Non-zero Centered Log-Ratio (CLR) transformation
    Only considers non-zero values when computing geometric mean
    """
    # Ensure data is numeric numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Convert to float and handle any remaining non-numeric values
    data = np.array(data, dtype=np.float64)

    clr_data = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[0]):
        row = data[i, :]

        # Get only non-zero values
        non_zero_vals = row[row > 0]

        if len(non_zero_vals) > 0:
            # Compute geometric mean of non-zero values only
            geo_mean = gmean(non_zero_vals)

            # Apply CLR: log(x / geometric_mean) for non-zero, 0 for zero
            clr_data[i, :] = np.where(row > 0, np.log(row / geo_mean), 0)
        else:
            # If all values are zero, keep as zero
            clr_data[i, :] = 0

    return clr_data

# ==================== DISTANCE FUNCTIONS ====================
def compute_bray_curtis(data):
    """Compute Bray-Curtis dissimilarity matrix"""
    return squareform(pdist(data, metric='braycurtis'))

def compute_jaccard(data):
    """
    Jaccard distance
    """
    return squareform(pdist(data > 0, metric='jaccard'))

def compute_aitchison(data):
    """Aitchison distance on CLR-transformed data"""
    clr_data = nonzero_clr_transform(data)
    return squareform(pdist(clr_data, metric='euclidean'))

# ==================== DIMENSIONALITY REDUCTION METHODS ====================
def run_pca(data, n_components=2):
    """Principal Component Analysis"""
    print(f"   Running PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    result = pca.fit_transform(data)
    print(f"   ✓ PCA complete: {result.shape}")
    return result

def run_pcoa(data, n_components=2):
    """Principal Coordinates Analysis (Jaccard PCoA)"""
    print(f"   Running PCoA with Jaccard distance...")
    dist_matrix = compute_jaccard(data)
    dm = DistanceMatrix(dist_matrix)
    pcoa_result = pcoa(dm, number_of_dimensions=n_components)
    result = pcoa_result.samples.values
    print(f"   ✓ PCoA complete: {result.shape}")
    return result

def run_mds(data, n_components=2):
    """Multidimensional Scaling"""
    print(f"   Running MDS...")
    mds = MDS(n_components=n_components, random_state=42, dissimilarity='euclidean', n_jobs=-1)
    result = mds.fit_transform(data)
    print(f"   ✓ MDS complete: {result.shape}")
    return result

def run_tsne(data, n_components=2):
    """t-SNE"""
    print(f"   Running t-SNE...")
    perplexity = min(30, max(5, len(data) - 1))
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    result = tsne.fit_transform(data)
    print(f"   ✓ t-SNE complete: {result.shape}")
    return result

def run_umap(data, n_components=2):
    """UMAP"""
    print(f"   Running UMAP...")
    n_neighbors = min(15, max(2, len(data) - 1))
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
    result = reducer.fit_transform(data)
    print(f"   ✓ UMAP complete: {result.shape}")
    return result

def run_pacmap(data, n_components=2):
    """PaCMAP"""
    print(f"   Running PaCMAP...")
    n_neighbors = min(15, max(2, len(data) - 1))
    reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    result = reducer.fit_transform(data)
    print(f"   ✓ PaCMAP complete: {result.shape}")
    return result

def run_phate(data, n_components=2):
    """PHATE"""
    print(f"   Running PHATE...")
    phate_op = phate.PHATE(n_components=n_components, random_state=42, n_jobs=-1, verbose=0)
    result = phate_op.fit_transform(data)
    print(f"   ✓ PHATE complete: {result.shape}")
    return result

def run_song(data, n_components=2):
    """SONG - Self Organizing Neural Graphs"""
    print(f"   Running SONG...")
    try:
        # Ensure data is float32 as required by SONG
        data_float32 = data.astype(np.float32)

        # Use parameters similar to the working example
        n_neighbors = min(2, max(1, len(data) // 20))
        song = SONG(
            n_components=n_components,
            n_neighbors=n_neighbors,
            lr=0.5,
            random_seed=42,
            verbose=0
        )
        result = song.fit_transform(data_float32)
        print(f"   ✓ SONG complete: {result.shape}")
        return result
    except Exception as e:
        print(f"   ✗ SONG failed: {e}")
        return None

# ==================== EVALUATION METRICS ====================
def compute_knn(dist_matrix, k=10):
    """Find k-nearest neighbors"""
    n = len(dist_matrix)
    k = min(k, n - 1)  # Ensure k is valid
    knn = []
    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1:k+1]  # Exclude self
        knn.append(set(neighbors))
    return knn

def trustworthiness_score(X_high, X_low, k=10):
    """Trustworthiness metric"""
    n = len(X_high)
    k = min(k, n - 1)
    dist_high = pairwise_distances(X_high)
    dist_low = pairwise_distances(X_low)

    knn_low = compute_knn(dist_low, k)

    trust = 0
    for i in range(n):
        knn_high_i = set(np.argsort(dist_high[i])[1:k+1])
        for j in knn_low[i]:
            if j not in knn_high_i:
                rank = np.where(np.argsort(dist_high[i]) == j)[0][0]
                trust += max(0, rank - k)

    trust = 1 - (2 / (n * k * (2*n - 3*k - 1))) * trust
    return trust

def continuity_score(X_high, X_low, k=10):
    """Continuity metric"""
    n = len(X_high)
    k = min(k, n - 1)
    dist_high = pairwise_distances(X_high)
    dist_low = pairwise_distances(X_low)

    knn_high = compute_knn(dist_high, k)

    cont = 0
    for i in range(n):
        knn_low_i = set(np.argsort(dist_low[i])[1:k+1])
        for j in knn_high[i]:
            if j not in knn_low_i:
                rank = np.where(np.argsort(dist_low[i]) == j)[0][0]
                cont += max(0, rank - k)

    cont = 1 - (2 / (n * k * (2*n - 3*k - 1))) * cont
    return cont

def knn_preservation_score(X_high, X_low, k=10):
    """KNN preservation metric"""
    n = len(X_high)
    k = min(k, n - 1)
    dist_high = pairwise_distances(X_high)
    dist_low = pairwise_distances(X_low)

    knn_high = compute_knn(dist_high, k)
    knn_low = compute_knn(dist_low, k)

    preserved = sum(len(knn_high[i] & knn_low[i]) for i in range(n))
    total = n * k

    return preserved / total

def normalized_stress(X_high, X_low):
    """Normalized stress"""
    dist_high = pairwise_distances(X_high).flatten()
    dist_low = pairwise_distances(X_low).flatten()

    stress = np.sum((dist_high - dist_low) ** 2)
    normalizer = np.sum(dist_high ** 2)

    return np.sqrt(stress / normalizer) if normalizer > 0 else 0

def distance_correlation(X_orig, X_emb, metric='braycurtis'):
    """Correlation between original and embedding distances"""
    if metric == 'braycurtis':
        dist_orig = compute_bray_curtis(X_orig)
    elif metric == 'aitchison':
        dist_orig = compute_aitchison(X_orig)
    elif metric == 'jaccard':
        dist_orig = compute_jaccard(X_orig)
    else:
        dist_orig = pairwise_distances(X_orig)

    dist_emb = pairwise_distances(X_emb)

    # Flatten and compute correlation
    corr, _ = spearmanr(dist_orig.flatten(), dist_emb.flatten())
    return corr

def evaluate_embedding(X_orig, X_emb, method_name):
    """Compute all evaluation metrics"""
    if X_emb is None:
        return None

    print(f"   Evaluating {method_name}...")

    metrics = {
        'Method': method_name,
        'Trustworthiness': trustworthiness_score(X_orig, X_emb, k=10),
        'Continuity': continuity_score(X_orig, X_emb, k=10),
        'KNN Preservation': knn_preservation_score(X_orig, X_emb, k=10),
        'Normalized Stress': normalized_stress(X_orig, X_emb),
        'Bray-Curtis Corr': distance_correlation(X_orig, X_emb, 'braycurtis'),
        'Aitchison Corr': distance_correlation(X_orig, X_emb, 'aitchison'),
        'Jaccard Corr': distance_correlation(X_orig, X_emb, 'jaccard')
    }

    print(f"   ✓ {method_name} evaluated")
    return metrics

# ==================== MAIN ANALYSIS ====================
def run_full_analysis(csv_path):
    """Run complete dimensionality reduction comparison"""

    # Load data
    print("=" * 70)
    print("LOADING AND CLEANING DATA")
    print("=" * 70)
    df = pd.read_csv(csv_path)

    # --- 1. CLEANING STEP ---
    print("   Performing data cleaning...")
    df = df.replace("nd", np.nan)
    df = df.replace("na", np.nan)
    df = df.replace("-", np.nan)
    df = df.replace(' -', np.nan)
    df = df.replace('unknown', np.nan)

    # --- 2. IDENTIFY FEATURE COLUMNS (motu_linkage_group_*) ---
    # All columns starting with 'motu_linkage_group_' are features
    feature_cols = [col for col in df.columns if col.startswith('motu_linkage_group_')]
    
    print(f"   ✓ Found {len(feature_cols)} MOTU linkage group features")

    # --- 3. EXTRACT TEMPERATURE METADATA ---
    temp_column = None
    temp_col_name = None
    
    # Look for temperature column - try multiple possible names
    possible_temp_names = ['Mean_Temperature [deg C]*', 'Temperature', 'Mean_Temperature', 
                          'temperature', 'temp', 'Temp']
    
    for col in df.columns:
        if col in possible_temp_names or 'temperature' in col.lower():
            temp_col_name = col
            break
            
    if temp_col_name:
        print(f"   ✓ Found Temperature column: '{temp_col_name}'")
        temp_column = pd.to_numeric(df[temp_col_name], errors='coerce')
        # Fill missing temperature values with median
        temp_column = temp_column.fillna(temp_column.median())
    else:
        print("   ! Warning: Temperature column not found. Points will be colored uniformly.")
        temp_column = pd.Series([20.0] * len(df))  # Default temperature

    # --- 4. EXTRACT SAMPLE IDENTIFIERS ---
    # Use the first column as sample identifier if it's not a feature
    if df.columns[0] not in feature_cols:
        sample_ids = df.iloc[:, 0]
        print(f"   ✓ Using '{df.columns[0]}' as sample identifier")
    else:
        sample_ids = df.index
        print("   ✓ Using row index as sample identifier")

    # --- 5. EXTRACT FEATURE DATA ONLY ---
    feature_data = df[feature_cols].copy()

    # Convert to numeric, coercing errors to NaN, then filling NaNs with 0
    feature_data = feature_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_orig = feature_data.values

    print(f"\nData shape: {X_orig.shape}")
    print(f"   Samples: {X_orig.shape[0]}")
    print(f"   Features (MOTU groups): {X_orig.shape[1]}")
    print(f"   Temperature range: {temp_column.min():.2f}°C to {temp_column.max():.2f}°C")

    # CLR transformation
    print("\n" + "=" * 70)
    print("NON-ZERO CLR TRANSFORMATION")
    print("=" * 70)
    X_clr = nonzero_clr_transform(X_orig)
    print(f"Non-zero CLR-transformed data shape: {X_clr.shape}")

    # Run all methods
    print("\n" + "=" * 70)
    print("RUNNING DIMENSIONALITY REDUCTION METHODS")
    print("=" * 70)

    methods = {
        'PCA': lambda: run_pca(X_clr),
        'Jaccard PCoA': lambda: run_pcoa(X_orig),
        'MDS': lambda: run_mds(X_clr),
        't-SNE': lambda: run_tsne(X_clr),
        'UMAP': lambda: run_umap(X_clr),
        'PaCMAP': lambda: run_pacmap(X_clr),
        'PHATE': lambda: run_phate(X_clr),
        'SONG': lambda: run_song(X_clr)
    }

    embeddings = {}
    for name, func in methods.items():
        print(f"\n{name}:")
        try:
            embeddings[name] = func()
        except Exception as e:
            print(f"   ✗ {name} error: {e}")
            embeddings[name] = None

    # Evaluate all embeddings
    print("\n" + "=" * 70)
    print("COMPUTING EVALUATION METRICS")
    print("=" * 70)

    results = []
    for name, emb in embeddings.items():
        if emb is not None:
            print(f"\n{name}:")
            metrics = evaluate_embedding(X_clr, emb, name)
            if metrics:
                results.append(metrics)

    results_df = pd.DataFrame(results)

    # Visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # --- 6. EMBEDDING GRID (Color by Temperature) ---
    valid_embeddings = [(name, emb) for name, emb in embeddings.items() if emb is not None]
    n_methods = len(valid_embeddings)
    n_cols = 4
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Use a continuous colormap for temperature
    temp_min, temp_max = temp_column.min(), temp_column.max()

    for idx, (name, emb) in enumerate(valid_embeddings):
        ax = axes[idx]
        
        # Scatter plot with temperature gradient coloring
        scatter = ax.scatter(emb[:, 0], emb[:, 1], 
                           c=temp_column, 
                           cmap='coolwarm',
                           vmin=temp_min, vmax=temp_max,
                           alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=10)
        ax.set_ylabel('Component 2', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar to each subplot
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (°C)', fontsize=9)

    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('dimensionality_reduction_grid.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: dimensionality_reduction_grid.png")
    plt.show()

    # --- 7. METRICS GRID (4 Top, 3 Bottom) ---
    metric_names = [
        'Trustworthiness', 'Continuity', 'KNN Preservation', 'Normalized Stress',
        'Bray-Curtis Corr', 'Aitchison Corr', 'Jaccard Corr'
    ]

    fig2 = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig2)
    
    ax_positions = [
        gs[0, 0], gs[0, 1], gs[0, 2], gs[0, 3],
        gs[1, 0], gs[1, 1], gs[1, 2]
    ]
    
    colors = plt.cm.Set3(range(len(results_df)))

    for idx, metric in enumerate(metric_names):
        if idx < 4:
            ax = fig2.add_subplot(gs[0, idx])
        else:
            ax = fig2.add_subplot(gs[1, idx - 4])

        values = results_df[metric].values
        methods_list = results_df['Method'].values

        bars = ax.barh(methods_list, values, color=colors)
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        
        if metric == 'Normalized Stress':
             ax.set_xlim([0, max(values) * 1.2])
             ax.set_title(f"{metric} (Lower is better)", fontsize=10)
        else:
             ax.set_xlim([0, 1.1])
        
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + (0.01 if val < 1 else val*0.01), i, f'{val:.3f}',
                   va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('evaluation_metrics_combined.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: evaluation_metrics_combined.png")
    plt.show()

    # --- 8. Summary heatmap ---
    fig4, ax = plt.subplots(figsize=(12, 8))

    heatmap_data = results_df.set_index('Method').T

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.5, ax=ax, cbar_kws={'label': 'Score'},
                linewidths=0.5)
    ax.set_title('Dimensionality Reduction Metrics Heatmap',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: metrics_heatmap.png")
    plt.show()

    # Save results
    results_df.to_csv('evaluation_results.csv', index=False)
    print("✓ Saved: evaluation_results.csv")

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    return results_df, embeddings

# ==================== RUN ANALYSIS ====================
if __name__ == "__main__":
    # Specify your CSV path
    csv_path = 'abundance.csv'

    # If you need to upload file in Colab, uncomment:
    # from google.colab import files
    # uploaded = files.upload()
    # csv_path = list(uploaded.keys())[0]

    print("\n" + "=" * 70)
    print("MARINE MICROBIOME DIMENSIONALITY REDUCTION COMPARISON")
    print("=" * 70)

    results_df, embeddings = run_full_analysis(csv_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. dimensionality_reduction_grid.png (Colored by Temperature)")
    print("  2. evaluation_metrics_combined.png (4 Top / 3 Bottom Layout)")
    print("  3. metrics_heatmap.png")
    print("  4. evaluation_results.csv")
    print("\n" + "=" * 70)