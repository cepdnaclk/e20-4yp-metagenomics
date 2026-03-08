"""
PotatoSCMP Soil Metagenomics Dimensionality Reduction Analysis
Includes: PCA, PCoA, MDS, t-SNE, UMAP, PaCMAP, PHATE, SONG
With nzCLR + Matrix Completion preprocessing and Soil_type coloring
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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')



# Import after installation
import umap
import pacmap
import phate
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix
from song.song import SONG

print("✓ All packages imported successfully!\n")

# ==================== UPDATED nzCLR WITH MATRIX COMPLETION ====================
def nzclr_with_imputation(data):
    """
    Non-zero CLR with Matrix Completion:
    1. Ignore zeros
    2. Calculate geometric mean using only non-zero values
    3. Perform CLR on non-zero values (leaving zeros as NaN)
    4. Apply matrix completion to fill the gaps
    5. Re-center data to ensure sum(clr) = 0
    """
    # Ensure data is numeric numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Convert to float
    data = np.array(data, dtype=np.float64)

    # Create CLR matrix with NaNs for zeros
    clr_data = np.full_like(data, np.nan, dtype=np.float64)

    print("   Step 1: Computing non-zero CLR...")
    for i in range(data.shape[0]):
        row = data[i, :]

        # Get only non-zero values
        non_zero_vals = row[row > 0]

        if len(non_zero_vals) > 0:
            # Compute geometric mean of non-zero values only
            geo_mean = gmean(non_zero_vals)

            # Apply CLR only to non-zero values, leave zeros as NaN
            clr_data[i, :] = np.where(row > 0, np.log(row / geo_mean), np.nan)
        else:
            # If all values are zero, keep as NaN
            clr_data[i, :] = np.nan

    # Count NaN values before imputation
    nan_count = np.sum(np.isnan(clr_data))
    total_values = clr_data.size
    nan_percent = (nan_count / total_values) * 100
    print(f"   - NaN values before imputation: {nan_count} ({nan_percent:.2f}%)")

    # Apply matrix completion (iterative imputation)
    print("   Step 2: Applying matrix completion (iterative imputation)...")
    imputer = IterativeImputer(
        max_iter=1,
        random_state=42,
        initial_strategy='mean',
        verbose=0
    )
    clr_imputed = imputer.fit_transform(clr_data)

    # --- RE-CENTERING ---
    # Imputation breaks the CLR zero-sum constraint. We must re-center.
    print("   Step 3: Re-centering imputed data (Correcting CLR constraint)...")
    clr_imputed = clr_imputed - np.mean(clr_imputed, axis=1, keepdims=True)

    print(f"   ✓ Matrix completion and centering complete")

    return clr_imputed.astype(np.float32)

# ==================== DISTANCE FUNCTIONS ====================
def compute_bray_curtis(data):
    """Compute Bray-Curtis dissimilarity matrix on RAW counts"""
    if np.any(data < 0):
        # Bray-Curtis doesn't handle negatives well; if imputed values are negative, shift or use abs
        # For evaluation on raw counts, this is fine.
        raise ValueError("Bray-Curtis requires non-negative raw counts.")
    return pdist(data, metric='braycurtis')

def compute_jaccard(data):
    """Jaccard Distance (Binary) on RAW counts."""
    return pdist(data > 0, metric='jaccard')

def compute_aitchison(data_clr):
    """Aitchison distance is Euclidean distance on CLR data."""
    return pdist(data_clr, metric='euclidean')

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

    dist_condensed = compute_jaccard(data)
    dist_square = squareform(dist_condensed)

    dm = DistanceMatrix(dist_square)
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
        data_float32 = data.astype(np.float32)
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
    """Find k-nearest neighbors from a distance matrix"""
    if dist_matrix.ndim == 1:
        dist_matrix = squareform(dist_matrix)

    n = len(dist_matrix)
    k = min(k, n - 1)
    knn = []
    for i in range(n):
        neighbors = np.argpartition(dist_matrix[i], k+1)[:k+1]
        neighbors = neighbors[neighbors != i]
        knn.append(set(neighbors))
    return knn

def trustworthiness_score(X_high, X_low, k=10):
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
    dist_high = pdist(X_high, metric='euclidean')
    dist_low = pdist(X_low, metric='euclidean')
    scale = np.sum(dist_high * dist_low) / np.sum(dist_low ** 2)
    dist_low_scaled = dist_low * scale
    stress = np.sum((dist_high - dist_low_scaled) ** 2)
    normalizer = np.sum(dist_high ** 2)
    return np.sqrt(stress / normalizer) if normalizer > 0 else 0

def distance_correlation(dist_truth_condensed, X_emb):
    dist_emb = pdist(X_emb, metric='euclidean')
    corr, _ = spearmanr(dist_truth_condensed, dist_emb)
    return corr

def evaluate_embedding(X_raw, X_clr, X_emb, method_name):
    if X_emb is None:
        return None
    print(f"   Evaluating {method_name}...")
    d_bray_truth = compute_bray_curtis(X_raw)
    d_jaccard_truth = compute_jaccard(X_raw)
    d_aitchison_truth = compute_aitchison(X_clr)

    metrics = {
        'Method': method_name,
        'Trustworthiness': trustworthiness_score(X_clr, X_emb, k=10),
        'Continuity': continuity_score(X_clr, X_emb, k=10),
        'KNN Preservation': knn_preservation_score(X_clr, X_emb, k=10),
        'Normalized Stress': normalized_stress(X_clr, X_emb),
        'Bray-Curtis Corr': distance_correlation(d_bray_truth, X_emb),
        'Jaccard Corr': distance_correlation(d_jaccard_truth, X_emb),
        'Aitchison Corr': distance_correlation(d_aitchison_truth, X_emb)
    }
    print(f"   ✓ {method_name} evaluated")
    return metrics

# ==================== UPDATED: EXTRACT SOIL TYPE ====================
def extract_soil_labels(df):
    """
    Extract Soil_type labels from the dataframe.
    Looks for: 'Soil_type', 'soil_type', 'SoilType', 'Type', 'Group'
    """
    print("\nChecking for Soil_type column...")

    target_col = None
    # Priority list for soil metagenomics
    possible_cols = ['Soil_type', 'soil_type', 'SoilType', 'soiltype', 'SOIL_TYPE', 'Type', 'Group']

    for col in possible_cols:
        if col in df.columns:
            target_col = col
            break

    if target_col:
        print(f"✓ Found target metadata column: '{target_col}'")

        # Clean missing values
        no_data_values = ["nd", "na", "unknown", "-", " -", "nan", "NaN"]
        labels = df[target_col].copy().astype(str)

        for val in no_data_values:
            labels = labels.replace(val, "Unknown")

        # Fill actual NaNs
        labels = labels.fillna("Unknown")

        unique_labels = labels.unique()
        print(f"   Unique groups: {len(unique_labels)}")
        print(f"   Groups: {list(unique_labels)}")

        return labels.values, target_col
    else:
        print("⚠ No 'Soil_type' or compatible column found. Will use default coloring.")
        return None, None

# ==================== MAIN ANALYSIS ====================
def run_full_analysis(file_path):
    """Run complete dimensionality reduction comparison for PotatoSCMP"""

    print("=" * 70)
    print(f"LOADING DATA: {file_path}")
    print("=" * 70)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found. Please upload the file.")
        return None, None

    # 1. Extract Soil Metadata
    soil_labels, metadata_col = extract_soil_labels(df)

    # 2. Identify Sample ID column
    if df.columns[0].lower() in ['sample_id', 'sampleid', 'sample', 'id', '#sampleid']:
        print(f"✓ identified ID column: {df.columns[0]}")
        sample_ids = df.iloc[:, 0]
        feature_data = df.iloc[:, 1:]
    else:
        print("⚠ No ID column found at index 0, assuming index is ID.")
        sample_ids = df.index
        feature_data = df

    # 3. Drop Metadata Column from Numerical Data
    if metadata_col and metadata_col in feature_data.columns:
        feature_data = feature_data.drop(columns=[metadata_col])
        print(f"✓ Removed metadata column '{metadata_col}' from feature matrix")

    # 4. Clean Numeric Data
    # Convert to numeric, forcing errors to NaN, then fill NaN with 0 (for raw counts)
    feature_data = feature_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Drop columns that are all zeros (useless features)
    feature_data = feature_data.loc[:, (feature_data != 0).any(axis=0)]

    X_orig = feature_data.values

    print(f"\nData shape: {X_orig.shape}")
    print(f"   Samples: {X_orig.shape[0]}")
    print(f"   Features (OTUs/Genes): {X_orig.shape[1]}")

    print("\n" + "=" * 70)
    print("nzCLR WITH MATRIX COMPLETION")
    print("=" * 70)
    X_clr = nzclr_with_imputation(X_orig)
    print(f"Transformed data shape: {X_clr.shape}")

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

    print("\n" + "=" * 70)
    print("COMPUTING EVALUATION METRICS")
    print("=" * 70)

    results = []
    for name, emb in embeddings.items():
        if emb is not None:
            print(f"\n{name}:")
            metrics = evaluate_embedding(X_orig, X_clr, emb, name)
            if metrics:
                results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # Define Colors based on Soil Type
    if soil_labels is not None:
        unique_groups = sorted(list(set([d for d in soil_labels if pd.notna(d)])))

        # Use a distinct colormap
        if len(unique_groups) <= 10:
            color_palette = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        else:
            color_palette = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))

        group_to_color = {group: color_palette[i] for i, group in enumerate(unique_groups)}
        group_to_color['Unknown'] = [0.8, 0.8, 0.8, 1.0] # Grey for unknown

        colors = [group_to_color.get(d, group_to_color['Unknown']) for d in soil_labels]
    else:
        colors = range(len(X_orig))
        unique_groups = []

    # 1. Grid of embeddings
    valid_embeddings = [(name, emb) for name, emb in embeddings.items() if emb is not None]
    n_methods = len(valid_embeddings)
    n_cols = 4
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for idx, (name, emb) in enumerate(valid_embeddings):
        ax = axes[idx]
        if soil_labels is not None:
            scatter = ax.scatter(emb[:, 0], emb[:, 1], c=colors,
                               alpha=0.8, s=60, edgecolors='white', linewidth=0.5)
        else:
            scatter = ax.scatter(emb[:, 0], emb[:, 1], c=colors, cmap='viridis',
                               alpha=0.8, s=60, edgecolors='white', linewidth=0.5)

        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(alpha=0.2, linestyle='--')

    # Remove empty subplots
    for idx in range(n_methods, len(axes)):
        fig1.delaxes(axes[idx])

    # Add Legend
    if soil_labels is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=group_to_color[d], markersize=10, label=d)
                   for d in unique_groups]
        # Place legend outside
        fig1.legend(handles=handles, title='Soil Type', loc='center right',
                   bbox_to_anchor=(1.08, 0.5), fontsize=10, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig('potatoscmp_dim_reduction.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: potatoscmp_dim_reduction.png")
    plt.show()

    # 2. Metrics heatmap
    if not results_df.empty:
        metric_cols = [col for col in results_df.columns if col != 'Method']
        metrics_matrix = results_df[metric_cols].values

        fig2, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(metric_cols)))
        ax.set_yticks(np.arange(len(results_df)))
        ax.set_xticklabels(metric_cols, rotation=45, ha='right')
        ax.set_yticklabels(results_df['Method'])

        for i in range(len(results_df)):
            for j in range(len(metric_cols)):
                text = ax.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)

        ax.set_title('Method Comparison Metrics (PotatoSCMP)', fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Score')
        plt.tight_layout()
        plt.savefig('metrics_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: metrics_heatmap.png")
        plt.show()

        # 3. Results table
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(results_df.to_string(index=False))
        results_df.to_csv('dim_reduction_results.csv', index=False)
        print("\n✓ Saved: dim_reduction_results.csv")

    return results_df, embeddings

# ==================== USAGE ====================
# Ensure your file is named correctly
abundance_csv_path = 'PotatoSCMP.csv'
results_df, embeddings = run_full_analysis(abundance_csv_path)