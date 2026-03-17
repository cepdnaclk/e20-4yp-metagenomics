"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MICROBIOME — UNIFIED PREPROCESSING + GRAPH CONSTRUCTION PIPELINE         ║
║   Prevalence Filter → TSS → nzCLR → 8 DR Methods + 2 Graph Types          ║
║                                                                              ║
║   DR Methods  : PCA · PCoA · MDS · t-SNE · UMAP · PaCMAP · PHATE · SONG   ║
║   Graph Types : KNN cosine graph  |  Bipartite graph                        ║
║   Outputs for : K-Means × 8 DR methods                                      ║
║               + DMoN    × KNN graph + Bipartite graph                       ║
║               + MinCutPool × KNN graph + Bipartite graph                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies
────────────
  pip install numpy pandas scikit-learn scipy matplotlib seaborn
  pip install umap-learn pacmap phate scikit-bio
  pip install song-vis   # for SONG (optional — skipped if missing)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
from pathlib import Path
from scipy.stats import gmean, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "abundance_csv"        : "abundance.csv",
    "output_dir"           : "preprocessed",
    "metadata_cols"        : 210,
    "disease_col_keyword"  : "disease",
    # Prevalence filter
    "prevalence_threshold" : 0.10,
    "min_abundance_thresh" : 0.0,
    # TSS
    "pseudocount"          : 0.0,
    # Aitchison PCA (for GNN node features)
    "pca_variance_target"  : 0.80,
    "pca_max_components"   : 100,
    "pca_dmon_components"  : 50,
    # KNN graph
    "knn_k"                : 10,
    "knn_mutual"           : True,
    # Bipartite graph
    "bipartite_threshold"  : 0.0,   # CLR > threshold → edge exists
    # t-SNE
    "tsne_perplexity"      : 30,
    "tsne_n_iter"          : 1000,
    # UMAP
    "umap_n_neighbors"     : 15,
    "umap_min_dist"        : 0.1,
}

DARK   = "#0d0d1a";  PANEL  = "#13132b";  GRID   = "#2a2a4a"
TEXT   = "#e8e8ff";  TEXT2  = "#9090c0"
COLORS = ["#00d4ff","#ff6b6b","#ffd700","#b388ff","#69ff85",
          "#ff9f43","#fd79a8","#a29bfe","#55efc4","#fdcb6e",
          "#e17055","#74b9ff","#00cec9","#6c5ce7","#fab1a0",
          "#dfe6e9","#2d3436","#636e72"]

def _sec(t): print(f"\n{'═'*64}\n  {t}\n{'═'*64}")
def _log(t, m): print(f"  [{t}]{'─'*max(52-len(t),1)} {m}")

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT2, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT2, fontsize=8)
    ax.grid(True, color=GRID, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    if title: ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
# Noise labels that should not count as disease classes
_NOISE = {"nd", "na", "-", " -", "unknown", "nan", "none", ""}

def _clean_disease(series):
    """Strip, lowercase, map noisy labels to 'unknown'."""
    s = series.astype(str).str.strip().str.lower()
    return s.where(~s.isin(_NOISE), other="unknown")

def load_and_validate(cfg):
    _sec("STEP 1 — Load & validate")
    df = pd.read_csv(cfg["abundance_csv"], low_memory=False)
    _log("raw shape", str(df.shape))

    n_meta    = cfg["metadata_cols"]
    metadata  = df.iloc[:, :n_meta].copy()
    abundance = df.iloc[:, n_meta:].copy()

    # Clean abundance columns only — do NOT replace in metadata
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)

    # Locate disease column
    dis_col = next((c for c in metadata.columns
                    if cfg["disease_col_keyword"].lower() in c.lower()), None)
    if dis_col is None:
        raise ValueError("No disease column found in metadata.")

    # Clean disease labels: strip whitespace, lowercase, map noise -> 'unknown'
    cleaned        = _clean_disease(metadata[dis_col])
    disease_series = cleaned.copy()

    le     = LabelEncoder()
    y_true = le.fit_transform(cleaned)

    _log("disease col", f"\'{dis_col}\'  —  {len(le.classes_)} classes: {list(le.classes_)}")
    for cls, cnt in zip(le.classes_, np.bincount(y_true)):
        _log("  class", f"{cls:<32} n={cnt}")

    return metadata, abundance, y_true, le, disease_series


def prevalence_filter(abundance, threshold=0.10, min_abund=0.0):
    _sec("STEP 2 — Prevalence filter")
    X         = abundance.values.astype(np.float64)
    prevalence = (X > min_abund).mean(axis=0)
    keep_mask  = prevalence >= threshold
    _log("before", f"{abundance.shape[1]} taxa")
    _log("after ", f"{keep_mask.sum()} taxa kept  ({(~keep_mask).sum()} dropped)")
    filtered = abundance.loc[:, keep_mask]
    return filtered, {"prevalence_vector": prevalence, "keep_mask": keep_mask,
                      "n_before": int(abundance.shape[1]), "n_after": int(keep_mask.sum())}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TSS NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def tss_normalise(abundance, pseudocount=0.0):
    _sec("STEP 3 — TSS normalisation")
    X = abundance.values.astype(np.float64) + pseudocount
    row_sums = X.sum(axis=1, keepdims=True)
    X_tss    = X / (row_sums + 1e-12)
    _log("shape",  f"{X_tss.shape}")
    _log("verify", f"row-sum mean={X_tss.sum(axis=1).mean():.6f}")
    return X_tss

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — nzCLR TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────
def nonzero_clr_transform(data):
    """Non-zero CLR: zero stays 0; non-zero → log(x / gmean_of_nonzeros)"""
    if isinstance(data, pd.DataFrame): data = data.values
    data     = np.array(data, dtype=np.float64)
    clr_data = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        row = data[i, :]
        nz  = row[row > 0]
        if len(nz) > 0:
            g = gmean(nz)
            clr_data[i, :] = np.where(row > 0, np.log(row / g), 0.0)
    return clr_data

def run_nzclr(X_tss):
    _sec("STEP 4 — nzCLR transformation")
    X_clr = nonzero_clr_transform(X_tss)
    _log("shape",   f"{X_clr.shape}  dtype={X_clr.dtype}")
    _log("range",   f"[{X_clr.min():.4f}, {X_clr.max():.4f}]")
    _log("zero %",  f"{(X_clr==0).mean():.2%}")
    return X_clr

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — AITCHISON PCA  (for GNN node features)
# ─────────────────────────────────────────────────────────────────────────────
def aitchison_pca(X_clr, variance_target=0.80, max_components=100, dmon_components=50):
    _sec("STEP 5 — Aitchison PCA (GNN node features)")
    n_max  = min(max_components, X_clr.shape[0]-1, X_clr.shape[1])
    pca    = PCA(n_components=n_max, random_state=42)
    X_full = pca.fit_transform(X_clr)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_tgt  = min(int(np.searchsorted(cumvar, variance_target))+1, n_max)
    n_dmon = min(dmon_components, n_max)
    _log(f"PCs for {variance_target:.0%} var", f"{n_tgt}  (cumvar={cumvar[n_tgt-1]:.3f})")
    _log("DMoN node features",                 f"{n_dmon} PCs")
    return X_full[:, :n_tgt], X_full[:, :n_dmon], pca, {
        "explained_var_ratio": pca.explained_variance_ratio_,
        "cumulative_var": cumvar,
        "n_components_target": n_tgt,
        "n_components_dmon": n_dmon,
        "var_at_target": float(cumvar[n_tgt-1]),
        "var_at_dmon":   float(cumvar[n_dmon-1]),
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — DIMENSIONALITY REDUCTION METHODS
# ─────────────────────────────────────────────────────────────────────────────
def compute_bray_curtis(data):
    return squareform(pdist(data, metric="braycurtis"))

def compute_jaccard(data):
    return squareform(pdist(data > 0, metric="jaccard"))

def compute_aitchison_dist(data):
    return squareform(pdist(nonzero_clr_transform(data), metric="euclidean"))

def run_all_dr_methods(X_clr, X_orig, cfg):
    _sec("STEP 6 — Dimensionality reduction (8 methods)")
    embeddings = {}

    # ── PCA ──────────────────────────────────────────────────────────────────
    _log("PCA", "running...")
    pca = PCA(n_components=2, random_state=42)
    embeddings["PCA"] = pca.fit_transform(X_clr)
    _log("PCA", f"done {embeddings['PCA'].shape}")

    # ── PCoA (Jaccard) ────────────────────────────────────────────────────────
    _log("PCoA", "computing Jaccard distance...")
    try:
        from skbio.stats.ordination import pcoa as skbio_pcoa
        from skbio import DistanceMatrix
        dm = DistanceMatrix(compute_jaccard(X_orig))
        result = skbio_pcoa(dm, number_of_dimensions=2)
        embeddings["Jaccard PCoA"] = result.samples.values
        _log("PCoA", f"done {embeddings['Jaccard PCoA'].shape}")
    except Exception as e:
        _log("PCoA", f"FAILED — {e}")
        embeddings["Jaccard PCoA"] = None

    # ── MDS ───────────────────────────────────────────────────────────────────
    _log("MDS", "running...")
    try:
        mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean", n_jobs=-1)
        embeddings["MDS"] = mds.fit_transform(X_clr)
        _log("MDS", f"done {embeddings['MDS'].shape}")
    except Exception as e:
        _log("MDS", f"FAILED — {e}"); embeddings["MDS"] = None

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    _log("t-SNE", "running...")
    try:
        perplexity = min(30, max(5, len(X_clr) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings["t-SNE"] = tsne.fit_transform(X_clr)
        _log("t-SNE", f"done {embeddings['t-SNE'].shape}")
    except Exception as e:
        _log("t-SNE", f"FAILED — {e}"); embeddings["t-SNE"] = None

    # ── UMAP ──────────────────────────────────────────────────────────────────
    _log("UMAP", "running...")
    try:
        import umap as umap_lib
        n_neighbors = min(15, max(2, len(X_clr) - 1))
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        embeddings["UMAP"] = reducer.fit_transform(X_clr)
        _log("UMAP", f"done {embeddings['UMAP'].shape}")
    except Exception as e:
        _log("UMAP", f"FAILED — {e}"); embeddings["UMAP"] = None

    # ── PaCMAP ────────────────────────────────────────────────────────────────
    _log("PaCMAP", "running...")
    try:
        import pacmap
        n_neighbors = min(15, max(2, len(X_clr) - 1))
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        embeddings["PaCMAP"] = reducer.fit_transform(X_clr)
        _log("PaCMAP", f"done {embeddings['PaCMAP'].shape}")
    except Exception as e:
        _log("PaCMAP", f"FAILED — {e}"); embeddings["PaCMAP"] = None

    # ── PHATE ─────────────────────────────────────────────────────────────────
    _log("PHATE", "running...")
    try:
        import phate
        phate_op = phate.PHATE(n_components=2, random_state=42, n_jobs=-1, verbose=0)
        embeddings["PHATE"] = phate_op.fit_transform(X_clr)
        _log("PHATE", f"done {embeddings['PHATE'].shape}")
    except Exception as e:
        _log("PHATE", f"FAILED — {e}"); embeddings["PHATE"] = None

    # ── SONG ──────────────────────────────────────────────────────────────────
    _log("SONG", "running...")
    try:
        from song.song import SONG
        nn = min(2, max(1, len(X_clr)//20))
        song = SONG(n_components=2, n_neighbors=nn, lr=0.5, random_seed=42, verbose=0)
        embeddings["SONG"] = song.fit_transform(X_clr.astype(np.float32))
        _log("SONG", f"done {embeddings['SONG'].shape}")
    except Exception as e:
        _log("SONG", f"FAILED — {e}"); embeddings["SONG"] = None

    valid = {k: v for k, v in embeddings.items() if v is not None}
    _log("summary", f"{len(valid)}/{len(embeddings)} methods succeeded")
    return embeddings

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_knn_graph(X_pca_dmon, k=10, mutual=True, metric="cosine"):
    """KNN cosine graph → edge_index (COO) + edge weights"""
    _sec("STEP 7a — KNN cosine graph")
    n  = X_pca_dmon.shape[0]
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nn.fit(X_pca_dmon)
    distances, indices = nn.kneighbors(X_pca_dmon)

    # cosine distance → similarity
    sims  = 1.0 - distances[:, 1:] / 2.0
    neigh = indices[:, 1:]

    src_l, tgt_l, w_l = [], [], []
    for i in range(n):
        for j_i, j in enumerate(neigh[i]):
            src_l.append(i); tgt_l.append(int(j)); w_l.append(float(sims[i, j_i]))

    src = np.array(src_l, dtype=np.int64)
    tgt = np.array(tgt_l, dtype=np.int64)
    wts = np.array(w_l,   dtype=np.float32)

    if mutual:
        edge_set = set(zip(src.tolist(), tgt.tolist()))
        keep = np.array([(i,j) in edge_set and (j,i) in edge_set
                         for i,j in zip(src.tolist(), tgt.tolist())], dtype=bool)
        src, tgt, wts = src[keep], tgt[keep], wts[keep]
        _log("mutual edges", f"{keep.sum()} / {len(keep)} kept")

    edge_index = np.stack([src, tgt], axis=0)
    _log("edge_index",  f"{edge_index.shape}  avg_deg={edge_index.shape[1]/n:.1f}")
    
    # Export as GEXF
    try:
        import networkx as nx
        G = nx.Graph() if mutual else nx.DiGraph()
        G.add_nodes_from(range(n))     
        edges_with_attr = [(int(s), int(t), {'weight': float(w)}) for s, t, w in zip(src, tgt, wts)]
        G.add_edges_from(edges_with_attr)
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        gexf_path = os.path.join(CONFIG["output_dir"], "knn_cosine_graph.gexf")
        nx.write_gexf(G, gexf_path)
        _log("GEXF saved", gexf_path)
    except Exception as e:
        _log("GEXF save FAILED", str(e))

    return edge_index, wts


def build_bipartite_graph(X_clr, taxa_names, threshold=0.0):
    """
    Bipartite graph: sample nodes ←→ taxon nodes
    Edge exists if CLR value > threshold (taxon is "active" in sample)
    
    Node layout:
      0 … n_samples-1          → sample nodes
      n_samples … n_samples+n_taxa-1 → taxon nodes

    Returns
    -------
    edge_index   : (2, E)  int64  COO
    edge_weights : (E,)    float32  absolute CLR values as weights
    n_sample_nodes, n_taxon_nodes
    """
    _sec("STEP 7b — Bipartite graph (sample ↔ taxon)")
    n_s, n_t = X_clr.shape

    rows, cols = np.where(X_clr > threshold)
    src = rows.astype(np.int64)              # sample node indices
    tgt = (cols + n_s).astype(np.int64)     # taxon node indices (offset)
    wts = np.abs(X_clr[rows, cols]).astype(np.float32)

    # Make undirected: add reverse edges
    edge_index = np.stack([
        np.concatenate([src, tgt]),
        np.concatenate([tgt, src])
    ], axis=0)
    edge_weights = np.concatenate([wts, wts])

    n_edges = edge_index.shape[1] // 2
    _log("sample nodes", f"{n_s}")
    _log("taxon nodes",  f"{n_t}")
    _log("total nodes",  f"{n_s + n_t}")
    _log("edges (dir)",  f"{n_edges}  →  {edge_index.shape[1]} (undirected)")
    _log("avg taxa/sample", f"{(X_clr > threshold).sum(axis=1).mean():.1f}")

    # Node feature matrix: sample nodes get CLR row; taxon nodes get CLR col mean
    sample_features = X_clr                                      # (n_s, n_t)
    taxon_features  = X_clr.T                                    # (n_t, n_s)
    # Pad to same width for stacking
    max_feat = max(n_s, n_t)
    sf_pad   = np.pad(sample_features, ((0,0),(0, max_feat-n_t)), mode="constant")
    tf_pad   = np.pad(taxon_features,  ((0,0),(0, max_feat-n_s)), mode="constant")
    node_features = np.vstack([sf_pad, tf_pad]).astype(np.float32)

    _log("node_features", f"{node_features.shape}")
    return edge_index, edge_weights, node_features, n_s, n_t

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — DR EMBEDDING EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _knn_sets(dist_matrix, k):
    k = min(k, len(dist_matrix)-1)
    return [set(np.argsort(dist_matrix[i])[1:k+1]) for i in range(len(dist_matrix))]

def trustworthiness_score(X_high, X_low, k=10):
    n = len(X_high); k = min(k, n-1)
    dh = pairwise_distances(X_high); dl = pairwise_distances(X_low)
    knn_low = _knn_sets(dl, k)
    t = 0
    for i in range(n):
        knn_hi_i = set(np.argsort(dh[i])[1:k+1])
        for j in knn_low[i]:
            if j not in knn_hi_i:
                rank = int(np.where(np.argsort(dh[i]) == j)[0][0])
                t += max(0, rank - k)
    return 1 - (2 / (n * k * (2*n - 3*k - 1))) * t

def continuity_score(X_high, X_low, k=10):
    n = len(X_high); k = min(k, n-1)
    dh = pairwise_distances(X_high); dl = pairwise_distances(X_low)
    knn_hi = _knn_sets(dh, k)
    c = 0
    for i in range(n):
        knn_lo_i = set(np.argsort(dl[i])[1:k+1])
        for j in knn_hi[i]:
            if j not in knn_lo_i:
                rank = int(np.where(np.argsort(dl[i]) == j)[0][0])
                c += max(0, rank - k)
    return 1 - (2 / (n * k * (2*n - 3*k - 1))) * c

def knn_preservation_score(X_high, X_low, k=10):
    n = len(X_high); k = min(k, n-1)
    dh = pairwise_distances(X_high); dl = pairwise_distances(X_low)
    knn_hi = _knn_sets(dh, k); knn_lo = _knn_sets(dl, k)
    return sum(len(knn_hi[i] & knn_lo[i]) for i in range(n)) / (n * k)

def normalized_stress(X_high, X_low):
    dh = pairwise_distances(X_high).flatten()
    dl = pairwise_distances(X_low).flatten()
    s  = np.sum((dh - dl)**2)
    nrm = np.sum(dh**2)
    return float(np.sqrt(s / nrm)) if nrm > 0 else 0.0

def distance_correlation(X_orig, X_emb, metric="braycurtis"):
    if metric == "braycurtis":   dist_orig = compute_bray_curtis(X_orig)
    elif metric == "aitchison":  dist_orig = compute_aitchison_dist(X_orig)
    elif metric == "jaccard":    dist_orig = compute_jaccard(X_orig)
    else:                        dist_orig = pairwise_distances(X_orig)
    dist_emb = pairwise_distances(X_emb)
    corr, _  = spearmanr(dist_orig.flatten(), dist_emb.flatten())
    return float(corr)

def evaluate_embedding(X_orig, X_emb, name, k=10):
    if X_emb is None: return None
    _log(f"  eval {name}", "computing metrics...")
    return {
        "Method"            : name,
        "Trustworthiness"   : trustworthiness_score(X_orig, X_emb, k),
        "Continuity"        : continuity_score(X_orig, X_emb, k),
        "KNN Preservation"  : knn_preservation_score(X_orig, X_emb, k),
        "Normalized Stress" : normalized_stress(X_orig, X_emb),
        "Bray-Curtis Corr"  : distance_correlation(X_orig, X_emb, "braycurtis"),
        "Aitchison Corr"    : distance_correlation(X_orig, X_emb, "aitchison"),
        "Jaccard Corr"      : distance_correlation(X_orig, X_emb, "jaccard"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _save_csv(arr, columns, index, path):
    pd.DataFrame(arr, columns=columns, index=index).to_csv(path)
    _log("saved", str(Path(path).name))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing_pipeline(cfg=CONFIG):
    print("\n" + "█"*64)
    print("  MICROBIOME UNIFIED PREPROCESSING PIPELINE")
    print("  nzCLR + 8 DR Methods + KNN graph + Bipartite graph")
    print("█"*64)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 1 Load
    metadata, abundance_raw, y_true, le, disease_series = load_and_validate(cfg)
    sample_ids = metadata.iloc[:,0].astype(str).tolist()

    # 2 Prevalence filter
    abundance_filt, prev_stats = prevalence_filter(
        abundance_raw, cfg["prevalence_threshold"], cfg["min_abundance_thresh"])
    taxa_names = list(abundance_filt.columns)
    X_orig     = abundance_filt.values.astype(np.float64)

    # 3 TSS
    X_tss = tss_normalise(abundance_filt, cfg["pseudocount"])

    # 4 nzCLR
    X_clr = run_nzclr(X_tss)

    # 5 Aitchison PCA
    X_pca, X_pca_dmon, pca_model, pca_stats = aitchison_pca(
        X_clr, cfg["pca_variance_target"], cfg["pca_max_components"], cfg["pca_dmon_components"])

    # 6 All DR methods
    embeddings = run_all_dr_methods(X_clr, X_orig, cfg)

    # 7 Graphs
    edge_index_knn, edge_weights_knn = build_knn_graph(
        X_pca_dmon, k=cfg["knn_k"], mutual=cfg["knn_mutual"])

    edge_index_bip, edge_weights_bip, node_feat_bip, n_s, n_t = build_bipartite_graph(
        X_clr, taxa_names, cfg["bipartite_threshold"])

    # 8 DR evaluation metrics
    _sec("STEP 8 — DR embedding evaluation")
    dr_results = []
    for name, emb in embeddings.items():
        m = evaluate_embedding(X_clr, emb, name, k=10)
        if m: dr_results.append(m)
    dr_df = pd.DataFrame(dr_results)

    # ── Save all outputs ──────────────────────────────────────────────────────
    _sec("Saving outputs")
    _save_csv(X_clr,    taxa_names,  sample_ids, f"{out_dir}/X_clr.csv")
    _save_csv(X_pca,    [f"PC{i+1}" for i in range(X_pca.shape[1])],
              sample_ids, f"{out_dir}/X_pca_kmeans.csv")
    _save_csv(X_pca_dmon, [f"PC{i+1}" for i in range(X_pca_dmon.shape[1])],
              sample_ids, f"{out_dir}/X_pca_dmon.csv")

    for name, emb in embeddings.items():
        if emb is not None:
            safe = name.replace(" ", "_").replace("/","_")
            _save_csv(emb, ["Dim1","Dim2"], sample_ids,
                      f"{out_dir}/embedding_{safe}.csv")

    np.save(f"{out_dir}/edge_index_knn.npy",    edge_index_knn)
    np.save(f"{out_dir}/edge_weights_knn.npy",  edge_weights_knn)
    np.save(f"{out_dir}/edge_index_bip.npy",    edge_index_bip)
    np.save(f"{out_dir}/edge_weights_bip.npy",  edge_weights_bip)
    np.save(f"{out_dir}/node_feat_bip.npy",     node_feat_bip)
    np.save(f"{out_dir}/y_true.npy",            y_true)
    np.save(f"{out_dir}/bip_n_sample_nodes.npy", np.array([n_s]))
    np.save(f"{out_dir}/bip_n_taxon_nodes.npy",  np.array([n_t]))

    pd.DataFrame({"taxon": taxa_names}).to_csv(f"{out_dir}/taxa_names.csv", index=False)
    dr_df.to_csv(f"{out_dir}/dr_evaluation_metrics.csv", index=False)
    disease_series.reset_index(drop=True).to_csv(
        f"{out_dir}/disease_labels.csv", index=False, header=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    _sec("PIPELINE COMPLETE")
    valid_dr = [k for k,v in embeddings.items() if v is not None]
    print(f"""
  Samples         : {len(sample_ids)}
  Taxa (raw)      : {prev_stats['n_before']}
  Taxa (kept)     : {prev_stats['n_after']}
  DR methods done : {valid_dr}

  Outputs → {out_dir}/
    X_clr.csv              nzCLR features       (K-Means input)
    X_pca_kmeans.csv       Aitchison PCA        (K-Means Aitchison input)
    X_pca_dmon.csv         Compact PCA          (GNN node features — KNN graph)
    node_feat_bip.npy      Bipartite node feats (GNN node features — bipartite)
    embedding_*.csv        2D embeddings per DR method
    edge_index_knn.npy     KNN cosine graph COO
    edge_index_bip.npy     Bipartite graph COO
    y_true.npy             Disease label integers
    dr_evaluation_metrics.csv  Trustworthiness · Continuity · KNN · Stress · Corr
""")
    return {
        "X_orig": X_orig, "X_tss": X_tss, "X_clr": X_clr,
        "X_pca": X_pca, "X_pca_dmon": X_pca_dmon,
        "embeddings": embeddings, "dr_df": dr_df,
        "edge_index_knn": edge_index_knn, "edge_weights_knn": edge_weights_knn,
        "edge_index_bip": edge_index_bip, "edge_weights_bip": edge_weights_bip,
        "node_feat_bip": node_feat_bip,
        "y_true": y_true, "le": le, "taxa_names": taxa_names,
        "disease_series": disease_series,
        "pca_stats": pca_stats, "prev_stats": prev_stats,
    }

if __name__ == "__main__":
    outputs = run_preprocessing_pipeline(CONFIG)
