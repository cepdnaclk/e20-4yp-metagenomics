"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MICROBIOME — UNIFIED PREPROCESSING + GRAPH CONSTRUCTION PIPELINE         ║
║   nzCLR → 8 DR Methods + 2 Graph Types                                     ║
║                                                                              ║
║   Data loading : noise marker replacement → Disease col detection →         ║
║                  sample-ID removal → numeric coerce → fillna(0)             ║
║                  (aligned with previous research pipeline)                  ║
║                                                                              ║
║   DR Methods  : PCA · PCoA(Jaccard) · MDS · t-SNE · UMAP · PaCMAP ·       ║
║                 PHATE · SONG                                                 ║
║   Graph Types : KNN cosine graph  |  Bipartite graph                        ║
║   Outputs for : K-Means × 8 DR methods                                      ║
║               + DMoN    × KNN graph + Bipartite graph                       ║
║               + MinCutPool × KNN graph + Bipartite graph                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Changes from previous version
──────────────────────────────
  REMOVED  Step 2 — Prevalence filter   (not in previous research pipeline)
  REMOVED  Step 3 — TSS normalisation   (not in previous research pipeline)
  REMOVED  Step 5 — Aitchison PCA       (not in previous research pipeline)
  UPDATED  Step 1 — Data loading now exactly mirrors previous research:
             • df.replace() noise markers before any column detection
             • Disease col found case-insensitively anywhere in CSV
             • Sample-ID col detected and dropped from features
             • All remaining cols coerced to numeric, NaN → 0, clip ≥ 0
  UPDATED  Step 4 — nzCLR applied directly on raw X_orig (no TSS first)
  UPDATED  DR methods — PCoA uses X_orig (raw); all others use X_clr,
             matching the previous research run_full_analysis() exactly.

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
    # ── Column detection ──────────────────────────────────────────────────────
    # Disease column is detected case-insensitively anywhere in the CSV.
    # Sample-ID column is detected by keyword match and excluded from features.
    # All remaining numeric columns are treated as abundance features.
    # Set metadata_cols to an integer only if you need a fixed-split fallback.
    "metadata_cols"        : None,
    "disease_col_keyword"  : "disease",
    "sample_id_keywords"   : ["sample_id", "sampleid", "sample", "id"],
    "noise_labels"         : {"nd", "na", "-", " -", "unknown", "nan", "none", ""},
    # KNN graph
    "knn_k"                : 10,
    "knn_mutual"           : True,
    # Bipartite graph
    "bipartite_threshold"  : 0.0,   # CLR > threshold → edge exists
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
# Mirrors previous research run_full_analysis() data loading exactly:
#   1. Replace noise markers with NaN across entire DataFrame
#   2. Find Disease column anywhere (case-insensitive), extract & drop it
#   3. Detect sample-ID column by keyword, extract & drop it
#   4. Coerce remaining columns to numeric, fillna(0), clip(lower=0)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_disease(series, noise_labels):
    """Strip, lowercase, map noisy labels to 'unknown'."""
    s = series.astype(str).str.strip().str.lower()
    return s.where(~s.isin(noise_labels), other="unknown")

def load_and_validate(cfg):
    _sec("STEP 1 — Load & validate")
    df = pd.read_csv(cfg["abundance_csv"], low_memory=False)
    _log("raw shape", str(df.shape))

    noise_labels = cfg.get("noise_labels", {"nd", "na", "-", " -", "unknown", "nan", "none", ""})

    # ── Step 1a: Replace noise markers (previous research cleaning step) ──────
    # Matches: df.replace("nd", np.nan) ... df.replace('unknown', np.nan)
    for marker in ["nd", "na", "-", " -", "unknown"]:
        df = df.replace(marker, np.nan)

    # ── Fixed-split fallback (only when metadata_cols is explicitly set) ──────
    n_meta = cfg.get("metadata_cols", None)
    if n_meta is not None:
        _log("mode", f"fixed metadata split at column {n_meta}")
        metadata  = df.iloc[:, :n_meta].copy()
        abundance = df.iloc[:, n_meta:].copy()
        abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)

        dis_col = next((c for c in metadata.columns
                        if cfg["disease_col_keyword"].lower() in c.lower()), None)
        if dis_col is None:
            raise ValueError("No disease column found in the metadata block.")

        disease_series = _clean_disease(metadata[dis_col], noise_labels)
        sample_ids     = metadata.iloc[:, 0].astype(str).tolist()

    else:
        # ── Dynamic detection — mirrors previous research exactly ─────────────
        _log("mode", "dynamic column detection (previous research pipeline)")

        # Step 1b: Find Disease column anywhere in CSV (case-insensitive)
        dis_col = next(
            (c for c in df.columns if cfg["disease_col_keyword"].lower() in c.lower()), None
        )
        if dis_col is None:
            _log("WARNING", "'Disease' column not found — labelling all samples 'Unknown'")
            disease_series = pd.Series(["Unknown"] * len(df))
        else:
            _log("disease col found", f"'{dis_col}'")
            # fillna('Unknown') matches previous research: disease_column.fillna('Unknown')
            disease_series = df[dis_col].fillna("Unknown").astype(str).str.strip()
            disease_series = _clean_disease(disease_series, noise_labels)
            df = df.drop(columns=[dis_col])

        # Step 1c: Detect and drop sample-ID column
        # Matches: if df.columns[0].lower() in ['sample_id', 'sampleid', 'sample', 'id']
        sample_id_keywords = cfg.get("sample_id_keywords",
                                     ["sample_id", "sampleid", "sample", "id"])
        sample_id_col = next(
            (c for c in df.columns if c.lower() in sample_id_keywords), None
        )
        if sample_id_col:
            sample_ids = df[sample_id_col].astype(str).tolist()
            df = df.drop(columns=[sample_id_col])
            _log("sample-ID col", f"'{sample_id_col}' — removed from features")
        else:
            sample_ids = [str(i) for i in df.index]
            _log("sample-ID col", "not found — using row index")

        # Step 1d: Coerce to numeric, fillna(0), clip negatives
        # Matches: feature_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        abundance = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        abundance = abundance.clip(lower=0)

        metadata = pd.DataFrame({"sample_id": sample_ids})

    # ── Encode disease labels ─────────────────────────────────────────────────
    le     = LabelEncoder()
    y_true = le.fit_transform(disease_series)

    _log("disease col", f"'{dis_col}'  —  {len(le.classes_)} classes: {list(le.classes_)}")
    for cls, cnt in zip(le.classes_, np.bincount(y_true)):
        _log("  class", f"{cls:<32} n={cnt}")

    _log("samples",            str(len(sample_ids)))
    _log("abundance features", str(abundance.shape[1]))

    return metadata, abundance, y_true, le, disease_series


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — nzCLR TRANSFORMATION
# Applied directly on raw X_orig (no prevalence filter, no TSS first),
# matching the previous research pipeline:
#   X_clr = nonzero_clr_transform(X_orig)
# ─────────────────────────────────────────────────────────────────────────────

def nonzero_clr_transform(data):
    """
    Non-zero Centered Log-Ratio (CLR) transformation.
    Zero values stay 0; non-zero values → log(x / gmean_of_nonzeros).
    Identical to previous research nonzero_clr_transform().
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    data     = np.array(data, dtype=np.float64)
    clr_data = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        row = data[i, :]
        nz  = row[row > 0]
        if len(nz) > 0:
            g = gmean(nz)
            clr_data[i, :] = np.where(row > 0, np.log(row / g), 0.0)
    return clr_data

def run_nzclr(X_orig):
    _sec("STEP 2 — nzCLR transformation  (applied directly on raw features)")
    X_clr = nonzero_clr_transform(X_orig)
    _log("shape",   f"{X_clr.shape}  dtype={X_clr.dtype}")
    _log("range",   f"[{X_clr.min():.4f}, {X_clr.max():.4f}]")
    _log("zero %",  f"{(X_clr==0).mean():.2%}")
    return X_clr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DIMENSIONALITY REDUCTION METHODS
# Mirrors previous research methods() dict exactly:
#   PCoA  → run_pcoa(X_orig)   — Jaccard on raw counts
#   others → run_*(X_clr)      — nzCLR transformed data
# ─────────────────────────────────────────────────────────────────────────────

def compute_bray_curtis(data):
    return squareform(pdist(data, metric="braycurtis"))

def compute_jaccard(data):
    return squareform(pdist(data > 0, metric="jaccard"))

def compute_aitchison_dist(data):
    return squareform(pdist(nonzero_clr_transform(data), metric="euclidean"))

def run_all_dr_methods(X_clr, X_orig, cfg):
    _sec("STEP 3 — Dimensionality reduction (8 methods)")
    embeddings = {}

    # ── PCA — on X_clr ────────────────────────────────────────────────────────
    _log("PCA", "running on X_clr...")
    pca = PCA(n_components=2, random_state=42)
    embeddings["PCA"] = pca.fit_transform(X_clr)
    _log("PCA", f"done {embeddings['PCA'].shape}")

    # ── PCoA (Jaccard) — on X_orig (raw counts) ───────────────────────────────
    # Previous research: run_pcoa(X_orig) — Jaccard distance on raw data
    _log("PCoA", "computing Jaccard distance on X_orig (raw)...")
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

    # ── MDS — on X_clr ────────────────────────────────────────────────────────
    _log("MDS", "running on X_clr...")
    try:
        mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean", n_jobs=-1)
        embeddings["MDS"] = mds.fit_transform(X_clr)
        _log("MDS", f"done {embeddings['MDS'].shape}")
    except Exception as e:
        _log("MDS", f"FAILED — {e}"); embeddings["MDS"] = None

    # ── t-SNE — on X_clr ──────────────────────────────────────────────────────
    _log("t-SNE", "running on X_clr...")
    try:
        perplexity = min(30, max(5, len(X_clr) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings["t-SNE"] = tsne.fit_transform(X_clr)
        _log("t-SNE", f"done {embeddings['t-SNE'].shape}")
    except Exception as e:
        _log("t-SNE", f"FAILED — {e}"); embeddings["t-SNE"] = None

    # ── UMAP — on X_clr ───────────────────────────────────────────────────────
    _log("UMAP", "running on X_clr...")
    try:
        import umap as umap_lib
        n_neighbors = min(15, max(2, len(X_clr) - 1))
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        embeddings["UMAP"] = reducer.fit_transform(X_clr)
        _log("UMAP", f"done {embeddings['UMAP'].shape}")
    except Exception as e:
        _log("UMAP", f"FAILED — {e}"); embeddings["UMAP"] = None

    # ── PaCMAP — on X_clr ─────────────────────────────────────────────────────
    _log("PaCMAP", "running on X_clr...")
    try:
        import pacmap
        n_neighbors = min(15, max(2, len(X_clr) - 1))
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        embeddings["PaCMAP"] = reducer.fit_transform(X_clr)
        _log("PaCMAP", f"done {embeddings['PaCMAP'].shape}")
    except Exception as e:
        _log("PaCMAP", f"FAILED — {e}"); embeddings["PaCMAP"] = None

    # ── PHATE — on X_clr ──────────────────────────────────────────────────────
    _log("PHATE", "running on X_clr...")
    try:
        import phate
        phate_op = phate.PHATE(n_components=2, random_state=42, n_jobs=-1, verbose=0)
        embeddings["PHATE"] = phate_op.fit_transform(X_clr)
        _log("PHATE", f"done {embeddings['PHATE'].shape}")
    except Exception as e:
        _log("PHATE", f"FAILED — {e}"); embeddings["PHATE"] = None

    # ── SONG — on X_clr ───────────────────────────────────────────────────────
    _log("SONG", "running on X_clr...")
    try:
        from song.song import SONG
        nn = min(2, max(1, len(X_clr) // 20))
        song = SONG(n_components=2, n_neighbors=nn, lr=0.5, random_seed=42, verbose=0)
        embeddings["SONG"] = song.fit_transform(X_clr.astype(np.float32))
        _log("SONG", f"done {embeddings['SONG'].shape}")
    except Exception as e:
        _log("SONG", f"FAILED — {e}"); embeddings["SONG"] = None

    valid = {k: v for k, v in embeddings.items() if v is not None}
    _log("summary", f"{len(valid)}/{len(embeddings)} methods succeeded")
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_graph(X_clr, k=10, mutual=True, metric="cosine"):
    """KNN cosine graph built on X_clr → edge_index (COO) + edge weights."""
    _sec("STEP 4a — KNN cosine graph  (built on X_clr)")
    n  = X_clr.shape[0]
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nn.fit(X_clr)
    distances, indices = nn.kneighbors(X_clr)

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

    try:
        import networkx as nx
        G = nx.Graph() if mutual else nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from([(int(s), int(t), {"weight": float(w)})
                          for s, t, w in zip(src, tgt, wts)])
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        gexf_path = os.path.join(CONFIG["output_dir"], "knn_cosine_graph.gexf")
        nx.write_gexf(G, gexf_path)
        _log("GEXF saved", gexf_path)
    except Exception as e:
        _log("GEXF save FAILED", str(e))

    return edge_index, wts


def build_bipartite_graph(X_clr, taxa_names, threshold=0.0):
    """
    Bipartite graph: sample nodes ←→ taxon nodes.
    Edge exists where CLR value > threshold.

    Node layout:
      0 … n_samples-1               → sample nodes
      n_samples … n_samples+n_taxa-1 → taxon nodes
    """
    _sec("STEP 4b — Bipartite graph  (sample ↔ taxon)")
    n_s, n_t = X_clr.shape

    rows, cols = np.where(X_clr > threshold)
    src = rows.astype(np.int64)
    tgt = (cols + n_s).astype(np.int64)
    wts = np.abs(X_clr[rows, cols]).astype(np.float32)

    edge_index = np.stack([
        np.concatenate([src, tgt]),
        np.concatenate([tgt, src])
    ], axis=0)
    edge_weights = np.concatenate([wts, wts])

    n_edges = edge_index.shape[1] // 2
    _log("sample nodes",    f"{n_s}")
    _log("taxon nodes",     f"{n_t}")
    _log("total nodes",     f"{n_s + n_t}")
    _log("edges (dir)",     f"{n_edges}  →  {edge_index.shape[1]} (undirected)")
    _log("avg taxa/sample", f"{(X_clr > threshold).sum(axis=1).mean():.1f}")

    sample_features = X_clr
    taxon_features  = X_clr.T
    max_feat = max(n_s, n_t)
    sf_pad   = np.pad(sample_features, ((0,0),(0, max_feat-n_t)), mode="constant")
    tf_pad   = np.pad(taxon_features,  ((0,0),(0, max_feat-n_s)), mode="constant")
    node_features = np.vstack([sf_pad, tf_pad]).astype(np.float32)

    _log("node_features", f"{node_features.shape}")
    return edge_index, edge_weights, node_features, n_s, n_t


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DR EMBEDDING EVALUATION METRICS
# Identical to previous research evaluate_embedding() logic.
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
    print("  nzCLR (on raw) + 8 DR Methods + KNN graph + Bipartite graph")
    print("█"*64)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load & validate ───────────────────────────────────────────────
    metadata, abundance_raw, y_true, le, disease_series = load_and_validate(cfg)
    sample_ids = metadata.iloc[:, 0].astype(str).tolist()
    taxa_names = list(abundance_raw.columns)
    X_orig     = abundance_raw.values.astype(np.float64)

    # ── Step 2: nzCLR directly on raw features (no TSS, no prevalence filter) ─
    X_clr = run_nzclr(X_orig)

    # ── Step 3: All 8 DR methods ──────────────────────────────────────────────
    # PCoA uses X_orig; all others use X_clr — matches previous research exactly
    embeddings = run_all_dr_methods(X_clr, X_orig, cfg)

    # ── Step 4: Graph construction (KNN on X_clr, bipartite on X_clr) ─────────
    edge_index_knn, edge_weights_knn = build_knn_graph(
        X_clr, k=cfg["knn_k"], mutual=cfg["knn_mutual"])

    edge_index_bip, edge_weights_bip, node_feat_bip, n_s, n_t = build_bipartite_graph(
        X_clr, taxa_names, cfg["bipartite_threshold"])

    # ── Step 5: DR embedding evaluation metrics ───────────────────────────────
    _sec("STEP 5 — DR embedding evaluation")
    dr_results = []
    for name, emb in embeddings.items():
        m = evaluate_embedding(X_clr, emb, name, k=10)
        if m: dr_results.append(m)
    dr_df = pd.DataFrame(dr_results)

    # ── Save all outputs ──────────────────────────────────────────────────────
    _sec("Saving outputs")
    _save_csv(X_clr, taxa_names, sample_ids, f"{out_dir}/X_clr.csv")

    for name, emb in embeddings.items():
        if emb is not None:
            safe = name.replace(" ", "_").replace("/", "_")
            _save_csv(emb, ["Dim1", "Dim2"], sample_ids,
                      f"{out_dir}/embedding_{safe}.csv")

    np.save(f"{out_dir}/edge_index_knn.npy",     edge_index_knn)
    np.save(f"{out_dir}/edge_weights_knn.npy",   edge_weights_knn)
    np.save(f"{out_dir}/edge_index_bip.npy",     edge_index_bip)
    np.save(f"{out_dir}/edge_weights_bip.npy",   edge_weights_bip)
    np.save(f"{out_dir}/node_feat_bip.npy",      node_feat_bip)
    np.save(f"{out_dir}/y_true.npy",             y_true)
    np.save(f"{out_dir}/bip_n_sample_nodes.npy", np.array([n_s]))
    np.save(f"{out_dir}/bip_n_taxon_nodes.npy",  np.array([n_t]))

    # X_clr is also used as the KNN graph node features (replaces X_pca_dmon)
    _save_csv(X_clr, taxa_names, sample_ids, f"{out_dir}/X_pca_dmon.csv")

    pd.DataFrame({"taxon": taxa_names}).to_csv(f"{out_dir}/taxa_names.csv", index=False)
    dr_df.to_csv(f"{out_dir}/dr_evaluation_metrics.csv", index=False)
    disease_series.reset_index(drop=True).to_csv(
        f"{out_dir}/disease_labels.csv", index=False, header=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    _sec("PIPELINE COMPLETE")
    valid_dr = [k for k, v in embeddings.items() if v is not None]
    print(f"""
  Samples         : {len(sample_ids)}
  Taxa (features) : {len(taxa_names)}
  Classes         : {len(le.classes_)}  {list(le.classes_)}
  DR methods done : {valid_dr}

  Outputs → {out_dir}/
    X_clr.csv              nzCLR features       (K-Means + GNN input)
    X_pca_dmon.csv         alias of X_clr.csv   (GNN node features — KNN graph)
    node_feat_bip.npy      Bipartite node feats (GNN node features — bipartite)
    embedding_*.csv        2-D embeddings per DR method
    edge_index_knn.npy     KNN cosine graph COO
    edge_index_bip.npy     Bipartite graph COO
    y_true.npy             Disease label integers
    dr_evaluation_metrics.csv  Trustworthiness · Continuity · KNN · Stress · Corr
""")
    return {
        "X_orig"         : X_orig,
        "X_clr"          : X_clr,
        "embeddings"     : embeddings,
        "dr_df"          : dr_df,
        "edge_index_knn" : edge_index_knn,
        "edge_weights_knn": edge_weights_knn,
        "edge_index_bip" : edge_index_bip,
        "edge_weights_bip": edge_weights_bip,
        "node_feat_bip"  : node_feat_bip,
        "y_true"         : y_true,
        "le"             : le,
        "taxa_names"     : taxa_names,
        "disease_series" : disease_series,
    }


if __name__ == "__main__":
    outputs = run_preprocessing_pipeline(CONFIG)