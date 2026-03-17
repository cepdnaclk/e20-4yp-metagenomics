"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OCEAN MICROBIOME — UNIFIED PREPROCESSING + GRAPH CONSTRUCTION PIPELINE   ║
║   nzCLR → 8 DR Methods + 2 Graph Types                                     ║
║                                                                              ║
║   Dataset   : Tara Oceans (abundance.csv)                                   ║
║   Features  : motu_linkage_group_* columns (1440 features)                  ║
║   Label     : Environmental Feature                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    # Ocean dataset specifics
    "feature_col_prefix"   : "motu_linkage_group_",
    "label_col"            : "Environmental Feature",
    "noise_labels"         : {"nd", "na", "-", " -", "unknown", "nan", "none", ""},
    # KNN graph
    "knn_k"                : 10,
    "knn_mutual"           : True,
    # Bipartite graph
    "bipartite_threshold"  : 0.0,
}

DARK  = "#ffffff"; PANEL = "#f5f5f5"; GRID = "#d0d0d0"
TEXT  = "#111111"; TEXT2 = "#555555"
COLORS = ["#1f77b4","#d62728","#2ca02c","#9467bd","#8c564b",
          "#e377c2","#7f7f7f","#bcbd22","#17becf","#ff7f0e",
          "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
          "#c49c94","#f7b6d2","#dbdb8d"]

def _sec(t): print(f"\n{'═'*64}\n  {t}\n{'═'*64}")
def _log(t, m): print(f"  [{t}]{'─'*max(52-len(t),1)} {m}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def _clean_label(series, noise_labels):
    s = series.astype(str).str.strip().str.lower()
    return s.where(~s.isin(noise_labels), other="unknown")

def load_and_validate(cfg):
    _sec("STEP 1 — Load & validate")
    df = pd.read_csv(cfg["abundance_csv"], low_memory=False)
    _log("raw shape", str(df.shape))

    noise_labels = cfg.get("noise_labels", set())

    # Replace noise markers
    for marker in ["nd", "na", "-", " -", "unknown"]:
        df = df.replace(marker, np.nan)

    # ── Extract label column ──────────────────────────────────────────────────
    label_col = cfg["label_col"]
    if label_col in df.columns:
        label_series = df[label_col].fillna("unknown").astype(str).str.strip()
        label_series = _clean_label(label_series, noise_labels)
        df = df.drop(columns=[label_col])
        _log("label col", f"'{label_col}'")
    else:
        _log("WARNING", f"'{label_col}' not found. Labelling all 'unknown'.")
        label_series = pd.Series(["unknown"] * len(df))

    # ── Extract feature columns by prefix ────────────────────────────────────
    prefix = cfg["feature_col_prefix"]
    feature_cols = [c for c in df.columns if c.startswith(prefix)]
    _log("feature cols", f"{len(feature_cols)} ({prefix}*)")

    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found with prefix '{prefix}'")

    abundance = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)

    # ── Encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_true = le.fit_transform(label_series)

    _log("label col", f"  {len(le.classes_)} classes: {list(le.classes_)}")
    for cls, cnt in zip(le.classes_, np.bincount(y_true)):
        _log("  class", f"{cls:<32} n={cnt}")
    _log("samples", str(len(df)))
    _log("abundance features", str(abundance.shape[1]))

    # Use first column as sample ID if it exists
    sample_ids = df.iloc[:, 0].astype(str).tolist() if df.shape[1] > 0 else [str(i) for i in df.index]

    return sample_ids, abundance, y_true, le, label_series


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — nzCLR TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def nonzero_clr_transform(data):
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
    _log("shape",  f"{X_clr.shape}  dtype={X_clr.dtype}")
    _log("range",  f"[{X_clr.min():.4f}, {X_clr.max():.4f}]")
    _log("zero %", f"{(X_clr==0).mean():.2%}")
    return X_clr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DIMENSIONALITY REDUCTION METHODS
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

    _log("PCA", "running on X_clr...")
    pca = PCA(n_components=2, random_state=42)
    embeddings["PCA"] = pca.fit_transform(X_clr)
    _log("PCA", f"done {embeddings['PCA'].shape}")

    _log("PCoA", "computing Jaccard distance on X_orig (raw)...")
    try:
        from skbio.stats.ordination import pcoa as skbio_pcoa
        from skbio import DistanceMatrix
        dm = DistanceMatrix(compute_jaccard(X_orig))
        result = skbio_pcoa(dm, number_of_dimensions=2)
        embeddings["Jaccard PCoA"] = result.samples.values
        _log("PCoA", f"done {embeddings['Jaccard PCoA'].shape}")
    except Exception as e:
        _log("PCoA", f"FAILED — {e}"); embeddings["Jaccard PCoA"] = None

    _log("MDS", "running on X_clr...")
    try:
        mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean", n_jobs=-1)
        embeddings["MDS"] = mds.fit_transform(X_clr)
        _log("MDS", f"done {embeddings['MDS'].shape}")
    except Exception as e:
        _log("MDS", f"FAILED — {e}"); embeddings["MDS"] = None

    _log("t-SNE", "running on X_clr...")
    try:
        perplexity = min(30, max(5, len(X_clr) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings["t-SNE"] = tsne.fit_transform(X_clr)
        _log("t-SNE", f"done {embeddings['t-SNE'].shape}")
    except Exception as e:
        _log("t-SNE", f"FAILED — {e}"); embeddings["t-SNE"] = None

    _log("UMAP", "running on X_clr...")
    try:
        import umap as umap_lib
        n_neighbors = min(15, max(2, len(X_clr) - 1))
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        embeddings["UMAP"] = reducer.fit_transform(X_clr)
        _log("UMAP", f"done {embeddings['UMAP'].shape}")
    except Exception as e:
        _log("UMAP", f"FAILED — {e}"); embeddings["UMAP"] = None

    _log("PaCMAP", "running on X_clr...")
    try:
        import pacmap
        n_neighbors = min(15, max(2, len(X_clr) - 1))
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        embeddings["PaCMAP"] = reducer.fit_transform(X_clr)
        _log("PaCMAP", f"done {embeddings['PaCMAP'].shape}")
    except Exception as e:
        _log("PaCMAP", f"FAILED — {e}"); embeddings["PaCMAP"] = None

    _log("PHATE", "running on X_clr...")
    try:
        import phate
        phate_op = phate.PHATE(n_components=2, random_state=42, n_jobs=-1, verbose=0)
        embeddings["PHATE"] = phate_op.fit_transform(X_clr)
        _log("PHATE", f"done {embeddings['PHATE'].shape}")
    except Exception as e:
        _log("PHATE", f"FAILED — {e}"); embeddings["PHATE"] = None

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
    print("  OCEAN MICROBIOME — UNIFIED PREPROCESSING PIPELINE")
    print("  nzCLR (on raw) + 8 DR Methods + KNN graph + Bipartite graph")
    print("█"*64)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Load & validate
    sample_ids, abundance_raw, y_true, le, label_series = load_and_validate(cfg)
    taxa_names = list(abundance_raw.columns)
    X_orig     = abundance_raw.values.astype(np.float64)

    # Step 2: nzCLR
    X_clr = run_nzclr(X_orig)

    # Step 3: DR methods
    embeddings = run_all_dr_methods(X_clr, X_orig, cfg)

    # Step 4: Graph construction
    edge_index_knn, edge_weights_knn = build_knn_graph(
        X_clr, k=cfg["knn_k"], mutual=cfg["knn_mutual"])

    edge_index_bip, edge_weights_bip, node_feat_bip, n_s, n_t = build_bipartite_graph(
        X_clr, taxa_names, cfg["bipartite_threshold"])

    # Step 5: DR evaluation
    _sec("STEP 5 — DR embedding evaluation")
    dr_results = []
    for name, emb in embeddings.items():
        m = evaluate_embedding(X_clr, emb, name, k=10)
        if m: dr_results.append(m)
    dr_df = pd.DataFrame(dr_results)

    # Save outputs
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

    # X_clr is also used as GNN node features
    _save_csv(X_clr, taxa_names, sample_ids, f"{out_dir}/X_pca_dmon.csv")

    pd.DataFrame({"taxon": taxa_names}).to_csv(f"{out_dir}/taxa_names.csv", index=False)
    dr_df.to_csv(f"{out_dir}/dr_evaluation_metrics.csv", index=False)
    label_series.reset_index(drop=True).to_csv(
        f"{out_dir}/disease_labels.csv", index=False, header=True)

    _sec("PIPELINE COMPLETE")
    valid_dr = [k for k, v in embeddings.items() if v is not None]
    print(f"""
  Samples         : {len(sample_ids)}
  Taxa (features) : {len(taxa_names)}
  Classes         : {len(le.classes_)}  {list(le.classes_)}
  DR methods done : {valid_dr}

  Outputs → {out_dir}/
    X_clr.csv              nzCLR features       (K-Means + GNN input)
    X_pca_dmon.csv         alias of X_clr.csv   (GNN node features)
    embedding_*.csv        2-D embeddings per DR method
    edge_index_knn.npy     KNN cosine graph COO
    edge_index_bip.npy     Bipartite graph COO
    y_true.npy             Label integers
    dr_evaluation_metrics.csv  DR quality metrics
""")
    return {
        "X_orig": X_orig, "X_clr": X_clr, "embeddings": embeddings,
        "dr_df": dr_df, "y_true": y_true, "le": le,
        "taxa_names": taxa_names, "label_series": label_series,
    }


if __name__ == "__main__":
    outputs = run_preprocessing_pipeline(CONFIG)
