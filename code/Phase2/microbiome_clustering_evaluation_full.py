"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MICROBIOME — COMPLETE EVALUATION SUITE                                    ║
║   4 GNN combinations + 8 DR × K-Means combinations = 12 methods total      ║
║                                                                              ║
║   GNN methods                                                                ║
║     1. KNN cosine graph   + DMoN                                            ║
║     2. KNN cosine graph   + MinCutPool                                      ║
║     3. Bipartite graph    + DMoN                                            ║
║     4. Bipartite graph    + MinCutPool                                      ║
║                                                                              ║
║   Classical methods (K-Means on each DR embedding)                          ║
║     5–12. PCA / PCoA / MDS / t-SNE / UMAP / PaCMAP / PHATE / SONG         ║
║                                                                              ║
║   Metrics: NMI · ARI · Silhouette (cosine) · Homogeneity · Completeness    ║
║            V-measure · Calinski-Harabasz · Davies-Bouldin                   ║
║   Plots  : embedding grid · metric bars · radar · heatmaps · purity        ║
║            overlap matrices · cluster sizes · stability · taxa heatmap      ║
║            DR metrics (trustworthiness etc.) · MASTER dashboard             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
─────
  # Step 1 — run preprocessing:
  #   python microbiome_preprocessing_full.py
  #
  # Step 2 — fill in your GNN results in GNN_RESULTS below
  #
  # Step 3 — run evaluation:
  #   python microbiome_clustering_evaluation_full.py

Dependencies
────────────
  pip install numpy pandas scikit-learn scipy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings, os
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score,
    silhouette_score, homogeneity_score, completeness_score,
    v_measure_score, calinski_harabasz_score, davies_bouldin_score,
    confusion_matrix,
)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "preprocessed_dir" : "preprocessed",
    "output_dir"       : "evaluation",
    "abundance_csv"    : "abundance.csv",
    "metadata_cols"    : 210,
    "disease_col_kw"   : "disease",
    "kmeans_n_init"    : 20,
    "kmeans_max_iter"  : 500,
    "stability_runs"   : 10,
    "sil_sample"       : 2000,
}

# ─────────────────────────────────────────────────────────────────────────────
# GNN RESULTS — paste your actual scores here after running your GNN scripts
# ─────────────────────────────────────────────────────────────────────────────
GNN_RESULTS = {
    "KNN+DMoN": {
        "NMI": 0.1818, "ARI": 0.0275, "Silhouette_cosine": 0.5509,
        "Homogeneity": None, "Completeness": None, "V_measure": None,
        "Calinski_Harabasz": None, "Davies_Bouldin": None,
    },
    "KNN+MinCutPool": {
        "NMI": None, "ARI": None, "Silhouette_cosine": None,
        "Homogeneity": None, "Completeness": None, "V_measure": None,
        "Calinski_Harabasz": None, "Davies_Bouldin": None,
    },
    "Bipartite+DMoN": {
        "NMI": None, "ARI": None, "Silhouette_cosine": None,
        "Homogeneity": None, "Completeness": None, "V_measure": None,
        "Calinski_Harabasz": None, "Davies_Bouldin": None,
    },
    "Bipartite+MinCutPool": {
        "NMI": None, "ARI": None, "Silhouette_cosine": None,
        "Homogeneity": None, "Completeness": None, "V_measure": None,
        "Calinski_Harabasz": None, "Davies_Bouldin": None,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & STYLE
# ─────────────────────────────────────────────────────────────────────────────
DARK  = "#0d0d1a"; PANEL = "#13132b"; GRID = "#2a2a4a"
TEXT  = "#e8e8ff"; TEXT2 = "#9090c0"
PAL   = ["#00d4ff","#ff6b6b","#ffd700","#b388ff","#69ff85","#ff9f43",
         "#fd79a8","#a29bfe","#55efc4","#fdcb6e","#e17055","#74b9ff",
         "#00cec9","#6c5ce7","#fab1a0","#dfe6e9","#2d3436","#636e72"]
# Fixed colours per method family
GNN_COLORS = {"KNN+DMoN":"#ff6b6b","KNN+MinCutPool":"#fd79a8",
              "Bipartite+DMoN":"#b388ff","Bipartite+MinCutPool":"#a29bfe"}
DR_COLORS  = {"PCA":"#00d4ff","Jaccard PCoA":"#69ff85","MDS":"#ffd700",
              "t-SNE":"#ff9f43","UMAP":"#55efc4","PaCMAP":"#74b9ff",
              "PHATE":"#00cec9","SONG":"#e17055"}

def _sec(t): print(f"\n{'═'*64}\n  {t}\n{'═'*64}")
def _log(t, m): print(f"  [{t}]{'─'*max(52-len(t),1)} {m}")

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=7)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT2, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT2, fontsize=8)
    ax.grid(True, color=GRID, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    if title: ax.set_title(title, color=TEXT, fontsize=8, fontweight="bold", pad=6)

def fv(v):
    """Format value — return float or string if not castable."""
    if v is None or str(v) in ("None","nan",""):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return v

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(cfg):
    _sec("MODULE 1 — Load preprocessed data")
    d = cfg["preprocessed_dir"]

    X_clr  = pd.read_csv(f"{d}/X_clr.csv",        index_col=0).values.astype(np.float32)
    X_pca  = pd.read_csv(f"{d}/X_pca_kmeans.csv",  index_col=0).values.astype(np.float32)
    y_true = np.load(f"{d}/y_true.npy")

    # Load embeddings
    embeddings = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("embedding_") and f.endswith(".csv"):
            name = f.replace("embedding_","").replace(".csv","").replace("_"," ")
            emb  = pd.read_csv(f"{d}/{f}", index_col=0).values
            embeddings[name] = emb

    # Taxa names
    taxa_path  = f"{d}/taxa_names.csv"
    taxa_names = (pd.read_csv(taxa_path)["taxon"].tolist()
                  if os.path.exists(taxa_path) else
                  [f"Taxon_{i}" for i in range(X_clr.shape[1])])

    # DR evaluation metrics
    dr_metrics_path = f"{d}/dr_evaluation_metrics.csv"
    dr_df = (pd.read_csv(dr_metrics_path)
             if os.path.exists(dr_metrics_path) else pd.DataFrame())

    # Load the cleaned disease labels saved by the preprocessing script
    # This guarantees the label encoder matches y_true exactly (no noise classes)
    _NOISE = {"nd", "na", "-", " -", "unknown_raw", "nan", "none", ""}
    dis_label_path = f"{d}/disease_labels.csv"
    if os.path.exists(dis_label_path):
        disease_series = pd.read_csv(dis_label_path).iloc[:, 0].astype(str).str.strip().str.lower()
        disease_series = disease_series.where(~disease_series.isin(_NOISE), other="unknown")
    else:
        # Fallback: re-read and clean the same way preprocessing does
        df      = pd.read_csv(cfg["abundance_csv"], low_memory=False)
        meta    = df.iloc[:, :cfg["metadata_cols"]]
        dis_col = next(c for c in meta.columns if cfg["disease_col_kw"].lower() in c.lower())
        raw     = meta[dis_col].astype(str).str.strip().str.lower()
        disease_series = raw.where(~raw.isin(_NOISE), other="unknown").reset_index(drop=True)

    le = LabelEncoder()
    le.fit(disease_series)

    _log("X_clr",      f"{X_clr.shape}")
    _log("X_pca",      f"{X_pca.shape}")
    _log("embeddings", str(list(embeddings.keys())))
    _log("classes",    f"{len(le.classes_)}: {list(le.classes_)}")

    return X_clr, X_pca, y_true, le, embeddings, taxa_names, dr_df, disease_series

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — K-MEANS ON EVERY DR EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
def run_kmeans_all(embeddings, X_clr, X_pca, n_clusters, cfg):
    _sec("MODULE 2 — K-Means on all DR embeddings")
    results = {}
    spaces  = {"CLR": X_clr, "AitchisonPCA": X_pca}
    spaces.update({f"KM_{k}": v for k,v in embeddings.items() if v is not None})

    for space_name, X in spaces.items():
        if X is None: continue
        km = KMeans(n_clusters=n_clusters, init="k-means++",
                    n_init=cfg["kmeans_n_init"], max_iter=cfg["kmeans_max_iter"],
                    random_state=42)
        labels = km.fit_predict(X)
        results[space_name] = {"labels": labels, "model": km}
        _log(f"K-Means on {space_name}", f"done  k={n_clusters}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — COMPUTE ALL CLUSTERING METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(labels, X_eval, y_true, name, sil_size=2000):
    """Compute 8 clustering metrics. X_eval used for Silhouette/CH/DB."""
    def _s(fn, *a, **kw):
        try: return fn(*a, **kw)
        except: return np.nan

    ss_cos = _s(silhouette_score, X_eval, labels, metric="cosine",
                sample_size=min(sil_size, len(labels)), random_state=42)
    ss_euc = _s(silhouette_score, X_eval, labels, metric="euclidean",
                sample_size=min(sil_size, len(labels)), random_state=42)
    return {
        "Method"             : name,
        "NMI"                : _s(normalized_mutual_info_score, y_true, labels),
        "ARI"                : _s(adjusted_rand_score, y_true, labels),
        "Silhouette_cosine"  : ss_cos,
        "Silhouette_euclid"  : ss_euc,
        "Homogeneity"        : _s(homogeneity_score, y_true, labels),
        "Completeness"       : _s(completeness_score, y_true, labels),
        "V_measure"          : _s(v_measure_score, y_true, labels),
        "Calinski_Harabasz"  : _s(calinski_harabasz_score, X_eval, labels),
        "Davies_Bouldin"     : _s(davies_bouldin_score, X_eval, labels),
    }

def build_full_metrics_table(km_results, X_clr, y_true, cfg):
    _sec("MODULE 3 — Full metrics table")
    rows = []

    import json
    import os
    json_path = f"{cfg['preprocessed_dir']}/gnn_results.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                GNN_RESULTS.update(json.load(f))
        except Exception as e:
            _log("WARNING", f"Could not load GNN results: {e}")

    # K-Means variants
    for space_name, res in km_results.items():
        labels = res["labels"]
        X_eval = X_clr   # always evaluate in CLR space for comparability
        display_name = (f"KMeans ({space_name})" if not space_name.startswith("KM_")
                        else f"KMeans ({space_name[3:]})")
        m = compute_metrics(labels, X_eval, y_true, display_name, cfg["sil_sample"])
        rows.append(m)
        _log(display_name,
             f"NMI={m['NMI']:.4f}  ARI={m['ARI']:.4f}  SIL={m['Silhouette_cosine']:.4f}")

    # GNN results from config
    for gnn_name, scores in GNN_RESULTS.items():
        row = {"Method": gnn_name}
        row.update({k: fv(v) for k, v in scores.items()})
        rows.append(row)
        nmi = fv(scores.get("NMI"))
        ari = fv(scores.get("ARI"))
        sil = fv(scores.get("Silhouette_cosine"))
        _log(gnn_name,
             f"NMI={nmi:.4f}  ARI={ari:.4f}  SIL={sil:.4f}"
             if not np.isnan(nmi) else "scores not yet available — fill GNN_RESULTS")

    df = pd.DataFrame(rows)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — PURITY
# ─────────────────────────────────────────────────────────────────────────────
def purity_analysis(labels, y_true, le, method_name):
    rows = []
    for cid in np.unique(y_true):
        mask = y_true == cid
        if not mask.any(): continue
        bc     = np.bincount(labels[mask], minlength=int(labels.max())+1)
        probs  = bc[bc>0] / bc.sum()
        rows.append({
            "Disease"     : le.classes_[cid],
            "N"           : int(mask.sum()),
            "Main_Cluster": int(bc.argmax()),
            "Purity"      : round(float(bc.max()/mask.sum()), 4),
            "Entropy_bits": round(float(-np.sum(probs*np.log2(probs+1e-10))), 4),
            "Method"      : method_name,
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — DR EMBEDDING GRID (coloured by disease)
# ─────────────────────────────────────────────────────────────────────────────
def fig_dr_embedding_grid(embeddings, disease_series, out_dir):
    _sec("FIGURE 1 — DR embedding grid (disease colours)")
    valid = [(n,e) for n,e in embeddings.items() if e is not None]
    if not valid: return

    diseases  = sorted(disease_series.unique())   # sort for stable colour assignment
    all_labels = sorted(set(disease_series.unique()))
    cmap       = {d: PAL[i % len(PAL)] for i, d in enumerate(all_labels)}
    n_cols   = 4
    n_rows   = (len(valid) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), facecolor=DARK)
    axes = np.array(axes).flatten()

    for idx, (name, emb) in enumerate(valid):
        ax = axes[idx]
        style_ax(ax, name, "Component 1", "Component 2")
        for dis in diseases:
            mask = disease_series.values == dis
            col  = cmap.get(dis, "#888888")   # default grey for unseen labels
            ax.scatter(emb[mask,0], emb[mask,1], c=[col],
                       label=dis, alpha=0.7, s=18, linewidths=0)
        if idx == 0:
            ax.legend(title="Disease", fontsize=6, facecolor=PANEL,
                      edgecolor=GRID, labelcolor=TEXT, markerscale=1.5,
                      loc="best", framealpha=0.8)

    for i in range(len(valid), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Dimensionality reduction embeddings — coloured by disease",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{out_dir}/fig1_dr_embedding_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — DR EMBEDDING GRID (coloured by K-Means cluster)
# ─────────────────────────────────────────────────────────────────────────────
def fig_dr_kmeans_grid(embeddings, km_results, n_clusters, out_dir):
    _sec("FIGURE 2 — DR embedding grid (K-Means cluster colours)")
    valid = [(n,e) for n,e in embeddings.items() if e is not None]
    if not valid: return

    n_cols = 4
    n_rows = (len(valid) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), facecolor=DARK)
    axes = np.array(axes).flatten()

    for idx, (name, emb) in enumerate(valid):
        ax = axes[idx]
        key = f"KM_{name}"
        labels = km_results.get(key, km_results.get("CLR", {})).get("labels", None)
        if labels is None:
            axes[idx].set_visible(False); continue
        style_ax(ax, f"K-Means on {name}", "Dim 1", "Dim 2")
        for cid in np.unique(labels):
            mask = labels == cid
            ax.scatter(emb[mask,0], emb[mask,1], c=PAL[int(cid)%len(PAL)],
                       label=f"C{cid}", alpha=0.75, s=18, linewidths=0)
        if idx == 0 and n_clusters <= 12:
            ax.legend(fontsize=6, facecolor=PANEL, edgecolor=GRID,
                      labelcolor=TEXT, markerscale=1.5, loc="best")

    for i in range(len(valid), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("K-Means cluster assignments per DR embedding",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{out_dir}/fig2_dr_kmeans_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — DR QUALITY METRICS (7 metrics, 4-top / 3-bottom layout)
# ─────────────────────────────────────────────────────────────────────────────
def fig_dr_quality_metrics(dr_df, out_dir):
    _sec("FIGURE 3 — DR quality metrics")
    if dr_df.empty: return

    metric_names = ["Trustworthiness","Continuity","KNN Preservation","Normalized Stress",
                    "Bray-Curtis Corr","Aitchison Corr","Jaccard Corr"]
    metric_names = [m for m in metric_names if m in dr_df.columns]
    if not metric_names: return

    colors = [DR_COLORS.get(m, PAL[i%len(PAL)])
              for i, m in enumerate(dr_df["Method"].tolist())]

    fig  = plt.figure(figsize=(22, 12), facecolor=DARK)
    gs   = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.35)

    for idx, metric in enumerate(metric_names):
        row_i = idx // 4; col_i = idx % 4
        ax    = fig.add_subplot(gs[row_i, col_i])
        style_ax(ax, metric)
        vals     = dr_df[metric].fillna(0).values
        methods  = dr_df["Method"].values
        bar_cols = [DR_COLORS.get(m, PAL[i%len(PAL)]) for i,m in enumerate(methods)]
        bars = ax.barh(methods, vals, color=bar_cols, edgecolor="white",
                       linewidth=0.4, zorder=3)
        if metric == "Normalized Stress":
            ax.set_title(f"{metric}\n(lower = better)", color=TEXT, fontsize=8,
                         fontweight="bold", pad=6)
        else:
            ax.set_xlim(0, 1.12)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", color="white", fontsize=8)

    fig.suptitle("DR embedding quality — 7 metrics",
                 color=TEXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{out_dir}/fig3_dr_quality_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — DR METRICS HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def fig_dr_metrics_heatmap(dr_df, out_dir):
    _sec("FIGURE 4 — DR metrics heatmap")
    if dr_df.empty: return

    metric_cols = [c for c in dr_df.columns if c != "Method"]
    heat_data   = dr_df.set_index("Method")[metric_cols].astype(float)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK)
    ax.set_facecolor(PANEL)
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0.5, ax=ax, cbar_kws={"label":"Score"},
                linewidths=0.4, linecolor=GRID)
    ax.set_title("DR quality metrics heatmap", color=TEXT, fontsize=11,
                 fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT)
    ax.collections[0].colorbar.ax.yaxis.set_tick_params(color=TEXT)
    ax.collections[0].colorbar.ax.yaxis.label.set_color(TEXT)

    plt.tight_layout()
    path = f"{out_dir}/fig4_dr_metrics_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — CLUSTERING METRIC BARS (NMI / ARI / Silhouette — all 12 methods)
# ─────────────────────────────────────────────────────────────────────────────
def fig_clustering_metric_bars(metrics_df, out_dir):
    _sec("FIGURE 5 — Clustering metric bars (all methods)")
    metrics_to_show = ["NMI","ARI","Silhouette_cosine","Homogeneity","Completeness","V_measure"]
    nice_names      = ["NMI","ARI","Silhouette\n(cosine)","Homogeneity","Completeness","V-measure"]

    n_meth = len(metrics_df)
    # Colour: GNN methods get their fixed colour, KMeans gets DR colour
    def method_color(name):
        for g in GNN_COLORS:
            if g.lower() in name.lower(): return GNN_COLORS[g]
        for d in DR_COLORS:
            if d.lower() in name.lower(): return DR_COLORS[d]
        return PAL[0]

    bar_colors = [method_color(m) for m in metrics_df["Method"]]

    fig, axes = plt.subplots(1, len(metrics_to_show),
                             figsize=(24, max(n_meth * 0.55 + 2, 7)),
                             facecolor=DARK)
    fig.subplots_adjust(wspace=0.35)

    for ax, metric, nice in zip(axes, metrics_to_show, nice_names):
        style_ax(ax, nice)
        vals = [fv(v) for v in metrics_df[metric]]
        # replace nan with 0 for display, mark them
        vals_plot = [v if not np.isnan(v) else 0.0 for v in vals]
        methods   = list(metrics_df["Method"])

        y = np.arange(n_meth)
        bars = ax.barh(y, vals_plot, color=bar_colors, edgecolor="white",
                       linewidth=0.4, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=7, color=TEXT)
        ax.set_xlim(0, max(max(v for v in vals_plot), 0.1) * 1.35)

        for bar, val, val_orig in zip(bars, vals_plot, vals):
            label = f"{val:.3f}" if not np.isnan(val_orig) else "—"
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    label, va="center", color="white", fontsize=7, fontweight="bold")

        ax.axvline(0, color="white", lw=0.6, linestyle="--", alpha=0.3)

    # Legend for method families
    from matplotlib.patches import Patch
    legend_items = (
        [Patch(facecolor=c, label=n) for n,c in GNN_COLORS.items()] +
        [Patch(facecolor=PAL[0], label="K-Means (CLR/PCA)")]
    )
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Clustering evaluation — all 12 method combinations",
                 color=TEXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{out_dir}/fig5_clustering_metric_bars.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — RADAR CHART (all methods, key metrics)
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar(metrics_df, out_dir):
    _sec("FIGURE 6 — Radar chart")
    radar_metrics = ["NMI","ARI","Silhouette_cosine","Homogeneity","Completeness","V_measure"]
    radar_labels  = ["NMI","ARI","Silhouette","Homogeneity","Completeness","V-measure"]
    N = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=DARK,
                           subplot_kw=dict(polar=True))
    ax.set_facecolor(PANEL); ax.spines["polar"].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(color=GRID, linestyle="--", alpha=0.5)

    def method_color(name):
        for g in GNN_COLORS:
            if g.lower() in name.lower(): return GNN_COLORS[g]
        for d in DR_COLORS:
            if d.lower() in name.lower(): return DR_COLORS[d]
        return PAL[hash(name) % len(PAL)]

    for _, row in metrics_df.iterrows():
        vals = [fv(row.get(m)) for m in radar_metrics]
        if all(np.isnan(v) for v in vals): continue
        vals = [0.0 if np.isnan(v) else v for v in vals]
        vals += vals[:1]
        col = method_color(row["Method"])
        ax.plot(angles, vals, color=col, lw=1.5, label=row["Method"])
        ax.fill(angles, vals, color=col, alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, color=TEXT, fontsize=9)
    ax.set_ylim(0, 1); ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], color=TEXT2, fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15),
              facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    ax.set_title("All methods — metric radar", color=TEXT, fontsize=12,
                 fontweight="bold", pad=20)

    plt.tight_layout()
    path = f"{out_dir}/fig6_radar_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — OVERLAP HEATMAPS (K-Means CLR + KNN+DMoN)
# ─────────────────────────────────────────────────────────────────────────────
def fig_overlap_heatmaps(km_results, y_true, le, out_dir):
    _sec("FIGURE 7 — Overlap heatmaps")
    target_keys = ["CLR","AitchisonPCA"] + [k for k in km_results if k.startswith("KM_")]
    target_keys = target_keys[:4]   # show top 4 to keep figure readable

    n_show = len(target_keys)
    fig, axes = plt.subplots(n_show, 2, figsize=(16, 6*n_show), facecolor=DARK)
    if n_show == 1: axes = axes.reshape(1,2)
    fig.suptitle("Disease → cluster overlap heatmaps (row-normalised)",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.01)

    cmaps = ["Blues","Purples","Oranges","Greens"]
    for ri, key in enumerate(target_keys):
        labels = km_results[key]["labels"]
        name   = key.replace("KM_","")
        cm_raw = confusion_matrix(y_true, labels)
        cm_n   = cm_raw / (cm_raw.sum(axis=1, keepdims=True) + 1e-8)

        for ci, (data, title, cmap) in enumerate([
            (cm_raw, f"K-Means ({name}) — raw counts", "Blues"),
            (cm_n,   f"K-Means ({name}) — row-normalised", cmaps[ri]),
        ]):
            ax = axes[ri, ci]
            im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=6)
            ax.set_xlabel("Cluster", color=TEXT2, fontsize=8)
            ax.set_ylabel("True disease", color=TEXT2, fontsize=8)
            ax.set_yticks(range(len(le.classes_)))
            ax.set_yticklabels(le.classes_, fontsize=7, color=TEXT)
            ax.set_xticks(range(data.shape[1]))
            ax.set_xticklabels([f"C{c}" for c in range(data.shape[1])], color=TEXT, fontsize=7)
            ax.tick_params(colors=TEXT)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=TEXT)
            thresh = data.max() * 0.5
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v   = data[i,j]
                    txt = str(int(v)) if ci==0 else f"{v:.2f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            color="black" if v > thresh else "white",
                            fontsize=6, fontweight="bold")

    plt.tight_layout()
    path = f"{out_dir}/fig7_overlap_heatmaps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — PER-DISEASE PURITY (GNN methods + best KMeans)
# ─────────────────────────────────────────────────────────────────────────────
def fig_purity_comparison(km_results, y_true, le, out_dir):
    _sec("FIGURE 8 — Per-disease purity")

    # Best K-Means by NMI (already computed externally) — use CLR
    keys  = ["CLR"] + list(GNN_RESULTS.keys())[:4]
    names = [f"KMeans (CLR)"] + list(GNN_RESULTS.keys())[:4]

    fig, axes = plt.subplots(1, len(keys), figsize=(5*len(keys), 6), facecolor=DARK)
    if len(keys) == 1: axes = [axes]
    fig.suptitle("Per-disease cluster purity", color=TEXT, fontsize=12,
                 fontweight="bold")

    for ax, key, name in zip(axes, keys, names):
        if key in km_results:
            labels  = km_results[key]["labels"]
            purity  = purity_analysis(labels, y_true, le, name)
        else:
            ax.set_visible(False); continue

        style_ax(ax, name, "Disease", "Purity")
        diseases = purity["Disease"].tolist()
        purities = purity["Purity"].tolist()
        bc       = [PAL[i%len(PAL)] for i in range(len(diseases))]
        bars     = ax.bar(range(len(diseases)), purities, color=bc,
                          edgecolor="white", linewidth=0.4, zorder=3)
        for b, p in zip(bars, purities):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{p:.2f}", ha="center", va="bottom",
                    color="white", fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(diseases)))
        ax.set_xticklabels(diseases, rotation=40, ha="right", fontsize=7, color=TEXT)
        ax.set_ylim(0, 1.15)
        ax.axhline(1/len(le.classes_), color="#ffd700", linestyle="--",
                   lw=1, alpha=0.7, label=f"Random ({1/len(le.classes_):.2f})")
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    plt.tight_layout()
    path = f"{out_dir}/fig8_purity_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — CLUSTER SIZE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
def fig_cluster_sizes(km_results, y_true, le, out_dir):
    _sec("FIGURE 9 — Cluster size distributions")
    keys = list(km_results.keys())[:6] + ["ground_truth"]

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5), facecolor=DARK)
    if n == 1: axes = [axes]
    fig.suptitle("Cluster size distributions — predicted vs ground truth",
                 color=TEXT, fontsize=11, fontweight="bold")

    for ax, key in zip(axes, keys):
        if key == "ground_truth":
            labels = y_true; title = "Ground truth"
            x_labels = le.classes_
        else:
            labels = km_results[key]["labels"]
            title  = f"KMeans\n({key.replace('KM_','')})"
            x_labels = [f"C{c}" for c in np.unique(labels)]

        style_ax(ax, title, "Cluster", "# Samples")
        ids, cnts = np.unique(labels, return_counts=True)
        bc = [PAL[i%len(PAL)] for i in ids]
        bars = ax.bar(range(len(ids)), cnts, color=bc,
                      edgecolor="white", linewidth=0.4, zorder=3)
        for b, cnt in zip(bars, cnts):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                    str(cnt), ha="center", va="bottom",
                    color="white", fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=7, color=TEXT)
        ax.axhline(len(labels)/len(ids), color="#ffd700", linestyle="--", lw=1,
                   alpha=0.7, label="Balanced")
        ax.legend(fontsize=6, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    plt.tight_layout()
    path = f"{out_dir}/fig9_cluster_sizes.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — K-MEANS STABILITY
# ─────────────────────────────────────────────────────────────────────────────
def fig_stability(X_clr, n_clusters, y_true, cfg, out_dir):
    _sec("FIGURE 10 — K-Means stability (CLR space)")
    seeds = list(range(cfg["stability_runs"]))
    nmi_v, ari_v, sil_v = [], [], []
    for seed in seeds:
        km  = KMeans(n_clusters=n_clusters, init="k-means++",
                     n_init=10, max_iter=300, random_state=seed)
        lbl = km.fit_predict(X_clr)
        nmi_v.append(normalized_mutual_info_score(y_true, lbl))
        ari_v.append(adjusted_rand_score(y_true, lbl))
        sil_v.append(silhouette_score(X_clr, lbl, metric="cosine",
                                      sample_size=cfg["sil_sample"], random_state=42))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK)
    fig.suptitle(f"K-Means stability — {cfg['stability_runs']} seeds",
                 color=TEXT, fontsize=12, fontweight="bold")
    for ax, vals, name in zip(axes,
                               [nmi_v, ari_v, sil_v],
                               ["NMI","ARI","Silhouette (cosine)"]):
        style_ax(ax, name, "Seed", "Score")
        ax.plot(seeds, vals, color="#00d4ff", lw=2, marker="o", markersize=6)
        ax.axhline(np.mean(vals), color="#ffd700", linestyle="--", lw=1.5,
                   label=f"μ={np.mean(vals):.4f}")
        ax.fill_between(seeds,
                        np.mean(vals)-np.std(vals),
                        np.mean(vals)+np.std(vals),
                        alpha=0.12, color="#00d4ff")
        ax.set_xticks(seeds)
        ax.text(0.98, 0.06, f"μ={np.mean(vals):.4f}\nσ={np.std(vals):.4f}",
                transform=ax.transAxes, ha="right", va="bottom",
                color=TEXT2, fontsize=9,
                bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle="round,pad=0.3"))
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    plt.tight_layout()
    path = f"{out_dir}/fig10_stability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)
    return nmi_v, ari_v, sil_v

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 — TOP TAXA HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def fig_taxa_heatmap(X_clr, km_labels, taxa_names, out_dir, top_n=30):
    _sec("FIGURE 11 — Top discriminative taxa heatmap")
    n_clust = len(np.unique(km_labels))
    clust_means = np.array([X_clr[km_labels==c].mean(axis=0)
                            for c in range(n_clust)])
    top_idx = np.argsort(clust_means.var(axis=0))[::-1][:top_n]
    heat_df = pd.DataFrame(
        {f"C{c}": X_clr[km_labels==c][:,top_idx].mean(axis=0) for c in range(n_clust)},
        index=[taxa_names[i] if i < len(taxa_names) else f"T{i}" for i in top_idx]
    )
    cmap = LinearSegmentedColormap.from_list("clr", ["#00d4ff","#13132b","#ff6b6b"], N=256)
    fig, ax = plt.subplots(figsize=(13, 11), facecolor=DARK)
    ax.set_facecolor(PANEL)
    sns.heatmap(heat_df, ax=ax, cmap=cmap, center=0,
                linewidths=0.3, linecolor=GRID,
                cbar_kws={"shrink":0.6,"label":"Mean CLR"})
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_xlabel("K-Means cluster (CLR)", color=TEXT2, fontsize=9)
    ax.set_title(f"Top {top_n} discriminative taxa — mean CLR per K-Means cluster",
                 color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.collections[0].colorbar.ax.yaxis.set_tick_params(color=TEXT)
    ax.collections[0].colorbar.ax.yaxis.label.set_color(TEXT)
    plt.tight_layout()
    path = f"{out_dir}/fig11_taxa_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 12 — MASTER DASHBOARD  (4×4 composite)
# ─────────────────────────────────────────────────────────────────────────────
def fig_master_dashboard(embeddings, disease_series, km_results,
                         metrics_df, y_true, le, stability_nmi, out_dir):
    _sec("FIGURE 12 — Master dashboard")

    fig = plt.figure(figsize=(30, 22), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.32)

    diseases = disease_series.unique()
    d_cmap   = dict(zip(diseases, PAL[:len(diseases)]))
    n_cls    = len(le.classes_)

    # ── Row 0: top-4 DR embeddings coloured by disease ───────────────────────
    valid_emb = [(n,e) for n,e in embeddings.items() if e is not None][:4]
    for col_i, (name, emb) in enumerate(valid_emb):
        ax = fig.add_subplot(gs[0, col_i])
        style_ax(ax, f"{name} — disease", "Dim1", "Dim2")
        for dis in diseases:
            mask = disease_series.values == dis
            ax.scatter(emb[mask,0], emb[mask,1],
                       c=[d_cmap[dis]], s=7, alpha=0.75, linewidths=0, label=dis)
        if col_i == 0 and n_cls <= 10:
            ax.legend(fontsize=5, facecolor=PANEL, edgecolor=GRID,
                      labelcolor=TEXT, markerscale=2, ncol=2, loc="best")

    # ── Row 1: NMI / ARI / Silhouette bars + Stability ───────────────────────
    for col_i, (metric, nice) in enumerate(
            [("NMI","NMI"), ("ARI","ARI"), ("Silhouette_cosine","Silhouette")]):
        ax = fig.add_subplot(gs[1, col_i])
        style_ax(ax, nice)

        def mc(n):
            for g in GNN_COLORS:
                if g.lower() in n.lower(): return GNN_COLORS[g]
            for d in DR_COLORS:
                if d.lower() in n.lower(): return DR_COLORS[d]
            return PAL[0]

        vals    = [fv(v) for v in metrics_df[metric]]
        methods = list(metrics_df["Method"])
        vp      = [v if not np.isnan(v) else 0.0 for v in vals]
        colors  = [mc(m) for m in methods]
        y = np.arange(len(methods))
        ax.barh(y, vp, color=colors, edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=6, color=TEXT)
        for yi, (v, vo) in enumerate(zip(vp, vals)):
            ax.text(v+0.005, yi, f"{v:.3f}" if not np.isnan(vo) else "—",
                    va="center", color="white", fontsize=6)

    ax_stab = fig.add_subplot(gs[1, 3])
    style_ax(ax_stab, "NMI stability (K-Means CLR)", "Seed", "NMI")
    seeds = list(range(len(stability_nmi)))
    ax_stab.plot(seeds, stability_nmi, color="#00d4ff", lw=2, marker="o", markersize=5)
    ax_stab.axhline(np.mean(stability_nmi), color="#ffd700", linestyle="--", lw=1,
                    label=f"μ={np.mean(stability_nmi):.4f}")
    ax_stab.fill_between(seeds,
                         np.mean(stability_nmi)-np.std(stability_nmi),
                         np.mean(stability_nmi)+np.std(stability_nmi),
                         alpha=0.12, color="#00d4ff")
    ax_stab.set_xticks(seeds)
    ax_stab.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Row 2: Confusion heatmap (K-Means CLR) + Cluster sizes + purity ──────
    ax_conf = fig.add_subplot(gs[2, 0:2])
    cm_n    = confusion_matrix(y_true, km_results["CLR"]["labels"]).astype(float)
    cm_n   /= (cm_n.sum(axis=1, keepdims=True) + 1e-8)
    im      = ax_conf.imshow(cm_n, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax_conf.set_facecolor(PANEL)
    ax_conf.set_title("K-Means (CLR) — row-normalised overlap",
                      color=TEXT, fontsize=9, fontweight="bold", pad=6)
    ax_conf.set_xlabel("Cluster", color=TEXT2, fontsize=8)
    ax_conf.set_ylabel("True disease", color=TEXT2, fontsize=8)
    ax_conf.set_yticks(range(len(le.classes_)))
    ax_conf.set_yticklabels(le.classes_, fontsize=7, color=TEXT)
    ax_conf.set_xticks(range(cm_n.shape[1]))
    ax_conf.set_xticklabels([f"C{c}" for c in range(cm_n.shape[1])], color=TEXT, fontsize=7)
    ax_conf.tick_params(colors=TEXT)
    plt.colorbar(im, ax=ax_conf, fraction=0.023, pad=0.02).ax.yaxis.set_tick_params(color=TEXT)
    thresh = cm_n.max() * 0.5
    for i in range(cm_n.shape[0]):
        for j in range(cm_n.shape[1]):
            ax_conf.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                         color="black" if cm_n[i,j] > thresh else "white",
                         fontsize=6, fontweight="bold")

    ax_sz = fig.add_subplot(gs[2, 2])
    style_ax(ax_sz, "Cluster sizes (CLR)", "Cluster", "# Samples")
    ids, cnts = np.unique(km_results["CLR"]["labels"], return_counts=True)
    ax_sz.bar(range(len(ids)), cnts,
              color=[PAL[i%len(PAL)] for i in ids],
              edgecolor="white", linewidth=0.4)
    ax_sz.set_xticks(range(len(ids)))
    ax_sz.set_xticklabels([f"C{c}" for c in ids], fontsize=7, color=TEXT)
    for i, cnt in enumerate(cnts):
        ax_sz.text(i, cnt+2, str(cnt), ha="center", va="bottom",
                   color="white", fontsize=7)

    ax_pur = fig.add_subplot(gs[2, 3])
    style_ax(ax_pur, "Per-disease purity (CLR)", "", "Purity")
    pur_df = purity_analysis(km_results["CLR"]["labels"], y_true, le, "CLR")
    ax_pur.bar(range(len(pur_df)), pur_df["Purity"].values,
               color=[PAL[i%len(PAL)] for i in range(len(pur_df))],
               edgecolor="white", linewidth=0.4)
    ax_pur.set_xticks(range(len(pur_df)))
    ax_pur.set_xticklabels(pur_df["Disease"].values, rotation=40,
                           ha="right", fontsize=6, color=TEXT)
    ax_pur.set_ylim(0, 1.15)
    ax_pur.axhline(1/n_cls, color="#ffd700", linestyle="--", lw=1, alpha=0.7)

    # Title
    km_clr = metrics_df[metrics_df["Method"].str.contains("KMeans", regex=False)]
    nmi_clr = fv(km_clr["NMI"].values[0]) if len(km_clr) else 0.0
    ari_clr = fv(km_clr["ARI"].values[0]) if len(km_clr) else 0.0
    fig.suptitle(
        f"Microbiome clustering — master dashboard  |  "
        f"K-Means (CLR): NMI={nmi_clr:.4f}  ARI={ari_clr:.4f}  |  "
        f"KNN+DMoN: NMI={GNN_RESULTS['KNN+DMoN']['NMI']:.4f}  "
        f"ARI={GNN_RESULTS['KNN+DMoN']['ARI']:.4f}",
        color=TEXT, fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = f"{out_dir}/fig12_master_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    _log("saved", path)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE REPORTS
# ─────────────────────────────────────────────────────────────────────────────
def save_reports(metrics_df, dr_df, stability_nmi, stability_ari,
                 stability_sil, out_dir):
    _sec("Save reports")
    metrics_df.to_csv(f"{out_dir}/all_methods_metrics.csv", index=False)
    _log("saved", "all_methods_metrics.csv")
    if not dr_df.empty:
        dr_df.to_csv(f"{out_dir}/dr_quality_metrics.csv", index=False)
        _log("saved", "dr_quality_metrics.csv")
    pd.DataFrame({
        "seed": list(range(len(stability_nmi))),
        "nmi": stability_nmi, "ari": stability_ari, "sil": stability_sil,
    }).to_csv(f"{out_dir}/kmeans_stability.csv", index=False)
    _log("saved", "kmeans_stability.csv")

    # Markdown report
    best_row = metrics_df.sort_values("NMI", ascending=False).iloc[0]
    md  = f"# Microbiome clustering — full evaluation report\n\n"
    md += f"## Best method by NMI: **{best_row['Method']}**  NMI={fv(best_row['NMI']):.4f}\n\n"
    md += "## All methods — clustering metrics\n\n"
    cols = ["Method","NMI","ARI","Silhouette_cosine","Homogeneity",
            "Completeness","V_measure","Calinski_Harabasz","Davies_Bouldin"]
    md += "| " + " | ".join(cols) + " |\n"
    md += "| " + " | ".join(["---"]*len(cols)) + " |\n"
    for _, row in metrics_df.iterrows():
        vals = []
        for c in cols:
            v = fv(row.get(c))
            vals.append(f"{v:.4f}" if not np.isnan(v) else "—")
        md += "| " + " | ".join(vals) + " |\n"

    md += f"\n## K-Means stability  ({len(stability_nmi)} seeds)\n"
    md += f"| NMI μ | NMI σ | ARI μ | ARI σ | SIL μ | SIL σ |\n|---|---|---|---|---|---|\n"
    md += (f"| {np.mean(stability_nmi):.4f} | {np.std(stability_nmi):.4f} "
           f"| {np.mean(stability_ari):.4f} | {np.std(stability_ari):.4f} "
           f"| {np.mean(stability_sil):.4f} | {np.std(stability_sil):.4f} |\n")

    md += "\n## Output figures\n\n"
    fig_list = [
        ("fig1_dr_embedding_grid.png",     "DR embeddings coloured by disease"),
        ("fig2_dr_kmeans_grid.png",        "DR embeddings coloured by K-Means cluster"),
        ("fig3_dr_quality_metrics.png",    "DR quality metrics — 7 bars (4+3 layout)"),
        ("fig4_dr_metrics_heatmap.png",    "DR quality heatmap"),
        ("fig5_clustering_metric_bars.png","All 12 method clustering metrics"),
        ("fig6_radar_chart.png",           "Radar chart — all methods"),
        ("fig7_overlap_heatmaps.png",      "Disease → cluster overlap heatmaps"),
        ("fig8_purity_comparison.png",     "Per-disease purity per method"),
        ("fig9_cluster_sizes.png",         "Cluster size distributions"),
        ("fig10_stability.png",            "K-Means stability analysis"),
        ("fig11_taxa_heatmap.png",         "Top-30 discriminative taxa CLR heatmap"),
        ("fig12_master_dashboard.png",     "MASTER dashboard — composite figure"),
    ]
    for fname, desc in fig_list:
        md += f"- `{fname}` — {desc}\n"

    with open(f"{out_dir}/evaluation_report.md", "w") as f:
        f.write(md)
    _log("saved", "evaluation_report.md")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(cfg=CONFIG):
    print("\n" + "█"*64)
    print("  MICROBIOME CLUSTERING — COMPLETE EVALUATION SUITE")
    print("  4 GNN combinations + 8 DR × K-Means combinations")
    print("█"*64)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 1 Load
    X_clr, X_pca, y_true, le, embeddings, taxa_names, dr_df, disease_series = load_data(cfg)
    n_cls = len(le.classes_)

    # 2 K-Means on all spaces
    km_results = run_kmeans_all(embeddings, X_clr, X_pca, n_cls, cfg)

    # 3 Metrics table (K-Means + GNN)
    metrics_df = build_full_metrics_table(km_results, X_clr, y_true, cfg)

    # 4 Stability
    stab_nmi, stab_ari, stab_sil = fig_stability(X_clr, n_cls, y_true, cfg, out_dir)

    # 5 All figures
    fig_dr_embedding_grid(embeddings, disease_series, out_dir)
    fig_dr_kmeans_grid(embeddings, km_results, n_cls, out_dir)
    fig_dr_quality_metrics(dr_df, out_dir)
    fig_dr_metrics_heatmap(dr_df, out_dir)
    fig_clustering_metric_bars(metrics_df, out_dir)
    fig_radar(metrics_df, out_dir)
    fig_overlap_heatmaps(km_results, y_true, le, out_dir)
    fig_purity_comparison(km_results, y_true, le, out_dir)
    fig_cluster_sizes(km_results, y_true, le, out_dir)
    fig_taxa_heatmap(X_clr, km_results["CLR"]["labels"], taxa_names, out_dir)
    fig_master_dashboard(embeddings, disease_series, km_results,
                         metrics_df, y_true, le, stab_nmi, out_dir)

    # 6 Reports
    save_reports(metrics_df, dr_df, stab_nmi, stab_ari, stab_sil, out_dir)

    # 7 Summary
    _sec("EVALUATION COMPLETE")
    best = metrics_df.sort_values("NMI", ascending=False).iloc[0]
    print(f"\n  Best method by NMI : {best['Method']}  NMI={fv(best['NMI']):.4f}")
    print(f"  K-Means (CLR) stability : σ_NMI={np.std(stab_nmi):.4f}")
    print(f"\n  {len(metrics_df)} methods evaluated")
    print(f"\n  Figures saved → {out_dir}/")
    for fname, desc in [
        ("fig1","DR embeddings (disease)"),
        ("fig2","DR embeddings (K-Means clusters)"),
        ("fig3","DR quality metrics bars"),
        ("fig4","DR quality heatmap"),
        ("fig5","All clustering metric bars"),
        ("fig6","Radar chart"),
        ("fig7","Overlap heatmaps"),
        ("fig8","Per-disease purity"),
        ("fig9","Cluster sizes"),
        ("fig10","Stability"),
        ("fig11","Taxa heatmap"),
        ("fig12","MASTER DASHBOARD"),
    ]:
        print(f"    {fname}_*.png  — {desc}")

    return {"metrics_df": metrics_df, "km_results": km_results,
            "embeddings": embeddings, "stability": dict(
                nmi=stab_nmi, ari=stab_ari, sil=stab_sil)}

if __name__ == "__main__":
    run_evaluation(CONFIG)
