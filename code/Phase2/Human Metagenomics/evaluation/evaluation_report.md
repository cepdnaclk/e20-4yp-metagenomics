# Microbiome clustering — full evaluation report

## Best method by NMI: **KMeans (PaCMAP)**  NMI=0.4744

## All methods — clustering metrics

| Method | NMI | ARI | Silhouette_cosine |
| --- | --- | --- | --- |
| KMeans (Jaccard PCoA) | 0.2685 | 0.0504 | -0.0205 |
| KMeans (MDS) | 0.3350 | 0.0792 | 0.0181 |
| KMeans (PCA) | 0.3451 | 0.0513 | 0.0496 |
| KMeans (PHATE) | 0.4327 | 0.0794 | 0.1578 |
| KMeans (PaCMAP) | 0.4744 | 0.0924 | 0.1845 |
| KMeans (SONG) | 0.4635 | 0.1039 | 0.1693 |
| KMeans (UMAP) | 0.4741 | 0.1055 | 0.2106 |
| KMeans (t-SNE) | 0.4609 | 0.1144 | 0.2049 |
| KNN+DMoN | 0.3030 | 0.0622 | 0.0329 |
| KNN+MinCutPool | 0.3833 | 0.1660 | 0.2472 |

## K-Means stability  (10 seeds)
| NMI μ | NMI σ | ARI μ | ARI σ | SIL μ | SIL σ |
|---|---|---|---|---|---|
| 0.4729 | 0.0140 | 0.1246 | 0.0180 | 0.1719 | 0.0196 |

## Output figures

- `fig1_dr_embedding_grid.png` — DR embeddings coloured by disease
- `fig2_dr_kmeans_grid.png` — DR embeddings coloured by K-Means cluster
- `fig3_dr_quality_metrics.png` — DR quality metrics — 7 bars (4+3 layout)
- `fig4_dr_metrics_heatmap.png` — DR quality heatmap
- `fig5_clustering_metric_bars.png` — All 12 method clustering metrics
- `fig6_radar_chart.png` — Radar chart — all methods
- `fig7_overlap_heatmaps.png` — Disease → cluster overlap heatmaps
- `fig8_purity_comparison.png` — Per-disease purity per method
- `fig9_cluster_sizes.png` — Cluster size distributions
- `fig10_stability.png` — K-Means stability analysis
- `fig11_taxa_heatmap.png` — Top-30 discriminative taxa CLR heatmap
- `fig12_master_dashboard.png` — MASTER dashboard — composite figure
