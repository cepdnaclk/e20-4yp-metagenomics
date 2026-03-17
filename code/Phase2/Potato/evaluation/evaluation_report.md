# Microbiome clustering — full evaluation report

## Best method by NMI: **KMeans (UMAP)**  NMI=0.1993

## All methods — clustering metrics

| Method | NMI | ARI | Silhouette_cosine |
| --- | --- | --- | --- |
| KMeans (Jaccard PCoA) | 0.0878 | 0.0403 | 0.0257 |
| KMeans (MDS) | 0.0717 | 0.0274 | 0.0092 |
| KMeans (PCA) | 0.0937 | 0.0416 | 0.0224 |
| KMeans (PHATE) | 0.1793 | 0.1059 | 0.0385 |
| KMeans (PaCMAP) | 0.1619 | 0.1019 | 0.0552 |
| KMeans (SONG) | 0.1761 | 0.1005 | 0.0389 |
| KMeans (UMAP) | 0.1993 | 0.1231 | 0.0442 |
| KMeans (t-SNE) | 0.1908 | 0.1065 | 0.0403 |
| KNN+DMoN | 0.1030 | 0.0408 | 0.0179 |
| KNN+MinCutPool | 0.0369 | 0.0448 | 0.1196 |

## K-Means stability  (10 seeds)
| NMI μ | NMI σ | ARI μ | ARI σ | SIL μ | SIL σ |
|---|---|---|---|---|---|
| 0.1635 | 0.0209 | 0.0796 | 0.0204 | 0.0461 | 0.0065 |

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
