# Microbiome clustering — full evaluation report

## Best method by NMI: **KMeans (Jaccard PCoA)**  NMI=0.3669

## All methods — clustering metrics

| Method | NMI | ARI | Silhouette_cosine |
| --- | --- | --- | --- |
| KMeans (Jaccard PCoA) | 0.3669 | 0.1965 | 0.1200 |
| KMeans (MDS) | 0.1971 | 0.0842 | 0.0648 |
| KMeans (PCA) | 0.2740 | 0.1210 | 0.0967 |
| KMeans (PHATE) | 0.2580 | 0.0797 | 0.0926 |
| KMeans (PaCMAP) | 0.2644 | 0.1315 | 0.1416 |
| KMeans (SONG) | 0.3093 | 0.1602 | 0.1493 |
| KMeans (UMAP) | 0.2472 | 0.0995 | 0.1015 |
| KMeans (t-SNE) | 0.3081 | 0.1766 | 0.1358 |
| KNN+DMoN | 0.3089 | 0.1558 | 0.1194 |
| KNN+MinCutPool | 0.2335 | 0.1420 | 0.1061 |

## K-Means stability  (10 seeds)
| NMI μ | NMI σ | ARI μ | ARI σ | SIL μ | SIL σ |
|---|---|---|---|---|---|
| 0.2698 | 0.0382 | 0.1212 | 0.0321 | 0.1414 | 0.0080 |

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
