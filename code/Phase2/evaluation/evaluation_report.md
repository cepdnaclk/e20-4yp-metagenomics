# Microbiome clustering — full evaluation report

## Best method by NMI: **KMeans (PaCMAP)**  NMI=0.2905

## All methods — clustering metrics

| Method | NMI | ARI | Silhouette_cosine |
| --- | --- | --- | --- |
| KMeans (CLR) | 0.2778 | 0.0591 | 0.0687 |
| KMeans (AitchisonPCA) | 0.2847 | 0.0592 | 0.0648 |
| KMeans (Jaccard PCoA) | 0.2408 | 0.0395 | -0.0062 |
| KMeans (MDS) | 0.2005 | 0.0367 | -0.0209 |
| KMeans (PCA) | 0.1928 | 0.0250 | -0.0149 |
| KMeans (PHATE) | 0.2579 | 0.0179 | 0.0168 |
| KMeans (PaCMAP) | 0.2905 | 0.0572 | 0.0733 |
| KMeans (SONG) | 0.2670 | 0.0566 | 0.0398 |
| KMeans (UMAP) | 0.2794 | 0.0576 | 0.0732 |
| KMeans (t-SNE) | 0.2661 | 0.0558 | 0.0539 |
| KNN+DMoN | 0.2735 | 0.0602 | 0.0373 |
| KNN+MinCutPool | 0.2232 | -0.0490 | 0.2139 |

## K-Means stability  (10 seeds)
| NMI μ | NMI σ | ARI μ | ARI σ | SIL μ | SIL σ |
|---|---|---|---|---|---|
| 0.2820 | 0.0052 | 0.0560 | 0.0041 | 0.0562 | 0.0096 |

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
