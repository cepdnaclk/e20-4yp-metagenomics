---
layout: home
permalink: index.html

repository-name: e20-4yp-metagenomics
title: Data Driven Methods For Comparative Metagenomics
---

# Data Driven Methods For Comparative Metagenomics

#### Team

- E/20/158, Jananga T.G.C., [email](mailto:e20158@eng.pdn.ac.lk)
- E/20/300, Prasadinie H.A.M.T., [email](mailto:e20300@eng.pdn.ac.lk)
- E/20/244, Malshan P.G.P., [email](mailto:e20244@eng.pdn.ac.lk)

#### Supervisors

- Dr. Damayanthi Herath — University of Peradeniya, [email](mailto:damayanthiherath@eng.pdn.ac.lk)
- Dr. Rajith Vidanaarachchi — University of Melbourne, [email](mailto:rajith.v@unimelb.edu.au)
- Dr. Vijini Mallawaarachchi — Flinders University, [email](mailto:vijini.mallawaarachchi@flinders.edu.au)

#### Table of Contents

1. [Abstract](#abstract)
2. [Related Works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Achievements](#achievements)
7. [Conclusion](#conclusion)
8. [Publications](#publications)
9. [Links](#links)

---

## Abstract

Metagenomic studies generate high-dimensional, sparse, and compositional datasets that challenge traditional analytical methods. This project systematically benchmarks **8 dimensionality reduction (DR) methods** across **3 diverse real-world metagenomic datasets** — Human Gut (3.6K samples, 18 disease classes), Tara Oceans (139 samples, ocean zone labels), and Potato Soil (PotatoSCMP dataset, 885 bacterial genus features) — and validates **Graph Neural Network (GNN)** architectures for unsupervised microbial community clustering. Key challenges include extreme sparsity (90–99% zeros), compositionality constraints, and the need to preserve phylogenetic structure in lower-dimensional representations.

## Related Works

Dimensionality reduction has been widely applied in metagenomics with methods such as PCA, PCoA, t-SNE, and UMAP being standard in microbiome analysis pipelines. However, systematic benchmarking across diverse ecosystem types and across multiple distance metrics remains limited. Recent advances in graph neural networks (DMoN, MinCutPool) have shown promise for unsupervised community detection but have not been extensively validated on real-world metagenomic data. This project fills critical gaps by providing a comprehensive DR comparison combined with GNN-based clustering validation across three distinct biological contexts.

## Methodology

The research follows a two-phase approach:

**Phase 1 — DR Benchmarking (Completed):**
- **Preprocessing:** Non-zero CLR (nzCLR) transformation on raw counts; Jaccard distance matrix for PCoA
- **Methods Evaluated:** PCA, Jaccard PCoA, MDS, t-SNE, UMAP, PaCMAP, PHATE, SONG
- **Evaluation Metrics:** Trustworthiness, Continuity, KNN Preservation, Normalized Stress, Bray-Curtis / Aitchison / Jaccard correlations
- **Datasets:** Human Gut, Tara Oceans, Potato Soil

**Phase 2 — GNN Clustering (In Progress):**
- **Graph Construction:** KNN cosine mutual graph (k=10) on nzCLR features; exported as GEXF
- **Architectures:** DMoN (Differentiable Modularity) and MinCutPool
- **Traditional Baseline:** K-Means clustering on 8 DR embeddings
- **Evaluation Metrics:** NMI, ARI, Silhouette (cosine), Stability analysis
- **Datasets:** Human Gut, Tara Oceans, Potato Soil

## Experiment Setup and Implementation

| Component | Tools / Libraries |
|---|---|
| **Preprocessing** | numpy, pandas, scipy — nzCLR, Bray-Curtis, Aitchison, Jaccard |
| **Dimensionality Reduction** | scikit-learn, umap-learn, PaCMAP, PHATE, SONG, scikit-bio |
| **GNN Frameworks** | PyTorch, PyTorch Geometric — DMoNPooling, dense_mincut_pool |
| **Graph Export** | NetworkX (GEXF), numpy (COO format) |
| **Visualization** | Matplotlib, Seaborn, Plotly |

- Pipeline implemented across **3 datasets × 2 phases**
- **24+ visualizations** generated (3 datasets × 8 DR methods)
- Full evaluation suite: metric tables, cluster distribution plots, heatmaps, dashboard figures

## Results and Analysis

### Phase 1 Key Findings

| Metric | Best Method | Score |
|---|---|---|
| Trustworthiness | SONG | 0.927 |
| Bray-Curtis Correlation | PCoA | 0.999 |

- **No universal best method** — performance is dataset and metric dependent
- **Jaccard PCoA** excels at preserving distance structure (up to 0.999 correlation)
- **PaCMAP and SONG** achieve the best balance between local and global structure preservation
- Preliminary DR rankings established across all three datasets

### Phase 2 Key Findings (Preliminary)

- GNN-based clustering (KNN+DMoN) captures microbial groupings beyond what flat K-Means on DR embeddings achieves
- Silhouette scores vary across datasets reflecting genuine biological structure differences
- Human Gut: 18 disease-class overlap is inherently difficult; GNN improves ARI over K-Means
- Tara Oceans and Potato Soil: stronger single-factor labels yield cleaner cluster separation

## Achievements

- 🏆 **ICIPROB 2025** — Paper presented at the International Conference on Image Processing & Robotics
- 🏛️ **PULSE Exposition** — Poster at Peradeniya University International Research Sessions & Exposition
- 🧬 **Bio Fusion** — Collaborative research initiative in biological data science

## Conclusion

This project provides evidence-based DR method recommendations for metagenomics researchers and validates GNN clustering performance against traditional approaches across three real-world datasets. The unified preprocessing-to-evaluation pipeline developed through this research serves as a reusable framework for future metagenomic studies. Key takeaway: **method selection should be guided by specific analytical goals and dataset characteristics** rather than defaulting to a single approach. GNN-based clustering offers measurable improvements when biological signals are complex and graph structure is informative.

## Publications

1. [Semester 7 Report](./data/)
2. [Final Year Project Presentation](./images/Final%20Presentation%20FYP.pdf)
3. ICIPROB 2025 — Conference Paper
4. Research Poster — PULSE Exposition
<!-- 5. [Semester 8 Report](./) — coming soon -->

## Links

- [Project Repository](https://github.com/cepdnaclk/e20-4yp-metagenomics)
- [Project Page](https://cepdnaclk.github.io/e20-4yp-metagenomics/)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)
