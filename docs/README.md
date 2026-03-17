---
layout: home
permalink: index.html

repository-name: e20-4yp-metagenomics
title: Data Driven Methods For Comparative Metagenomics
---

# Data Driven Methods For Comparative Metagenomics

#### Team

- E/20/158, Jananga T.G.C., [email](mailto:e20158@eng.pdn.ac.lk)
- E/20/300, Prasadinie H.A.M.T, [email](mailto:e20300@eng.pdn.ac.lk)
- E/20/244, Malshan P.G.P, [email](mailto:e20244@eng.pdn.ac.lk)

#### Supervisors

- Dr. Damayanthi Herath, [email](mailto:damayanthiherath@eng.pdn.ac.lk)
- Dr. Rajith Vidanaarachchi, [email](mailto:rajith.v@unimelb.edu.au)
- Dr. Vijini Mallawaarachchi, [email](mailto:vijini.mallawaarachchi@flinders.edu.au)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract

Metagenomic studies generate high-dimensional, sparse, and compositional datasets that challenge traditional analytical methods. This project systematically benchmarks **8 dimensionality reduction (DR) methods** across **3 diverse metagenomic datasets** (Human — 3.6K samples, Soil — 423 samples, Marine — 1.2K samples) and validates **Graph Neural Network (GNN)** architectures for unsupervised microbial community clustering. The research addresses critical challenges including extreme sparsity (90–99% zeros), compositionality constraints imposed by sequencing depth, and the need to preserve phylogenetic structure in lower-dimensional representations.

## Related works

Dimensionality reduction has been widely applied in genomics and metagenomics, with methods such as PCA, PCoA, t-SNE, and UMAP being standard in microbiome analysis pipelines. However, systematic benchmarking across diverse ecosystem types and distance metrics remains limited. Recent advances in graph neural networks (DMoN, MinCutPool) have shown promise for unsupervised community detection but have not been extensively validated on metagenomic data. This project fills critical gaps by providing the first comprehensive DR comparison combined with GNN-based clustering validation for microbial community analysis.

## Methodology

The research follows a two-phase approach:

**Phase 1 — DR Benchmarking (Completed):**
- **Preprocessing:** nzCLR transformation and UniFrac distance computation to handle compositionality
- **Methods Evaluated:** PCA, UniFrac PCoA, MDS, t-SNE, UMAP, PaCMAP, PHATE, SONG
- **Evaluation Metrics:** Trustworthiness, Continuity, UniFrac / Bray-Curtis / Aitchison distance correlations

**Phase 2 — GNN Clustering (In Progress):**
- **Graph Construction:** k-NN graphs from DR embeddings with hard and soft edges
- **Architectures:** DMoN, MinCutPool, GCN + GAT combined with k-means
- **Evaluation Metrics:** NMI, ARI, Silhouette score, Modularity

## Experiment Setup and Implementation

| Component | Tools / Libraries |
|---|---|
| **Preprocessing** | nzCLR transformation, UniFrac distances, Bray-Curtis, Aitchison |
| **Dimensionality Reduction** | scikit-learn, umap-learn, PaCMAP, PHATE, SONG |
| **GNN Frameworks** | PyTorch Geometric, DGL |
| **Visualization** | Matplotlib, Seaborn, Plotly |

- Experiments conducted across 3 diverse metagenomic datasets
- 24 visualizations generated (3 datasets × 8 methods)

## Results and Analysis

### Key Findings

| Metric | Best Method | Score |
|---|---|---|
| Trustworthiness | SONG | 0.927 |
| UniFrac Correlation | PCoA | 0.999 |

- **No universal best method** — performance is dataset and metric dependent
- **UniFrac PCoA** excels at preserving phylogenetic distance structure (correlation up to 0.999)
- **PaCMAP and SONG** achieve the best balance between local and global structure preservation
- Preliminary rankings established by structure preservation capabilities across all three datasets

## Conclusion

This project provides evidence-based DR method recommendations for metagenomics researchers and validates GNN clustering performance against traditional approaches. The unified preprocessing-to-evaluation pipeline developed through this research serves as a reusable framework for future metagenomic studies. Key takeaway: method selection should be guided by the specific analytical goals and dataset characteristics rather than defaulting to a single approach.

## Publications

1. [Semester 7 report](./)
2. [Semester 7 slides](./)
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->

## Links

- [Project Repository](https://github.com/cepdnaclk/e20-4yp-metagenomics.git)
- [Project Page](https://cepdnaclk.github.io/e20-4yp-metagenomics/)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)
