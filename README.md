# Data-Driven Metagenomics: Topology-Aware DR & GNN Clustering

**Comparative evaluation of 8 dimensionality reduction methods and graph neural network clustering for high-dimensional, sparse, compositional metagenomic data.**

## 🎯 Project Overview
Systematic benchmarking of dimensionality reduction (DR) techniques and GNN-based clustering specifically optimized for metagenomics challenges:
- **Extreme sparsity** (90-99% zeros)
- **Compositionality** (sequencing depth constraints) 
- **Phylogenetic structure preservation**

## 📊 Phase 1: DR Benchmarking (Completed)
**Datasets**: Human (3.6K samples), Soil (423 samples), Marine (1.2K samples)
**Methods**: PCA, UniFrac PCoA, MDS, t-SNE, UMAP, PaCMAP, PHATE, SONG
**Metrics**: Trustworthiness, Continuity, UniFrac/Bray-Curtis/Aitchison correlations

**Key Findings**: No universal best method - UniFrac PCoA excels phylogenetically, PaCMAP/SONG balance local/global structure

## 🔬 Phase 2: GNN Clustering (In Progress)
**Graph Construction**: k-NN from DR embeddings, hard/soft edges
**Architectures**: DMoN, MinCutPool, GCN+GAT + k-means
**Evaluation**: NMI, ARI, Silhouette, modularity

## 🛠️ Tech Stack
- **Preprocessing**: nzCLR transformation, UniFrac distances
- **DR**: scikit-learn, umap-learn, pacmap, phate, song
- **GNN**: PyTorch Geometric, DGL
- **Visualization**: matplotlib, seaborn, plotly

## 📈 Current Results
- 24 visualizations across 3 datasets × 8 methods
- Comprehensive metric tables (Trustworthiness: SONG 0.927, UniFrac: PCoA 0.999)
- Preliminary rankings by structure preservation

## 🚀 Goals
1. Evidence-based DR method recommendations for metagenomics
2. Validate GNN clustering performance vs traditional methods
3. Develop unified preprocessing-evaluation pipeline

**Fills critical gaps**: Systematic DR comparison + GNN validation for microbial community analysis
