# Data-Driven Methods For Comparative Metagenomics

> **Final Year Project** — Department of Computer Engineering, University of Peradeniya

[![GitHub Pages](https://img.shields.io/badge/Project%20Page-Live-blue?style=flat-square)](https://cepdnaclk.github.io/e20-4yp-metagenomics/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/cepdnaclk/e20-4yp-metagenomics)

---

## 👥 Team

| Reg. No. | Name | Email |
|----------|------|-------|
| E/20/158 | Jananga T.G.C. | [e20158@eng.pdn.ac.lk](mailto:e20158@eng.pdn.ac.lk) |
| E/20/300 | Prasadinie H.A.M.T. | [e20300@eng.pdn.ac.lk](mailto:e20300@eng.pdn.ac.lk) |
| E/20/244 | Malshan P.G.P. | [e20244@eng.pdn.ac.lk](mailto:e20244@eng.pdn.ac.lk) |

## 🎓 Supervisors

| Name | Affiliation | Email |
|------|-------------|-------|
| Dr. Damayanthi Herath | University of Peradeniya | [damayanthiherath@eng.pdn.ac.lk](mailto:damayanthiherath@eng.pdn.ac.lk) |
| Dr. Rajith Vidanaarachchi | University of Melbourne | [rajith.v@unimelb.edu.au](mailto:rajith.v@unimelb.edu.au) |
| Dr. Vijini Mallawaarachchi | Flinders University | [vijini.mallawaarachchi@flinders.edu.au](mailto:vijini.mallawaarachchi@flinders.edu.au) |

---

## 🎯 Abstract

Metagenomic studies generate high-dimensional, sparse, and compositional datasets that challenge traditional analytical methods. This project systematically benchmarks **8 dimensionality reduction (DR) methods** across **3 diverse real-world metagenomic datasets** and validates **Graph Neural Network (GNN)** architectures (DMoN, MinCutPool) for unsupervised microbial community clustering.

**Key challenges addressed:**
- ⚡ Extreme sparsity (90–99% zeros)
- 🔗 Compositionality constraints imposed by sequencing depth
- 🌿 Preservation of phylogenetic structure in lower-dimensional space

---

## 📊 Research Phases

### Phase 1 — DR Benchmarking ✅ Completed

| | |
|---|---|
| **Datasets** | Human Gut (3.6K samples, 18 disease classes), Ocean/Tara Oceans (139 samples), Potato Soil (PotatoSCMP, 885 bacterial genus features) |
| **Preprocessing** | Non-zero CLR (nzCLR) transformation on raw counts; Jaccard distances on raw for PCoA |
| **DR Methods** | PCA, Jaccard PCoA, MDS, t-SNE, UMAP, PaCMAP, PHATE, SONG |
| **Evaluation Metrics** | Trustworthiness, Continuity, KNN Preservation, Normalized Stress, Bray-Curtis/Aitchison/Jaccard correlations |

**Key Findings:**
- No single DR method universally outperforms all others across every dataset and metric
- UniFrac/Jaccard PCoA excels at preserving distance structure (Bray-Curtis correlation up to 0.999)
- PaCMAP and SONG achieve the best balance between local and global structure preservation

### Phase 2 — GNN Clustering 🔬 In Progress

| | |
|---|---|
| **Graph Construction** | KNN cosine graph (mutual, k=10) built on nzCLR-transformed features; GEXF export |
| **GNN Architectures** | DMoN (Differentiable Modularity), MinCutPool |
| **Traditional Baseline** | K-Means on 8 DR embeddings (PCA, PCoA, MDS, t-SNE, UMAP, PaCMAP, PHATE, SONG) |
| **Evaluation Metrics** | NMI, ARI, Silhouette (cosine), Stability across runs |
| **Datasets** | Human Gut, Tara Oceans, Potato Soil |

---

## 🛠️ Tech Stack

| Component | Tools / Libraries |
|-----------|-------------------|
| **Preprocessing** | numpy, pandas, scipy — nzCLR, Bray-Curtis, Aitchison, Jaccard |
| **Dimensionality Reduction** | scikit-learn, umap-learn, pacmap, phate, song-vis, scikit-bio |
| **GNN Frameworks** | PyTorch, PyTorch Geometric — DMoNPooling, dense_mincut_pool |
| **Graph Export** | NetworkX (GEXF), numpy (COO format) |
| **Visualization** | Matplotlib, Seaborn, Plotly |

---

## 📁 Repository Structure

```
e20-4yp-metagenomics/
├── code/
│   ├── Phase 01/               # Phase 1: DR benchmarking scripts
│   │   ├── Human/              # human.py — gut microbiome analysis
│   │   ├── Ocean/              # ocean.py — Tara Oceans analysis
│   │   └── Potato/             # Potato.py — soil microbiome analysis
│   └── Phase2/                 # Phase 2: GNN clustering pipeline
│       ├── Human Metagenomics/ # ✅ Complete pipeline
│       ├── Ocean/              # ✅ Complete pipeline
│       └── Potato/             # ✅ Complete pipeline
│           ├── microbiome_preprocessing_full.py
│           ├── microbiome_gnn_clustering.py
│           └── microbiome_clustering_evaluation_full.py
└── docs/                       # GitHub Pages website
    ├── index.html
    ├── style.css
    └── images/                 # Team, conference & poster images
```

---

## 🏆 Achievements & Recognition

- 🏆 **ICIPROB 2025** — Paper presented at the International Conference on Image Processing & Robotics
- 🏛️ **PULSE Exposition** — Poster at Peradeniya University International Research Sessions & Exposition
- 🧬 **Bio Fusion** — Collaborative research initiative in biological data science

---

## 📈 Key Results (Phase 1)

| Metric | Best Method | Score |
|--------|------------|-------|
| Trustworthiness | SONG | 0.927 |
| Bray-Curtis Correlation | PCoA | 0.999 |
| KNN Preservation | PaCMAP | — |

---

## 📄 Publications & Reports

1. Semester 7 Report — Phase 1 DR Benchmarking
2. Final Year Project Presentation — [Download PDF](./docs/images/Final%20Presentation%20FYP.pdf)
3. Research Poster — PULSE Exposition
4. ICIPROB 2025 — Conference Paper
5. Semester 8 Report — Phase 2 GNN Clustering *(coming soon)*

---

## 🔗 Links

- 🌐 [Project Website](https://cepdnaclk.github.io/e20-4yp-metagenomics/)
- 💻 [GitHub Repository](https://github.com/cepdnaclk/e20-4yp-metagenomics)
- 🏫 [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- 🎓 [University of Peradeniya](https://eng.pdn.ac.lk/)
