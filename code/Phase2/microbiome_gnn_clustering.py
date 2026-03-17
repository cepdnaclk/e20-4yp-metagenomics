import os
import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, DMoNPooling, dense_mincut_pool
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score, silhouette_score, 
    homogeneity_score, completeness_score, v_measure_score, 
    calinski_harabasz_score, davies_bouldin_score
)

def compute_metrics(labels, X_eval, y_true):
    def _s(fn, *a, **kw):
        try: return float(fn(*a, **kw))
        except: return None

    # Handle cases where all samples are mapped to one cluster
    n_unique = len(np.unique(labels))
    ss_cos = _s(silhouette_score, X_eval, labels, metric="cosine", random_state=42) if n_unique > 1 else None
    ch_score = _s(calinski_harabasz_score, X_eval, labels) if n_unique > 1 else None
    db_score = _s(davies_bouldin_score, X_eval, labels) if n_unique > 1 else None

    return {
        "NMI": _s(normalized_mutual_info_score, y_true, labels),
        "ARI": _s(adjusted_rand_score, y_true, labels),
        "Silhouette_cosine": ss_cos,
        "Homogeneity": _s(homogeneity_score, y_true, labels),
        "Completeness": _s(completeness_score, y_true, labels),
        "V_measure": _s(v_measure_score, y_true, labels),
        "Calinski_Harabasz": ch_score,
        "Davies_Bouldin": db_score,
    }

class DMoNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # dropout is 0 default
        self.pool = DMoNPooling([hidden_channels, hidden_channels], num_clusters)

    def forward(self, x, edge_index, edge_weight):
        x = torch.relu(self.conv1(x, edge_index, edge_weight))
        x = torch.relu(self.conv2(x, edge_index, edge_weight))
        x_dense = x.unsqueeze(0)
        # to_dense_adj returns (batch, N, N)
        adj_dense = to_dense_adj(edge_index, edge_attr=edge_weight).unsqueeze(0)
        _, _, _, _, pool_loss, ortho_loss = self.pool(x_dense, adj_dense)
        return pool_loss + ortho_loss
        
    def get_clusters(self, x, edge_index, edge_weight):
        with torch.no_grad():
            x = torch.relu(self.conv1(x, edge_index, edge_weight))
            x = torch.relu(self.conv2(x, edge_index, edge_weight))
            x_dense = x.unsqueeze(0)
            adj_dense = to_dense_adj(edge_index, edge_attr=edge_weight).unsqueeze(0)
            s, _, _, _, _, _ = self.pool(x_dense, adj_dense)
            return s.squeeze(0).argmax(dim=-1).cpu().numpy()

class MinCutNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool_lin = torch.nn.Linear(hidden_channels, num_clusters)

    def forward(self, x, edge_index, edge_weight):
        x = torch.relu(self.conv1(x, edge_index, edge_weight))
        x = torch.relu(self.conv2(x, edge_index, edge_weight))
        s = self.pool_lin(x)
        x_dense = x.unsqueeze(0)
        adj_dense = to_dense_adj(edge_index, edge_attr=edge_weight).unsqueeze(0)
        s_dense = s.unsqueeze(0)
        # returns x, adj, mincut_loss, ortho_loss (4 elements in dense mincut pool)
        out = dense_mincut_pool(x_dense, adj_dense, s_dense)
        return out[2] + out[3]

    def get_clusters(self, x, edge_index, edge_weight):
        with torch.no_grad():
            x = torch.relu(self.conv1(x, edge_index, edge_weight))
            x = torch.relu(self.conv2(x, edge_index, edge_weight))
            s = self.pool_lin(x).unsqueeze(0) # (1, N, C)
            s = torch.softmax(s, dim=-1)
            return s.squeeze(0).argmax(dim=-1).cpu().numpy()

def train_unsupervised(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = model(data.x, data.edge_index, data.edge_attr)
        loss.backward()
        optimizer.step()
    model.eval()
    return model.get_clusters(data.x, data.edge_index, data.edge_attr)

def main():
    print("="*60)
    print("  GNN UNSUPERVISED CLUSTERING (DMoN & MinCutPool)")
    print("="*60)

    out_dir = "preprocessed"
    if not os.path.exists(out_dir):
        print("Preprocessed directory not found. Please run preprocessing first.")
        return

    # Load baseline datasets for Evaluation
    X_clr_df = pd.read_csv(f"{out_dir}/X_clr.csv", index_col=0)
    X_clr = X_clr_df.values.astype(np.float32)
    y_true = np.load(f"{out_dir}/y_true.npy")
    num_clusters = len(np.unique(y_true))
    
    # ── 1. KNN Graph Data
    X_pca_dmon_df = pd.read_csv(f"{out_dir}/X_pca_dmon.csv", index_col=0)
    X_pca_dmon = torch.tensor(X_pca_dmon_df.values.astype(np.float32))
    edge_index_knn = torch.tensor(np.load(f"{out_dir}/edge_index_knn.npy"), dtype=torch.long)
    edge_weight_knn = torch.tensor(np.load(f"{out_dir}/edge_weights_knn.npy"), dtype=torch.float32)
    
    knn_data = Data(x=X_pca_dmon, edge_index=edge_index_knn, edge_attr=edge_weight_knn)

    # ── 2. Bipartite Graph Data
    node_feat_bip = torch.tensor(np.load(f"{out_dir}/node_feat_bip.npy"), dtype=torch.float32)
    edge_index_bip = torch.tensor(np.load(f"{out_dir}/edge_index_bip.npy"), dtype=torch.long)
    edge_weight_bip = torch.tensor(np.load(f"{out_dir}/edge_weights_bip.npy"), dtype=torch.float32)
    n_sample_nodes = np.load(f"{out_dir}/bip_n_sample_nodes.npy")[0]

    bip_data = Data(x=node_feat_bip, edge_index=edge_index_bip, edge_attr=edge_weight_bip)

    experiments = []

    # Hidden Channels
    hidden_dim = 64

    # Exp 1: KNN + DMoN
    print("\nRunning KNN + DMoN...")
    model = DMoNNet(in_channels=knn_data.num_features, hidden_channels=hidden_dim, num_clusters=num_clusters)
    labels_knn_dmon = train_unsupervised(model, knn_data, epochs=80)
    metrics_knn_dmon = compute_metrics(labels_knn_dmon, X_clr, y_true)
    experiments.append(("KNN+DMoN", metrics_knn_dmon))

    # Exp 2: KNN + MinCutPool
    print("Running KNN + MinCutPool...")
    model = MinCutNet(in_channels=knn_data.num_features, hidden_channels=hidden_dim, num_clusters=num_clusters)
    labels_knn_mincut = train_unsupervised(model, knn_data, epochs=80)
    metrics_knn_mincut = compute_metrics(labels_knn_mincut, X_clr, y_true)
    experiments.append(("KNN+MinCutPool", metrics_knn_mincut))

    # Exp 3: Bipartite + DMoN
    print("Running Bipartite + DMoN...")
    model = DMoNNet(in_channels=bip_data.num_features, hidden_channels=hidden_dim, num_clusters=num_clusters)
    labels_bip_all = train_unsupervised(model, bip_data, epochs=80)
    labels_bip_samples = labels_bip_all[:n_sample_nodes] # Only evaluate the sample nodes
    metrics_bip_dmon = compute_metrics(labels_bip_samples, X_clr, y_true)
    experiments.append(("Bipartite+DMoN", metrics_bip_dmon))

    # Exp 4: Bipartite + MinCutPool
    print("Running Bipartite + MinCutPool...")
    model = MinCutNet(in_channels=bip_data.num_features, hidden_channels=hidden_dim, num_clusters=num_clusters)
    labels_bip_all = train_unsupervised(model, bip_data, epochs=80)
    labels_bip_samples = labels_bip_all[:n_sample_nodes]
    metrics_bip_mincut = compute_metrics(labels_bip_samples, X_clr, y_true)
    experiments.append(("Bipartite+MinCutPool", metrics_bip_mincut))

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    results_dict = {}
    for name, metrics in experiments:
        results_dict[name] = metrics
        print(f"[{name}] NMI: {metrics['NMI']:.4f}  ARI: {metrics['ARI']:.4f}  SIL: {metrics['Silhouette_cosine']}")
        
    json_path = f"{out_dir}/gnn_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"\nSaved metrics to {json_path}")

if __name__ == "__main__":
    main()
