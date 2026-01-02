import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataHandler:
    def __init__(self, args):
        self.path = args.dataset_path
        self.batch_size = args.batch
        self.radius = args.subgraph_radius
        
        print(f"Loading data from {self.path}...")
        
        # 1. Read CSV
        # Assumes columns are [Gene, Drug]
        df = pd.read_csv(self.path)
        df.dropna(inplace=True)
        
        # 2. Label Encoding (String -> Integer)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Column 0 is Gene (User), Column 1 is Drug (Item)
        u_ids = self.user_encoder.fit_transform(df.iloc[:, 0].astype(str))
        i_ids = self.item_encoder.fit_transform(df.iloc[:, 1].astype(str))
        
        # 3. Handle Signs
        # If dataset has no 3rd column, default sign to 1 (Positive)
        if len(df.columns) >= 3:
            s_values = df.iloc[:, 2].values
            # Logic: >0 is positive (1), <=0 is negative (0)
            signs = np.where(s_values > 0, 1, 0)
        else:
            print("No sign column detected. Defaulting all edges to Positive (1).")
            signs = np.ones(len(df), dtype=int)
            
        # Combine into edge array: [u, i, sign]
        all_edges = np.column_stack((u_ids, i_ids, signs))
        
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)
        
        # 4. Train/Test Split
        self.train_edges, self.test_edges = train_test_split(
            all_edges, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Stats: {self.num_users} Genes, {self.num_items} Drugs.")
        print(f"Train samples: {len(self.train_edges)}, Test samples: {len(self.test_edges)}")
        
        # 5. Build NetworkX Graph for Subgraph Extraction
        # Note: We shift item indices by num_users to put them in the same graph space
        self.G = nx.Graph()
        for u, i, s in self.train_edges:
            self.G.add_edge(u, i + self.num_users, sign=s)

    def get_global_graph_tensor(self):
        """
        Builds the global edge index for the whole graph (PyG format).
        Returns: edge_index, edge_attr
        """
        src = []
        dst = []
        signs = []
        
        for u, i, s in self.train_edges:
            # Forward: u -> i (shifted)
            src.append(u)
            dst.append(i + self.num_users)
            signs.append(s)
            
            # Backward: i (shifted) -> u
            src.append(i + self.num_users)
            dst.append(u)
            signs.append(s)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(signs, dtype=torch.long)
        return edge_index, edge_attr

    def extract_subgraph(self, u, i):
        """
        Extracts k-hop subgraph and computes Two-Anchor Distance Labels.
        """
        real_i = i + self.num_users
        
        # 1. Get k-hop neighbors for both u and i
        # Using a try-except block to handle potentially isolated nodes in sparse data
        try:
            nodes_u = nx.single_source_shortest_path_length(self.G, u, cutoff=self.radius)
        except:
            nodes_u = {u: 0}
            
        try:
            nodes_v = nx.single_source_shortest_path_length(self.G, real_i, cutoff=self.radius)
        except:
            nodes_v = {real_i: 0}

        # Union of neighbors
        sub_nodes = set(nodes_u.keys()) | set(nodes_v.keys())
        sub_nodes = list(sub_nodes)
        # Local mapping: Global ID -> Local Index (0 to len-1)
        mapping = {node: idx for idx, node in enumerate(sub_nodes)}
        
        # 2. Calculate Distance Labels (Eq. 6 in paper)
        dist_labels = []
        for node in sub_nodes:
            du = nodes_u.get(node, 999) # 999 = infinity
            dv = nodes_v.get(node, 999)
            
            # Heuristic for label encoding: 1 + min(du, dv) + (du + dv)
            label = 1 + min(du, dv) + (du + dv)
            label = min(label, 10) # Cap at max label
            dist_labels.append(label)
            
        # 3. Build Subgraph Edges
        sub_src, sub_dst, sub_signs = [], [], []
        sub_G = self.G.subgraph(sub_nodes)
        
        for src_node, dst_node in sub_G.edges():
            if src_node in mapping and dst_node in mapping:
                s_idx = mapping[src_node]
                d_idx = mapping[dst_node]
                
                # Get sign, default to 1 if missing
                sign = sub_G[src_node][dst_node].get('sign', 1)
                
                # Add undirected edges
                sub_src.extend([s_idx, d_idx])
                sub_dst.extend([d_idx, s_idx])
                sub_signs.extend([sign, sign])

        # Safety check for empty subgraphs
        if len(sub_src) == 0:
            sub_src = [0, 0]
            sub_dst = [0, 0]
            sub_signs = [1, 1]

        return {
            'x_idx': torch.tensor(sub_nodes, dtype=torch.long), # Global indices used for lookup
            'edge_index': torch.tensor([sub_src, sub_dst], dtype=torch.long),
            'edge_sign': torch.tensor(sub_signs, dtype=torch.long),
            'dist_labels': torch.tensor(dist_labels, dtype=torch.long)
        }

    def get_loader(self, mode='train'):
        edges = self.train_edges if mode == 'train' else self.test_edges
        return data.DataLoader(
            dataset=CSIHDataset(edges),
            batch_size=self.batch_size,
            shuffle=(mode=='train'),
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        # Packs a list of subgraphs into a PyG Batch object
        from torch_geometric.data import Data, Batch
        data_list = []
        for (u, i, label) in batch:
            sub_info = self.extract_subgraph(u, i)
            d = Data(
                x_idx=sub_info['x_idx'], 
                edge_index=sub_info['edge_index'], 
                edge_attr=sub_info['edge_sign'],
                dist_label=sub_info['dist_labels'],
                y=torch.tensor([float(label)]) # Label is usually the sign (1 or 0)
            )
            d.root_u = int(u)
            d.root_i = int(i)
            data_list.append(d)
        return Batch.from_data_list(data_list)

class CSIHDataset(data.Dataset):
    def __init__(self, edges):
        self.edges = edges
    def __len__(self):
        return len(self.edges)
    def __getitem__(self, idx):
        return self.edges[idx]