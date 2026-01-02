import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax

class SignEnhancedMessagePassing(MessagePassing):
    """
    Global Encoder: Propagates messages respecting edge signs.
    """
    def __init__(self, dim):
        super().__init__(aggr='add')
        self.W = nn.Linear(dim, dim)
        self.sign_emb = nn.Embedding(2, 1) # 0: Negative, 1: Positive

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Modulate weight by sign embedding
        sign_w = self.sign_emb(edge_attr).view(-1, 1)
        return sign_w * self.W(x_j)
    
    def update(self, aggr_out, x):
        return x + F.relu(aggr_out)

class CSIH(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(CSIH, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_users = num_users
        self.n_items = num_items
        self.dim = args.latdim
        self.K = args.n_intents
        
        # --- A. Embeddings ---
        self.user_emb = nn.Embedding(num_users, self.dim)
        self.item_emb = nn.Embedding(num_items, self.dim)
        # Distance Label Embeddings (0-10)
        self.dist_emb = nn.Embedding(args.max_dist_label + 1, self.dim)
        
        # --- B. Intent Prototypes ---
        # Learnable centers for intents
        self.c_u = nn.Parameter(torch.randn(self.K, self.dim))
        self.c_v = nn.Parameter(torch.randn(self.K, self.dim))
        
        # --- C. Global Module ---
        self.global_gnn = SignEnhancedMessagePassing(self.dim)
        # MLP to project embeddings to intent space
        self.intent_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim), 
            nn.Tanh(), 
            nn.Linear(self.dim, self.K)
        )
        
        # --- D. Local Module ---
        # Project concatenation of node feature + distance feature
        self.local_proj = nn.Linear(self.dim * 2, self.dim) 
        # Local GNN (Using GCNConv for stability)
        from torch_geometric.nn import GCNConv
        self.local_gnn = GCNConv(self.dim, self.dim) 
        self.pool_proj = nn.Linear(self.dim, self.dim)

        # --- E. Prediction ---
        # Fusing: Subgraph + Global U + Global I + Intent Interaction
        self.final_mlp = nn.Sequential(
            nn.Linear(self.dim * 4, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
        
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def get_intent_distribution(self, emb, prototypes):
        """
        Calculates P(intent | embedding) and the weighted intent representation.
        """
        logits = self.intent_mlp(emb)
        probs = F.softmax(logits, dim=-1)
        # Weighted sum of prototypes
        r = torch.matmul(probs, prototypes)
        return r, probs

    def forward(self, global_edge_index, global_edge_attr, batch_data):
        """
        global_edge_index: Full graph edges
        batch_data: Batch of local subgraphs
        """
        
        # 1. Prepare Global Features
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        
        # 2. Global GNN Pass
        global_repr = self.global_gnn(all_emb, global_edge_index, global_edge_attr)
        
        # 3. Lookup specific User/Item embeddings for the batch
        u_batch_idx = torch.tensor(batch_data.root_u).to(self.device)
        i_batch_idx = torch.tensor(batch_data.root_i).to(self.device) + self.n_users
        
        gu = global_repr[u_batch_idx]
        gi = global_repr[i_batch_idx]
        
        # 4. Intent Disentanglement
        ru, pu = self.get_intent_distribution(gu, self.c_u)
        rv, pv = self.get_intent_distribution(gi, self.c_v)
        
        # 5. Local Subgraph Processing
        # Lookup global features for nodes involved in the subgraphs
        node_feat = global_repr[batch_data.x_idx] 
        dist_feat = self.dist_emb(batch_data.dist_label)
        
        # Fuse and Project
        local_h = torch.cat([node_feat, dist_feat], dim=-1)
        local_h = self.local_proj(local_h)
        
        # Local GNN Aggregation
        local_h = self.local_gnn(local_h, batch_data.edge_index)
        
        # Graph Pooling (Avg Pooling) to get one vector per subgraph
        h_sub = global_mean_pool(local_h, batch_data.batch)
        h_sub = torch.tanh(self.pool_proj(h_sub))
        
        # 6. Final Prediction
        # Feature Fusion (Eq. 21 in paper)
        final_vec = torch.cat([h_sub, gu, gi, ru * rv], dim=-1) 
        pred = torch.sigmoid(self.final_mlp(final_vec))
        
        return pred.squeeze(), (gu, h_sub), (pu, pv)

    def calc_loss(self, pred, target, gl_pairs, intent_probs, args):
        # 1. Prediction Loss (BCE)
        loss_pred = F.binary_cross_entropy(pred, target)
        
        # 2. Global-Local Contrastive Loss (InfoNCE)
        gu, h_sub = gl_pairs
        gu = F.normalize(gu, dim=1)
        h_sub = F.normalize(h_sub, dim=1)
        
        # Positive pairs dot product
        pos_score = (gu * h_sub).sum(dim=1) / args.temp_gl
        # Contrastive loss
        loss_gl = -torch.log(torch.exp(pos_score) / torch.exp(pos_score).sum()).mean()
        
        # 3. Intent Regularization (Entropy)
        # Encourages diverse intent usage
        pu, pv = intent_probs
        # Simple regularization: minimize negative entropy (maximize entropy)
        loss_intent = (pu * torch.log(pu + 1e-9)).sum(dim=1).mean()
        
        total_loss = loss_pred + args.lambda_gl * loss_gl + args.lambda_intent * loss_intent
        return total_loss