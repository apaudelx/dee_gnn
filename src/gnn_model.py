import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops


class Identity(nn.Module):
    def forward(self, x):
        return x


class EdgeConv(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super(EdgeConv, self).__init__(aggr='add', flow='source_to_target')
        self.input_node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        if node_dim != hidden_dim:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.node_mlp = Identity()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        if edge_attr.size(0) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_attr = torch.zeros((0, self.edge_dim), dtype=torch.float32, device=x.device)
        else:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=num_nodes, fill_value=0.0
            )
        
        x_transformed = self.node_mlp(x)
        if edge_attr.size(0) > 0:
            edge_attr_transformed = self.edge_mlp(edge_attr)
        else:
            edge_attr_transformed = edge_attr
        
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr_transformed, 
                            transformed_x=x_transformed, original_edge_attr=edge_attr)
        
        return out
    
    def message(self, x_i, x_j, edge_attr, original_edge_attr):
        msg_input = torch.cat([x_j, x_i, original_edge_attr], dim=-1)
        msg = self.message_mlp(msg_input)
        return msg
    
    def update(self, aggr_out, transformed_x):
        update_input = torch.cat([transformed_x, aggr_out], dim=-1)
        out = self.update_mlp(update_input)
        return out


class EncapsulationGNN(nn.Module):
    def __init__(self, 
                 node_dim: int = 5,
                 edge_dim: int = 3,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 num_bead_types: int = 100,
                 embedding_dim: int = 32):
        super(EncapsulationGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.bead_type_embedding = nn.Embedding(num_bead_types, embedding_dim)
        self.input_proj = nn.Linear(node_dim + embedding_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EdgeConv(hidden_dim, edge_dim, hidden_dim))
        
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.graph_feat_proj = nn.Linear(8, hidden_dim // 2)
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x, edge_index, edge_attr, batch, bead_type_id, 
                num_atoms=None, num_bonds=None, avg_degree=None,
                max_degree=None, graph_density=None, total_charge=None, 
                charge_std=None, unique_bead_types=None):
        bead_emb = self.bead_type_embedding(bead_type_id)
        x = torch.cat([x, bead_emb], dim=-1)
        x = self.input_proj(x)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if i > 0:
                x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            if i > 0:
                x = x + x_residual
        
        graph_embedding = global_add_pool(x, batch)
        
        if num_atoms is not None and num_bonds is not None:
            def squeeze_if_needed(t):
                return t.squeeze() if t.dim() > 1 else t
            
            graph_feats = torch.stack([
                squeeze_if_needed(num_atoms) / 100.0,
                squeeze_if_needed(num_bonds) / 100.0,
                squeeze_if_needed(avg_degree) / 10.0,
                squeeze_if_needed(max_degree) / 10.0 if max_degree is not None else torch.zeros(graph_embedding.size(0), device=graph_embedding.device),
                squeeze_if_needed(graph_density) if graph_density is not None else torch.zeros(graph_embedding.size(0), device=graph_embedding.device),
                squeeze_if_needed(total_charge) / 10.0 if total_charge is not None else torch.zeros(graph_embedding.size(0), device=graph_embedding.device),
                squeeze_if_needed(charge_std) / 5.0 if charge_std is not None else torch.zeros(graph_embedding.size(0), device=graph_embedding.device),
                squeeze_if_needed(unique_bead_types) / 50.0 if unique_bead_types is not None else torch.zeros(graph_embedding.size(0), device=graph_embedding.device),
            ], dim=-1)
            graph_feat_emb = self.graph_feat_proj(graph_feats)
            graph_embedding = torch.cat([graph_embedding, graph_feat_emb], dim=-1)
        else:
            graph_feat_emb = torch.zeros(graph_embedding.size(0), self.hidden_dim // 2, 
                                        device=graph_embedding.device)
            graph_embedding = torch.cat([graph_embedding, graph_feat_emb], dim=-1)
        
        out = self.regression_head(graph_embedding)
        # out = out * 2.5
        out = torch.sigmoid(out)
        
        return out
