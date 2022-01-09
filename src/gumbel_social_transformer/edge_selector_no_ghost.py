import torch
import torch.nn as nn
from torch.nn.functional import softmax
from src.gumbel_social_transformer.mha import VanillaMultiheadAttention
from src.gumbel_social_transformer.utils import _get_activation_fn, gumbel_softmax

class EdgeSelector(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1, activation="relu"):
        super(EdgeSelector, self).__init__()
        assert 4 * d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = 4 * d_model // nhead
        self.self_attn = VanillaMultiheadAttention(4*d_model, nhead, dropout=0.0)
        self.norm_edge = nn.LayerNorm(2*d_model)
        self.norm_node = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(self.head_dim, self.head_dim)
        self.linear2 = nn.Linear(self.head_dim, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, x, A, attn_mask, tau=1., hard=False, device='cuda:0'):
        bsz, nnode, d_model = x.shape
        attn_mask_ped = (attn_mask.sum(-1) > 0).float().unsqueeze(-1).to(device)
        x = self.norm_node(x)
        x = x * attn_mask_ped
        x_neighbor = torch.ones(bsz,nnode,nnode,d_model).to(device)*x.view(bsz,nnode,1,d_model)
        x_target = torch.ones(bsz,nnode,nnode,d_model).to(device)*x.view(bsz,1,nnode,d_model)
        x_neighbor_target = torch.cat((x_neighbor, x_target), dim=-1)
        A = self.norm_edge(A)
        A = A * attn_mask.permute(0,2,1).unsqueeze(-1)
        A = torch.cat((x_neighbor_target, A), dim=-1)

        attn_mask_neighbors = attn_mask.view(bsz, nnode, nnode, 1) * attn_mask.view(bsz, nnode, 1, nnode)
        attn_mask_neighbors = attn_mask_neighbors.view(bsz*nnode, nnode, nnode)
        attn_mask_neighbors = torch.stack([attn_mask_neighbors for _ in range(self.nhead)], dim=1) # (time_step*target_node, nhead, neighbor_node, neighbor_node)
        attn_mask_neighbors = attn_mask_neighbors.view(attn_mask_neighbors.shape[0]*attn_mask_neighbors.shape[1], \
            attn_mask_neighbors.shape[2], attn_mask_neighbors.shape[3]) # (time_step*target_node*nhead, neighbor_node, neighbor_node)

        A_perm = A.permute(0,2,1,3)
        A_perm = A_perm.reshape(bsz*nnode, nnode, 4*d_model)
        A_perm = A_perm.permute(1,0,2)
        
        _, _, A2 = self.self_attn(A_perm, A_perm, A_perm, attn_mask=attn_mask_neighbors) 
        A2 = A2.reshape(bsz, nnode, self.nhead, nnode, self.head_dim)

        A2 = self.linear2(self.dropout1(self.activation(self.linear1(A2)))).squeeze(-1)
        edge_multinomial = softmax(A2, dim=-1)
        edge_multinomial = edge_multinomial * attn_mask.unsqueeze(2)
        edge_multinomial = edge_multinomial / (edge_multinomial.sum(-1).unsqueeze(-1)+1e-10)
        sampled_edges = self.edge_sampler(edge_multinomial, tau=tau, hard=hard)
        return edge_multinomial, sampled_edges

    def edge_sampler(self, edge_multinomial, tau=1., hard=False):
        logits = torch.log(edge_multinomial+1e-10)
        sampled_edges = gumbel_softmax(logits, tau=tau, hard=hard, eps=1e-10)
        return sampled_edges