import torch
import torch.nn as nn
from src.gumbel_social_transformer.utils import _get_clones

class GumbelSocialTransformer(nn.Module):
    def __init__(self, d_model, nhead_nodes, nhead_edges, nlayer, dim_feedforward=512, dim_hidden=32, \
        dropout=0.1, activation="relu", attn_mech="vanilla", ghost=True):
        super(GumbelSocialTransformer, self).__init__()
        if ghost:
            if nhead_edges == 0:
                raise RuntimeError("Full connectivity conflicts with the Ghost setting.")
            print("Ghost version.")
            from src.gumbel_social_transformer.edge_selector_ghost import EdgeSelector
            from src.gumbel_social_transformer.node_encoder_layer_ghost import NodeEncoderLayer
        else:
            print("No ghost version.")
            from src.gumbel_social_transformer.edge_selector_no_ghost import EdgeSelector
            from src.gumbel_social_transformer.node_encoder_layer_no_ghost import NodeEncoderLayer
        if nhead_edges != 0:
            # 0 means it is fully connected
            self.edge_selector = EdgeSelector(
                d_model,
                nhead=nhead_edges,
                dropout=dropout,
                activation=activation,
            )
        node_encoder_layer = NodeEncoderLayer(
            d_model,
            nhead_nodes,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            attn_mech=attn_mech,
        )
        self.node_encoder_layers = _get_clones(node_encoder_layer, nlayer)
        self.nlayer = nlayer
        self.nhead_nodes = nhead_nodes
        self.nhead_edges = nhead_edges

    def forward(self, x, A, attn_mask, tau=1., hard=False, device='cuda:0'):
        """
        Pass the input through the encoder layers in turn.
        inputs:
            - x: vertices representing pedestrians of one sample. 
                # * bsz is batch size corresponding to Transformer setting. it corresponds to time steps in pedestrian setting.
                # (bsz, nnode, d_motion)
            - A: edges representation relationships between pedestrians of one sample.
                # (bsz, nnode <neighbor>, nnode <target>, d_motion)
                # row -> neighbor, col -> target
            - attn_mask: attention mask provided in advance.
                # (bsz, nnode <target>, nnode <neighbor>)
                # row -> target, col -> neighbor
                # 1. means yes, i.e. attention exists.  0. means no.
            - tau: temperature hyperparameter of gumbel softmax.
                # Need annealing though training.
            - hard: hard or soft sampling.
                # True means one-hot sample for evaluation.
                # False means soft sample for reparametrization.
            - device: 'cuda:0' or 'cpu'.
        outputs:
            - x: encoded vertices representing pedestrians of one sample. 
                # (bsz, nnode, d_model) # same as input
            - sampled_edges: sampled adjacency matrix at the last column.
                # (time_step, nnode <target>, nhead_edges, neighbor_node)
                # * where neighbor_node = nnode+1 <neighbor> for ghost==True,
                # * and   neighbor_node = nnode   <neighbor> for ghost==False.
            - edge_multinomial: multinomial where sampled_edges are sampled.
                # (time_step, nnode <target>, nhead_edges, neighbor_node)
            - attn_weights: attention weights during self-attention for nodes x.
                # (nlayer, bsz, nhead, nnode <target>, neighbor_node)
        """
        if self.nhead_edges != 0:
            edge_multinomial, sampled_edges = \
                self.edge_selector(x, A, attn_mask, tau=tau, hard=hard, device=device)
        else:
            time_step, nnode = attn_mask.shape[0], attn_mask.shape[1]
            sampled_edges = torch.ones(time_step, nnode, 1, nnode).to(device) * attn_mask.unsqueeze(2)
            edge_multinomial = torch.ones(time_step, nnode, 1, nnode).to(device) * attn_mask.unsqueeze(2)

        attn_weights_list = []
        for i in range(self.nlayer):
            x, attn_weights_layer = self.node_encoder_layers[i](x, sampled_edges, attn_mask, device=device)
            attn_weights_list.append(attn_weights_layer)
        attn_weights = torch.stack(attn_weights_list, dim=0)
        return x, sampled_edges, edge_multinomial, attn_weights
