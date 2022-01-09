import torch
import torch.nn as nn
from src.gumbel_social_transformer.gumbel_social_transformer import GumbelSocialTransformer
from src.gumbel_social_transformer.temporal_convolution_net import TemporalConvolutionNet


def offset_error_square_full_partial(x_pred, x_target, loss_mask_ped, loss_mask_pred_seq):
    assert x_pred.shape[0] == loss_mask_ped.shape[0] == loss_mask_pred_seq.shape[0] == 1
    assert x_pred.shape[1] == x_target.shape[1] == loss_mask_pred_seq.shape[2]
    assert x_pred.shape[2] == x_target.shape[2] == loss_mask_ped.shape[1] == loss_mask_pred_seq.shape[1]
    assert x_pred.shape[3] == x_target.shape[3] == 2
    loss_mask_rel_pred = loss_mask_pred_seq.permute(0, 2, 1).unsqueeze(-1)
    x_pred_m = x_pred * loss_mask_rel_pred
    x_target_m = x_target * loss_mask_rel_pred
    x_pred_m = x_pred_m * loss_mask_ped.unsqueeze(1).unsqueeze(-1)
    x_target_m = x_target_m * loss_mask_ped.unsqueeze(1).unsqueeze(-1)

    pos_pred = torch.cumsum(x_pred_m, dim=1)
    pos_target = torch.cumsum(x_target_m, dim=1)
    offset_error_sq = (((pos_pred-pos_target)**2.).sum(3))[0]

    eventual_loss_mask = loss_mask_rel_pred[0,:,:,0] * loss_mask_ped[0]
    offset_error_sq = offset_error_sq * eventual_loss_mask

    return offset_error_sq, eventual_loss_mask


class st_model(nn.Module):

    def __init__(self, args, device='cuda:0'):
        super(st_model, self).__init__()
        if args.spatial == 'gumbel_social_transformer':
            self.node_embedding = nn.Linear(args.motion_dim, args.embedding_size).to(device)
            self.edge_embedding = nn.Linear(args.motion_dim, 2 * args.embedding_size).to(device)
            self.gumbel_social_transformer = GumbelSocialTransformer(
                args.embedding_size,
                args.spatial_num_heads,
                args.spatial_num_heads_edges,
                args.spatial_num_layers,
                dim_feedforward=128,
                dim_hidden=32,
                dropout=0.1,
                activation="relu",
                attn_mech="vanilla",
                ghost=args.ghost,
            ).to(device)
        else:
            raise RuntimeError('The spatial component is not found.')
        if args.temporal == 'temporal_convolution_net':
            self.temporal_conv_net = TemporalConvolutionNet(
                in_channels=args.embedding_size,
                out_channels=args.output_dim,
                dim_hidden=32,
                nconv=6,
                obs_seq_len=args.obs_seq_len,
                pred_seq_len=args.pred_seq_len).to(device)
        else:
            raise RuntimeError('The temporal component is not tcn.')
        self.args = args

    def raw2gaussian(self, prob_raw):
        mu = prob_raw[:,:,:,:2]
        sx, sy = torch.exp(prob_raw[:,:,:,2:3]), torch.exp(prob_raw[:,:,:,3:4])
        corr = torch.tanh(prob_raw[:,:,:,4:5])
        gaussian_params = (mu, sx, sy, corr)
        return gaussian_params

    def sample_gaussian(self, gaussian_params, device='cuda:0', detach_sample=False, sampling=True):
        mu, sx, sy, corr = gaussian_params
        if sampling:
            if detach_sample:
                mu, sx, sy, corr = mu.detach(), sx.detach(), sy.detach(), corr.detach()
            sample_unit = torch.empty(mu.shape).normal_().to(device)
            sample_unit_x, sample_unit_y = sample_unit[:,:,:,0:1], sample_unit[:,:,:,1:2]
            sample_x = sx*sample_unit_x
            sample_y = corr*sy*sample_unit_x+((1.-corr**2.)**0.5)*sy*sample_unit_y
            sample = torch.cat((sample_x, sample_y), dim=3)+mu
        else:
            sample = mu
        return sample


    def edge_evolution(self, xt_plus, At, device='cuda:0'):
        xt_plus = xt_plus[0,0]
        At = At[0, 0]
        num_nodes, motion_dim = xt_plus.shape
        xt_plus_row = torch.ones(num_nodes,num_nodes,motion_dim).to(device)*xt_plus.view(num_nodes,1,motion_dim)
        xt_plus_col = torch.ones(num_nodes,num_nodes,motion_dim).to(device)*xt_plus.view(1,num_nodes,motion_dim)
        At_plus = At + (xt_plus_row - xt_plus_col)
        At_plus = At_plus.unsqueeze(0).unsqueeze(0)
        return At_plus

    def forward(self, x, A, attn_mask, loss_mask_rel, tau=1., hard=False, sampling=True, device='cuda:0'):
        info = {}
        loss_mask_per_pedestrian = (loss_mask_rel[0].sum(1)==self.args.obs_seq_len+self.args.pred_seq_len).float().unsqueeze(0)
        if self.args.only_observe_full_period:
            assert loss_mask_per_pedestrian.shape[0] == 1
            attn_mask = []
            for tt in range(self.args.obs_seq_len):
                attn_mask.append(torch.outer(loss_mask_per_pedestrian[0], loss_mask_per_pedestrian[0]).float())
            attn_mask = torch.stack(attn_mask, dim=0).unsqueeze(0)
        
        if self.args.spatial == 'gumbel_social_transformer':
            x_embedding = self.node_embedding(x)[0]
            A_embedding = self.edge_embedding(A)[0]
            attn_mask = attn_mask[0].permute(0,2,1)
            xs, sampled_edges, edge_multinomial, attn_weights = self.gumbel_social_transformer(x_embedding, A_embedding, attn_mask, tau=tau, hard=hard, device=device)
            xs = xs.unsqueeze(0)
            info['sampled_edges'], info['edge_multinomial'], info['attn_weights'] = sampled_edges, edge_multinomial, attn_weights
        else:
            raise RuntimeError("The spatial component is not found.")
        
        if self.args.only_observe_full_period:
            loss_mask_rel_full_partial = loss_mask_per_pedestrian[0]
        else:
            loss_mask_rel_obs = loss_mask_rel[0,:,:self.args.obs_seq_len]
            loss_mask_rel_full_partial = loss_mask_rel_obs[:,-1]


        if self.args.decode_style == 'readout':
            xs = xs * loss_mask_rel_obs.permute(1,0).unsqueeze(-1)
            xs = xs * loss_mask_rel_full_partial.unsqueeze(-1)
            if self.args.temporal == 'temporal_convolution_net':
                prob_raw_pred = self.temporal_conv_net(xs)
            else:
                raise RuntimeError('The temporal component can only be tcn for readout decode_style.')
            x_sample_pred, A_sample_pred = [], []
            A_sample = A[:, -1:]
            for tt in range(self.args.pred_seq_len):
                prob_raw = prob_raw_pred[:, tt:tt+1]
                gaussian_params = self.raw2gaussian(prob_raw)
                x_sample = self.sample_gaussian(gaussian_params, device=device, detach_sample=self.args.detach_sample, sampling=sampling)
                A_sample = self.edge_evolution(x_sample, A_sample, device=device)
                x_sample_pred.append(x_sample)
                A_sample_pred.append(A_sample)
            x_sample_pred = torch.cat(x_sample_pred, dim=1)
            A_sample_pred = torch.cat(A_sample_pred, dim=1)
            gaussian_params_pred = self.raw2gaussian(prob_raw_pred)
            info['A_sample_pred'] = A_sample_pred
            info['loss_mask_rel_full_partial'] = loss_mask_rel_full_partial.unsqueeze(0)
            info['loss_mask_per_pedestrian'] = loss_mask_per_pedestrian
            results = (gaussian_params_pred, x_sample_pred, info)
            return results
        else:
            raise RuntimeError("The decoder style is not found.")