import torch
import torch.nn as nn
from src.gumbel_social_transformer.gumbel_social_transformer import GumbelSocialTransformer


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
        if args.temporal == 'lstm':
            self.lstm = nn.LSTM(
                    input_size=args.embedding_size,
                    hidden_size=args.lstm_hidden_size,
                    num_layers=args.lstm_num_layers,
                    batch_first=False,
                    dropout=0.,
                    bidirectional=False,
                    ).to(device)
            self.hidden2pos = nn.Linear(args.lstm_num_layers*args.lstm_hidden_size, args.output_dim).to(device)
        else:
            raise RuntimeError('The temporal component is not lstm.')
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
            info['sampled_edges'], info['edge_multinomial'], info['attn_weights'] = [], [], []
            info['sampled_edges'].append(sampled_edges)
            info['edge_multinomial'].append(edge_multinomial)
            info['attn_weights'].append(attn_weights) 
        else:
            raise RuntimeError("The spatial component is not found.")
        if self.args.temporal == 'lstm':
            num_peds = xs.shape[2]
            ht = torch.zeros(self.args.lstm_num_layers, num_peds, self.args.lstm_hidden_size).to(device)
            ct = torch.zeros(self.args.lstm_num_layers, num_peds, self.args.lstm_hidden_size).to(device) 
            for tt in range(self.args.obs_seq_len):
                loss_mask_rel_tt = loss_mask_rel[0,:,tt:tt+1]
                xs_tt = xs[0, tt:tt+1]*loss_mask_rel_tt
                _, (htp, ctp) = self.lstm(xs_tt, (ht, ct))
                ht = htp * loss_mask_rel_tt + ht * (1.-loss_mask_rel_tt)
                ct = ctp * loss_mask_rel_tt + ct * (1.-loss_mask_rel_tt)
        else:
            raise RuntimeError('The temporal component is not lstm.')
 
        if self.args.only_observe_full_period:
            loss_mask_rel_full_partial = loss_mask_per_pedestrian[0]
        else:
            loss_mask_rel_obs = loss_mask_rel[0,:,:self.args.obs_seq_len]
            loss_mask_rel_full_partial = loss_mask_rel_obs[:,-1]
        ht = ht * loss_mask_rel_full_partial.unsqueeze(-1)
        ct = ct * loss_mask_rel_full_partial.unsqueeze(-1)
        attn_mask_pred = torch.outer(loss_mask_rel_full_partial, loss_mask_rel_full_partial).unsqueeze(0).float()
        
        if self.args.decode_style == 'recursive':
            if self.args.temporal == 'lstm':
                prob_raw = self.hidden2pos(ht.permute(1,0,2).reshape(num_peds, -1)).unsqueeze(0).unsqueeze(0)
                gaussian_params = self.raw2gaussian(prob_raw)
                x_sample = self.sample_gaussian(gaussian_params, device=device, detach_sample=self.args.detach_sample, sampling=sampling)
                x_sample = x_sample * loss_mask_rel_full_partial.unsqueeze(-1)
                A_sample = self.edge_evolution(x_sample, A[:,-1:], device=device)
                prob_raw_pred, x_sample_pred, A_sample_pred = [], [], []
                prob_raw_pred.append(prob_raw)
                x_sample_pred.append(x_sample)
                A_sample_pred.append(A_sample)
                for tt in range(1, self.args.pred_seq_len):
                    if self.args.spatial == 'gumbel_social_transformer':
                        x_sample_embedding = self.node_embedding(x_sample)[0]
                        A_sample_embedding = self.edge_embedding(A_sample)[0]
                        attn_mask_pred_perm = attn_mask_pred.permute(0,2,1)
                        xs_tt, sampled_edges, edge_multinomial, attn_weights = \
                            self.gumbel_social_transformer(x_sample_embedding, A_sample_embedding, attn_mask_pred_perm, \
                            tau=tau, hard=hard, device=device)
                        info['sampled_edges'].append(sampled_edges)
                        info['edge_multinomial'].append(edge_multinomial)
                        info['attn_weights'].append(attn_weights)
                        loss_mask_rel_tt = loss_mask_rel_full_partial.unsqueeze(-1)
                        xs_tt = xs_tt*loss_mask_rel_tt
                        _, (htp, ctp) = self.lstm(xs_tt, (ht, ct))
                        ht = htp * loss_mask_rel_tt + ht * (1.-loss_mask_rel_tt)
                        ct = ctp * loss_mask_rel_tt + ct * (1.-loss_mask_rel_tt)
                        prob_raw = self.hidden2pos(ht.permute(1,0,2).reshape(num_peds, -1)).unsqueeze(0).unsqueeze(0)
                        gaussian_params = self.raw2gaussian(prob_raw)
                        x_sample = self.sample_gaussian(gaussian_params, device=device, detach_sample=self.args.detach_sample, sampling=sampling)
                        x_sample = x_sample * loss_mask_rel_full_partial.unsqueeze(-1)
                        A_sample = self.edge_evolution(x_sample, A_sample, device=device)
                        prob_raw_pred.append(prob_raw)
                        x_sample_pred.append(x_sample)
                        A_sample_pred.append(A_sample)
                    else:
                        raise RuntimeError("The spatial component is not found.")
                
                prob_raw_pred = torch.cat(prob_raw_pred, dim=1)
                x_sample_pred = torch.cat(x_sample_pred, dim=1)
                A_sample_pred = torch.cat(A_sample_pred, dim=1)
                gaussian_params_pred = self.raw2gaussian(prob_raw_pred)

                info['sampled_edges'] = torch.cat(info['sampled_edges'], dim=0)
                info['edge_multinomial'] = torch.cat(info['edge_multinomial'], dim=0)
                info['attn_weights'] = torch.cat(info['attn_weights'], dim=1)
                info['A_sample_pred'] = A_sample_pred
                info['loss_mask_rel_full_partial'] = loss_mask_rel_full_partial.unsqueeze(0)
                info['loss_mask_per_pedestrian'] = loss_mask_per_pedestrian
                results = (gaussian_params_pred, x_sample_pred, info)
                return results
            else:
                raise RuntimeError('The temporal component is not lstm.')
        else:
            raise RuntimeError("The decoder style is not recursive.")