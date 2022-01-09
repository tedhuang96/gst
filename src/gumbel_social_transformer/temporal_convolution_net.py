import torch.nn as nn
from src.gumbel_social_transformer.utils import _get_clones, _get_activation_fn

class TemporalConvolutionNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim_hidden,
        nconv=2,
        obs_seq_len=8,
        pred_seq_len=12,
        kernel_size=3,
        stride=1,
        dropout=0.1,
        activation="relu",
        ):
        super(TemporalConvolutionNet,self).__init__()
        assert kernel_size % 2 == 1
        assert nconv >= 2
        padding = ((kernel_size - 1) // 2, 0)
        norm_layer = nn.LayerNorm(in_channels)
        timeconv_layer = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), (stride, 1), padding)

        self.nconv = nconv
        self.norms = _get_clones(norm_layer, nconv)
        self.timeconvs = _get_clones(timeconv_layer, nconv)

        self.timelinear1 = nn.Linear(obs_seq_len, pred_seq_len)
        self.timelinear2 = nn.Linear(pred_seq_len, pred_seq_len)
        self.timedropout1 = nn.Dropout(dropout)
        self.timedropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(in_channels, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
    
    def forward(self, x):
        for i in range(self.nconv):
            x_norm = self.norms[i](x)
            x_perm = x_norm.permute(0, 3, 1, 2)
            x_perm = self.activation(self.timeconvs[i](x_perm))
            x_perm = x_perm.permute(0, 2, 3, 1)
            x = x + x_perm
        x = x.permute(0, 2, 3, 1)
        x = self.timedropout1(self.activation(self.timelinear1(x)))
        x = self.timedropout2(self.activation(self.timelinear2(x)))
        x = x.permute(0, 3, 1, 2)
        x = self.dropout(self.activation(self.linear1(x)))
        out = self.linear2(x)
        return out