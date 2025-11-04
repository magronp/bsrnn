"""
These modules are adapted (with substantial modifications) from the BSRNN authors' version [1].
Improving it with division into heads is inspired from DTTNet [2] and TFGridNet [3].
The TAC module is inspired from [4].

[1] http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit
[2] https://github.com/junyuchen-cjy/DTTNet-Pytorch
[3] https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18
[4] https://github.com/yluo42/TAC/blob/master/utility/models.py
"""

import numpy as np
import torch
import torch.nn as nn
from models.attention import AttentionBlock


def get_bandsplit(sr, enc_dim, target=None):

    r = enc_dim / (sr / 2.0)
    # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
    bandwidth_50 = int(np.floor(50 * r))
    bandwidth_100 = int(np.floor(100 * r))
    bandwidth_250 = int(np.floor(250 * r))
    bandwidth_500 = int(np.floor(500 * r))
    bandwidth_1k = int(np.floor(1000 * r))
    bandwidth_2k = int(np.floor(2000 * r))

    if target == "vocals" or target == "other":
        band_width = [bandwidth_100] * 10
        band_width += [bandwidth_250] * 12
        band_width += [bandwidth_500] * 8
        band_width += [bandwidth_1k] * 8
        band_width += [bandwidth_2k] * 2
    elif target == "bass":
        band_width = [bandwidth_50] * 10
        band_width += [bandwidth_100] * 5
        band_width += [bandwidth_500] * 6
        band_width += [bandwidth_1k] * 4
        band_width += [bandwidth_2k] * 4
    elif target == "drums":
        band_width = [bandwidth_50] * 20
        band_width += [bandwidth_100] * 10
        band_width += [bandwidth_250] * 8
        band_width += [bandwidth_500] * 8
        band_width += [bandwidth_1k] * 8
    else:
        # Joint bandsplit scheme
        band_width = [bandwidth_50] * 20
        band_width += [bandwidth_100] * 10
        band_width += [bandwidth_250] * 8
        band_width += [bandwidth_500] * 8
        band_width += [bandwidth_1k] * 8
        band_width += [bandwidth_2k] * 2

    band_width.append(enc_dim - np.sum(band_width))

    return band_width


class ResNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_fac=2,
        layer_type="lstm",
        eps=1e-7,
        group_num=1,
        bidirectional=True,
    ):
        super().__init__()

        self.input_size = input_size  # the feature dimension (input of the RNN)
        self.layer_type = layer_type
        self.hidden_size = input_size * hidden_fac
        self.proj_input_size = self.hidden_size * (bidirectional + 1)

        # Input normalization layer
        self.norm = nn.GroupNorm(group_num, input_size, eps)

        # Main layer (LSTM / GRU / CNN)
        if layer_type == "lstm":
            self.main_layer = nn.LSTM(
                input_size,
                self.hidden_size,
                1,
                batch_first=True,
                bidirectional=bidirectional,
            )

        elif layer_type == "gru":
            self.main_layer = nn.GRU(
                input_size,
                self.hidden_size,
                1,
                batch_first=True,
                bidirectional=bidirectional,
            )

        elif layer_type == "conv":
            self.hidden_size = self.hidden_size * (bidirectional + 1)
            self.main_layer = nn.Sequential(
                nn.Conv1d(input_size, self.hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        else:
            NameError("Unknown layer type")

        # linear projection layer
        self.proj = nn.Linear(self.proj_input_size, input_size)

    def forward(self, input):
        # input: [B, N, seq_len]
        # "N" is the feature dimension (input size of the RNNs)
        # "seq_len" can be either T (time layer) or K (band/frequency layer)

        # Normalize input
        output = self.norm(input)  # [B, N, seq_len]

        # Main layer (RNN or CNN)
        if self.layer_type == "conv":
            output = self.main_layer(output)  # [B, hidden_size, seq_len]
            output = output.transpose(1, 2)  # [B, seq_len, hidden_size]
        else:
            output = output.transpose(1, 2).contiguous()  # [B, seq_len, N]
            output, _ = self.main_layer(output)  # [B, seq_len, hidden_size]

        # Linear projector (back to input shape)
        output = self.proj(output)  # [B, seq_len, N]
        output = output.transpose(1, 2)  # [B, N, seq_len]

        return input + output


class BSNet(nn.Module):
    def __init__(
        self,
        feature_dim,
        fac_sep=2,
        time_layer="lstm",
        band_layer="lstm",
        bidirectional=True,
        n_heads=1,
        group_num=None,
        n_att_head=0,
        n_bands=20,
        attn_enc_dim=20,
        eps=1e-7,
        tac=False,
        nb_channels=2,
        fac_tac=3,
        act_tac="tanh",
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.input_dim = feature_dim // n_heads
        self.group_num = group_num
        if self.group_num is None:
            self.group_num = max(self.input_dim // 16, 1)
        self.tac = tac
        self.nb_channels = nb_channels

        # Time and band modeling layers
        self.time_net = ResNet(
            self.input_dim,
            hidden_fac=fac_sep,
            layer_type=time_layer,
            eps=eps,
            group_num=self.group_num,
            bidirectional=bidirectional,
        )
        self.band_net = ResNet(
            self.input_dim,
            hidden_fac=fac_sep,
            layer_type=band_layer,
            eps=eps,
            group_num=self.group_num,
            bidirectional=bidirectional,
        )

        # Attention blocks (full-band as per TFGridNet)
        self.n_att_head = n_att_head
        if n_att_head > 0:
            self.attn_block = AttentionBlock(
                self.feature_dim,
                n_bands,
                n_att_head=n_att_head,
                attn_enc_dim=attn_enc_dim,
            )

        # TAC module
        if self.tac:
            self.tac_module = TAC(feature_dim, fac_tac, act_fn=act_tac)

    def forward(self, input):
        # input: [B, K, N, T]
        # B is the batch size (potentially * number of channels)
        # K denotes a frequency-like dimension, e.g., bands after band-split, or frequencies after downsampling
        # N denotes the feature_dim, e.g., latent space of BSRNN, or channels after conv layers
        # T denotes the number of time frames / sequence length

        B, K, N, T = input.shape
        BH = B * self.n_heads
        NdH = N // self.n_heads

        # Split into n_heads
        input = input.transpose(1, 2)  # [B, N, K, T]
        input = input.reshape(BH, NdH, K, T)  # [BH, N/H, K, T]
        input = input.transpose(1, 2)  # [BH, K, N/H, T]

        # Sequence modeling over time
        input = input.reshape(BH * K, NdH, -1)  # [BH*K, N/H, T]
        band_output = self.time_net(input)  # [BH*K, N/H, T]

        # Sequence modeling over bands -- need some reshaping to get to [..., N/H, K]
        band_output = band_output.view(BH, K, -1, T)  # [BH, K, N/H, T]
        band_output = band_output.permute(0, 3, 2, 1).contiguous()  # [BH, T, N/H, K]
        band_output = band_output.view(BH * T, -1, K)  # [BH*T, N/H, K]

        output = self.band_net(band_output)  # [BH*T, N/H, K]

        # Some reshaping
        output = output.view(BH, T, -1, K)  # [BH, T, N/H, K]
        output = output.permute(0, 3, 2, 1).contiguous()  # [BH, K, N/H, T]

        # Back to the input shape
        output = output.transpose(1, 2)  # [BH, N/H, K, T]
        output = output.reshape(B, N, K, T)  # [B, N, K, T]
        output = output.transpose(1, 2)  # [B, K, N, T]

        if self.tac:
            output = output.view(
                B // self.nb_channels, self.nb_channels, K, N, T
            )  # [B', C, K, N, T]
            output = (
                output.permute(0, 2, 4, 1, 3).contiguous().view(-1, self.nb_channels, N)
            )  # [B'*K*T, C, N]
            output = self.tac_module(output)  # [B'*K*T, C, N]
            output = output.view(-1, K, T, self.nb_channels, N)  # [B', K, T, C, N]

            output = (
                output.permute(0, 3, 1, 4, 2).contiguous().view(B, K, N, T)
            )  # [B, K, N, T]

        # Attention heads
        if self.n_att_head > 0:
            output = output.permute(0, 2, 3, 1).contiguous()  # [B, N, T, K]
            output = self.attn_block(output)  # [B, N, T, K]
            output = output.permute(0, 3, 1, 2).contiguous()  # [B, K, N, T]

        return output


class TAC(nn.Module):
    def __init__(self, feature_dim, fac_tac=3, act_fn="tanh"):
        super().__init__()

        hidden_size = feature_dim * fac_tac
        if act_fn == "tanh":
            fn = nn.Tanh()
        elif act_fn == "prelu":
            fn = nn.PReLU()

        self.transform = nn.Sequential(nn.Linear(feature_dim, hidden_size), fn)
        self.average = nn.Sequential(nn.Linear(hidden_size, hidden_size), fn)
        self.concat = nn.Sequential(nn.Linear(hidden_size * 2, feature_dim), fn)

    def forward(self, input):
        # input: [B, C, N]
        v = self.transform(input)  # [B, C, H]
        r = self.average(v.mean(1))  # [B, H]
        r = r.unsqueeze(1).expand_as(v)
        y = torch.cat([v, r], -1)
        y = self.concat(y)
        return y + input  # res connection


class Masker(nn.Module):
    def __init__(
        self, band_width, feature_dim, fac_mask=4, mask_ctxt=0, fac_out_ch=1, eps=1e-8
    ):
        super().__init__()

        self.band_width = band_width
        self.feature_dim = feature_dim
        self.fac_mask = fac_mask
        self.mask_ctxt = mask_ctxt
        self.fac_out_ch = fac_out_ch

        self.mask = nn.ModuleList([])
        for i in range(len(self.band_width)):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * self.fac_mask, 1),
                    nn.Tanh(),
                    nn.Conv1d(
                        self.feature_dim * self.fac_mask,
                        self.feature_dim * self.fac_mask,
                        1,
                    ),
                    nn.Tanh(),
                    nn.Conv1d(
                        self.feature_dim * self.fac_mask,
                        self.band_width[i] * 4 * self.fac_out_ch,
                        1,
                    ),
                )
            )

    def forward(self, input, subband_mix_spec):
        # input: [bsize, nband, N, T] (stereo) or [B, nband, N, T] (classical)
        # subband_mix_spec: list of tensors [B, 2, BW, T]
        # here, B=bsize*nch

        T = input.shape[-1]

        # masking
        sep_subband_spec = []
        for i in range(len(self.band_width)):
            # Compute the mask
            out = self.mask[i](
                input[:, i]
            )  # [bsize, BW*4 *nch, T] (stereo) or [B, BW*4, T] (classical)

            # Reshape to match both the "naive" stereo and classical cases
            out = out.view(-1, 2, 2, self.band_width[i], T)  # [B, 2, 2, BW, T]

            # Compute GLU ang get RI parts
            mask = out[:, 0] * torch.sigmoid(out[:, 1])  # B, 2, BW, T
            mask_real = mask[:, 0]  # B, BW, T
            mask_imag = mask[:, 1]  # B, BW, T

            # Apply the mask
            sb_real = subband_mix_spec[i][:, 0]
            sb_imag = subband_mix_spec[i][:, 1]
            est_real = sb_real * mask_real - sb_imag * mask_imag  # B, BW, T
            est_imag = sb_real * mask_imag + sb_imag * mask_real  # B, BW, T
            est_cplx = torch.complex(est_real, est_imag) # B, BW, T

            # Mask context
            if self.mask_ctxt > 0:
                est_cplx = est_cplx.permute(0, 2, 1).contiguous()
                w = torch.ones((T, 1, self.mask_ctxt * 2 + 1)).type_as(est_cplx)
                est_cplx = torch.nn.functional.conv1d(
                    est_cplx,
                    w,
                    padding="same",
                    groups=T,
                )
                est_cplx = est_cplx.permute(0, 2, 1).contiguous()
            sep_subband_spec.append(est_cplx)

        est_spec = torch.cat(sep_subband_spec, 1)  # B, F, T

        return est_spec


# EOF
