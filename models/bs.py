"""
These modules are adapted (with substantial modifications) from the BSRNN authors' version [1].
Improving it with division into heads is inspired from DTTNet [2] and TFGridNet [3].

[1] http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit
[2] https://github.com/junyuchen-cjy/DTTNet-Pytorch
[3] https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18
"""

import torch.nn as nn
from models.attention import AttentionBlock


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
        time_layer="lstm",
        band_layer="lstm",
        bidirectional=True,
        n_heads=1,
        group_num=None,
        n_att_head=0,
        n_bands=20,
        attn_enc_dim=20,
        eps=1e-7,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.input_dim = feature_dim // n_heads
        self.group_num = group_num
        if self.group_num is None:
            self.group_num = max(self.input_dim // 16, 1)

        # Time and band modeling layers
        self.time_net = ResNet(
            self.input_dim,
            layer_type=time_layer,
            eps=eps,
            group_num=self.group_num,
            bidirectional=bidirectional,
        )
        self.band_net = ResNet(
            self.input_dim,
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

    def forward(self, input):
        # input: [B, K, N, T]
        # K denotes a frequency-like dimension, e.g., bands after band-split, or frequencies after downsampling
        # N denotes the feature_dim, e.g., latent space of BSRNN, or channels after conv layers

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

        # Attention heads
        if self.n_att_head > 0:
            output = output.permute(0, 2, 3, 1).contiguous()  # [BH, N/H, T, K]
            output = self.attn_block(output)  # [BH, N/H, T, K]
            output = output.permute(0, 3, 1, 2).contiguous()  # [BH, K, N/H, T]

        return output


# EOF
