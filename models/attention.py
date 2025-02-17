"""
The 'LayerNormalization4DCF', 'get_layer', and 'AttentionBlock' functions are adapted from the TFGridNet implementation in the ESPNET toolbox:
https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18
"""

import torch
import torch.nn as nn
import difflib


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = nn.parameter.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.parameter.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,n_band]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])

    return layer_handler


class AttentionBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        feature_dim,
        n_freqs,
        n_att_head=4,
        attn_enc_dim=20,  # 20 at 44kHz ; 4 at 8kHz (as in TFGridNet)
        activation="prelu",
        eps=1e-7,
    ):
        super().__init__()

        assert feature_dim % n_att_head == 0

        self.feature_dim = feature_dim
        self.n_att_head = n_att_head
        self.attn_enc_dim = attn_enc_dim

        for ii in range(n_att_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(feature_dim, attn_enc_dim, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((attn_enc_dim, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(feature_dim, attn_enc_dim, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((attn_enc_dim, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(feature_dim, feature_dim // n_att_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF(
                        (feature_dim // n_att_head, n_freqs), eps=eps
                    ),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((feature_dim, n_freqs), eps=eps),
            ),
        )

    def forward(self, x):
        """GridNetBlock Forward.

        Args:
            x: [B, N, T, n_band]
            out: [B, N, T, n_band]
            N is the feature_dim (embedding size)
        """
        B, _, T, _ = x.shape

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_att_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](x))  # [B, N, T, n_band]
            all_K.append(self["attn_conv_K_%d" % ii](x))  # [B, N, T, n_band]
            all_V.append(self["attn_conv_V_%d" % ii](x))  # [B, N, T, n_band]

        Q = torch.cat(all_Q, dim=0)  # [B', N, T, n_band]
        K = torch.cat(all_K, dim=0)  # [B', N, T, n_band]
        V = torch.cat(all_V, dim=0)  # [B', N, T, n_band]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, N*n_band]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, N*n_band]
        V = V.transpose(1, 2)  # [B', T, N, n_band]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, N*n_band]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = nn.functional.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, N*n_band]

        V = V.reshape(old_shape)  # [B', T, N, n_band]
        V = V.transpose(1, 2)  # [B', N, T, n_band]
        emb_dim = V.shape[1]

        out = V.view([self.n_att_head, B, emb_dim, T, -1])  # [n_head, B, N, T, n_band])
        out = out.transpose(0, 1)  # [B, n_head, N, T, n_band])
        out = out.contiguous().view(
            [B, self.n_att_head * emb_dim, T, -1]
        )  # [B, N, T, n_band])
        out = self["attn_concat_proj"](out)  # [B, N, T, n_band])

        return out + x
