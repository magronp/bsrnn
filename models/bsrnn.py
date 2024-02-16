"""
The 'LayerNormalization4DCF', 'get_layer', and 'AttentionBlock' functions are adapted from the TFGridNet implementation in the ESPNET toolbox (https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18)

BSRNN-related classes (ResNet, BSNet, BSRNN) are adapted from the authors' version (http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit )

Main modifications:
- using the ligthning module, and adding the stft/istft as attributes in BSRNN
- some reshaping in the forward function to allow for more flexibility in defining BSNets / ResNets
- output both the stft and the waveform, since both are needed to compute the loss mentioned in the BSRNN paper
"""

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from models.pl_module import PLModule
from helpers.transforms import mySTFT, myISTFT
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
        )  # [B,1,T,F]
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
            x: [B, N, T, F]
            out: [B, N, T, F]
            Here, F denotes number of bands (after band-split), N is the feature_dim (embedding size)
        """
        B, _, T, _ = x.shape

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_att_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](x))  # [B, N, T, F]
            all_K.append(self["attn_conv_K_%d" % ii](x))  # [B, N, T, F]
            all_V.append(self["attn_conv_V_%d" % ii](x))  # [B, N, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', N, T, F]
        K = torch.cat(all_K, dim=0)  # [B', N, T, F]
        V = torch.cat(all_V, dim=0)  # [B', N, T, F]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, N*F]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, N*F]
        V = V.transpose(1, 2)  # [B', T, N, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, N*F]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = nn.functional.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, N*F]

        V = V.reshape(old_shape)  # [B', T, N, F]
        V = V.transpose(1, 2)  # [B', N, T, F]
        emb_dim = V.shape[1]

        out = V.view([self.n_att_head, B, emb_dim, T, -1])  # [n_head, B, N, T, F])
        out = out.transpose(0, 1)  # [B, n_head, N, T, F])
        out = out.contiguous().view(
            [B, self.n_att_head * emb_dim, T, -1]
        )  # [B, N, T, F])
        out = self["attn_concat_proj"](out)  # [B, N, T, F])

        return out + x


class ResNet(nn.Module):
    def __init__(self, input_size, layer_type="gru", eps=1e-7):
        super().__init__()

        self.input_size = input_size
        self.proj_input_size = input_size * 4
        self.layer_type = layer_type

        # Input normalization layer
        self.norm = nn.GroupNorm(1, input_size, eps)

        # Main layer (RNN or CNN)
        if layer_type in ["gru", "lstm"]:
            self.hidden_size = input_size * 2
            self.main_layer = get_layer(layer_type)(
                input_size, self.hidden_size, 1, batch_first=True, bidirectional=True
            )

        elif layer_type == "conv":
            self.hidden_size = input_size * 4
            self.main_layer = nn.Sequential(
                nn.Conv1d(input_size, self.hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        else:
            NameError("Unknown layer type")

        # linear projection layer
        self.proj = nn.Linear(self.proj_input_size, input_size)

    def forward(self, input):
        # input: [B, input_size, seq_len]

        # Normalize input
        output = self.norm(input)  # [B, input_size, seq_len]

        # Main layer (RNN or CNN)
        if self.layer_type == "conv":
            output = self.main_layer(output)  # [B, hidden_size, seq_len]
            output = output.transpose(1, 2)  # [B, seq_len, hidden_size]
        else:
            output = output.transpose(1, 2).contiguous()  # [B, seq_len, input_size]
            output, _ = self.main_layer(output)  # [B, seq_len, hidden_size]

        # Linear projector (back to input shape)
        output = self.proj(output)  # [B, seq_len, input_size]
        output = output.transpose(1, 2)  # [B, input_size, seq_len]

        return input + output


class BSNet(nn.Module):
    def __init__(
        self,
        feature_dim,
        time_layer="gru",
        band_layer="conv",
        n_att_head=0,
        n_bands=20,
        attn_enc_dim=20,
        eps=1e-7,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_att_head = n_att_head

        self.time_net = ResNet(self.feature_dim, layer_type=time_layer, eps=eps)
        self.band_net = ResNet(self.feature_dim, layer_type=band_layer, eps=eps)
        if n_att_head > 0:
            self.attn_block = AttentionBlock(
                self.feature_dim,
                n_bands,
                n_att_head=n_att_head,
                attn_enc_dim=attn_enc_dim,
            )

    def forward(self, input):
        # input: [B, nband, N, T] (N=feature_dim)
        B, nband, N, T = input.shape

        # Sequence modeling over time
        input = input.view(B * nband, N, -1)  # [B*nband, N, T]
        band_output = self.time_net(input)  # [B*nband, N, T]

        # Sequence modeling over bands -- need some reshaping to get to [..., N, nband]
        band_output = band_output.view(B, nband, -1, T)  # [B, nband, N, T]
        band_output = band_output.permute(0, 3, 2, 1).contiguous()  # [B, T, N, nband]
        band_output = band_output.view(B * T, -1, nband)  # [B*T, N, nband]

        output = self.band_net(band_output)  # [B*T, N, nband]

        # Back to the input shape
        output = output.view(B, T, -1, nband)  # [B, T, N, nband]
        output = output.permute(0, 3, 2, 1).contiguous()  # [B, nband, N, T]

        if self.n_att_head > 0:
            output = output.permute(0, 2, 3, 1).contiguous()  # [B, N, T, nband]
            output = self.attn_block(output)  # [B, N, T, nband]
            output = output.permute(0, 3, 1, 2).contiguous()  # [B, nband, N, T]

        return output


class BSRNN(PLModule):

    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        target="vocals",
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        time_layer="gru",
        band_layer="conv",
        feature_dim=128,
        num_repeat=12,
        n_att_head=0,
        attn_enc_dim=20,
        eps=1e-7,
        *args,
        **kwargs
    ):

        # Inherit from the PLModule
        super().__init__(
            cfg_optim, cfg_scheduler, targets=target, sample_rate=sample_rate, eps=eps
        )

        instrument = target
        sr = sample_rate
        self.sr = sr
        self.win = n_fft
        self.stride = n_hop
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.num_repeat = num_repeat
        self.time_layer = time_layer
        self.band_layer = band_layer

        # Transforms
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)

        # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_50 = int(np.floor(50 / (sr / 2.0) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sr / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * self.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (sr / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * self.enc_dim))

        if instrument == "vocals" or instrument == "other":
            self.band_width = [bandwidth_100] * 10
            self.band_width += [bandwidth_250] * 12
            self.band_width += [bandwidth_500] * 8
            self.band_width += [bandwidth_1k] * 8
            self.band_width += [bandwidth_2k] * 2
        elif instrument == "bass":
            self.band_width = [bandwidth_50] * 10
            self.band_width += [bandwidth_100] * 5
            self.band_width += [bandwidth_500] * 6
            self.band_width += [bandwidth_1k] * 4
            self.band_width += [bandwidth_2k] * 4
        elif instrument == "drums":
            self.band_width = [bandwidth_50] * 20
            self.band_width += [bandwidth_100] * 10
            self.band_width += [bandwidth_250] * 8
            self.band_width += [bandwidth_500] * 8
            self.band_width += [bandwidth_1k] * 8
        else:
            print("Unknown Instrument {}".format(instrument))
            raise NotImplementedError

        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                )
            )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(
                BSNet(
                    self.feature_dim,
                    time_layer=time_layer,
                    band_layer=band_layer,
                    n_att_head=n_att_head,
                    n_bands=self.nband,
                    attn_enc_dim=attn_enc_dim,
                    eps=self.eps,
                )
            )
        self.separator = nn.Sequential(*self.separator)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, self.eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1),
                )
            )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        # input shape: (B, nch, n_samples)

        batch_size, nch, nsample = input.shape
        input = input.view(batch_size * nch, -1)

        spec = self.stft(input)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(
                spec_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous()
            )
            subband_mix_spec.append(
                spec[:, band_idx : band_idx + self.band_width[i]]
            )  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](
                    subband_spec[i].view(batch_size * nch, self.band_width[i] * 2, -1)
                )
            )
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        # separator
        sep_output = self.separator(subband_feature)
        sep_output = sep_output.view(batch_size * nch, self.nband, self.feature_dim, -1)

        # masking
        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i]).view(
                batch_size * nch, 2, 2, self.band_width[i], -1
            )
            this_mask = this_output[:, 0] * torch.sigmoid(
                this_output[:, 1]
            )  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = (
                subband_mix_spec[i].real * this_mask_real
                - subband_mix_spec[i].imag * this_mask_imag
            )  # B*nch, BW, T
            est_spec_imag = (
                subband_mix_spec[i].real * this_mask_imag
                + subband_mix_spec[i].imag * this_mask_real
            )  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T

        output = self.istft(est_spec, length=nsample)
        output = output.view(batch_size, nch, -1)

        # adjust to proper sizes
        est_spec = est_spec.view(batch_size, nch, self.enc_dim, -1)
        est_spec = est_spec.unsqueeze(1)  # B, 1, nch, F, T
        output = output.unsqueeze(1)  # B, 1, nch, n_samples

        return {"waveforms": output, "stfts": est_spec}


if __name__ == "__main__":

    cfg_optim = OmegaConf.create({"lr": 0.001, "loss_type": "L1", "loss_domain": "t"})
    cfg_scheduler = OmegaConf.create({"name": "plateau", "factor": 0.5, "patience": 3})

    model = BSRNN(
        cfg_optim,
        cfg_scheduler,
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        feature_dim=16,
        num_repeat=2,
        target="vocals",
    )
    print("Number of parameters:", model.count_params())

    # Example for the forward pass
    x = torch.randn((4, 2, 10000))
    output = model(x)
    print(output["waveforms"].shape, output["stfts"].shape)

# EOF
