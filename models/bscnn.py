from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from models.pl_module import PLModule
from helpers.transforms import mySTFT, myISTFT
from models.attention import AttentionBlock


class ResNetCNN(nn.Module):
    def __init__(
        self, input_size, layer_params={"n_dil_conv": 2, "ks": 3, "hs_fac": 2}, eps=1e-7
    ):
        super().__init__()

        self.input_size = input_size
        self.n_dil_conv = layer_params["n_dil_conv"]
        self.ks = layer_params["ks"]
        self.hs_fac = layer_params["hs_fac"]

        self.hidden_size = input_size * self.hs_fac
        self.in_layer = nn.Sequential(
            nn.GroupNorm(1, input_size, eps),
            nn.Conv1d(
                input_size, self.hidden_size, kernel_size=self.ks, padding="same"
            ),
            nn.ReLU(),
        )

        self.dil_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, self.hidden_size),
                    nn.Conv1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=self.ks,
                        padding="same",
                        dilation=2**il,
                    ),
                    nn.ReLU(),
                )
                for il in range(self.n_dil_conv)
            ]
        )

        self.hidden_size_conc = self._compute_hid_size_list(self.n_dil_conv)
        self.concat_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, self.hidden_size_conc[il]),
                    nn.Conv1d(
                        self.hidden_size_conc[il],
                        self.hidden_size,
                        kernel_size=self.ks,
                        padding="same",
                    ),
                    nn.ReLU(),
                )
                for il in range(self.n_dil_conv)
            ]
        )

        self.out_in_size = self.hidden_size * max(1, self.n_dil_conv)
        self.out_layer = nn.Sequential(
            nn.GroupNorm(1, self.out_in_size, eps),
            nn.Conv1d(
                self.out_in_size, input_size, kernel_size=self.ks, padding="same"
            ),
            nn.ReLU(),
        )

    def _compute_hid_size_list(self, n):
        if n < 3:
            l = [n] * n
        else:
            l = [3] * (n - 2)
            l.insert(0, 2)
            l.append(2)
        return [x * self.hidden_size for x in l]

    def forward(self, input):
        # input: [B, input_size, seq_len]

        # Input convolution (to hidden_size)
        output = self.in_layer(input)  # [B, hidden_size, seq_len]

        # Apply the stacked dilated conv if needed
        if self.n_dil_conv > 0:
            # Dilated Conv
            all_dil = []
            for l in self.dil_layers:
                output = l(output)  # [B, hidden_size, seq_len]
                all_dil.append(output)

            # Init the stack of in/out for the concat convolutions
            stack_in = [all_dil[0]]
            stack_out = []
            for il, l in enumerate(self.concat_layers):

                # remove the previous dil_conv output (if not the first or second)
                if il > 1:
                    stack_in = stack_in[1:]
                # add the nex dil_conv output (if not the last)
                if il != self.n_dil_conv - 1:
                    stack_in.append(all_dil[il + 1])
                in_conc = torch.cat(stack_in, dim=1)
                # feed it to the conv
                out_l = l(in_conc)
                stack_out.append(out_l)
            output = torch.cat(stack_out, dim=1)

        # Output layer
        output = self.out_layer(output)  # [B, input_size, seq_len]

        return input + output


class BSNet(nn.Module):
    def __init__(
        self,
        feature_dim=64,
        time_layer={"n_dil_conv": 2, "ks": 3, "hs_fac": 2},
        band_layer={"n_dil_conv": 2, "ks": 3, "hs_fac": 2},
        n_bands=20,
        n_att_head=0,
        attn_enc_dim=20,
        eps=1e-7,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_att_head = n_att_head
        self.time_layer = time_layer
        self.band_layer = band_layer

        if time_layer is not None:
            self.time_net = ResNetCNN(feature_dim, layer_params=time_layer, eps=eps)
        if band_layer is not None:
            self.band_net = ResNetCNN(feature_dim, layer_params=band_layer, eps=eps)

        if n_att_head > 0:
            self.attn_block = AttentionBlock(
                self.feature_dim,
                n_bands,
                n_att_head=n_att_head,
                attn_enc_dim=attn_enc_dim,
            )

    def forward(self, input):
        # input shape: [B, nband, N, T] (N=feature_dim)
        B, nband, N, T = input.shape
        output = input  # [B, nband, N, T]

        if self.time_layer is not None:
            # Sequence modeling over time
            output = output.view(B * nband, N, -1)  # [B*nband, N, T]
            output = self.time_net(output)  # [B*nband, N, T]
            output = output.view(B, nband, -1, T)  # [B, nband, N, T]

        if self.band_layer is not None:
            # Sequence modeling over bands -- need some reshaping to get to [..., N, nband]
            output = output.permute(0, 3, 2, 1).contiguous()  # [B, T, N, nband]
            output = output.view(B * T, -1, nband)  # [B*T, N, nband]

            output = self.band_net(output)  # [B*T, N, nband]

            # Back to the input shape
            output = output.view(B, T, -1, nband)  # [B, T, N, nband]
            output = output.permute(0, 3, 2, 1).contiguous()  # [B, nband, N, T]

        # Attention
        if self.n_att_head > 0:
            output = output.permute(0, 2, 3, 1).contiguous()  # [B, N, T, nband]
            output = self.attn_block(output)  # [B, N, T, nband]
            output = output.permute(0, 3, 1, 2).contiguous()  # [B, nband, N, T]

        return output


class BSCNN(PLModule):

    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        cfg_eval,
        target="vocals",
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        feature_dim=128,
        num_repeat=12,
        time_layer={"n_dil_conv": 2, "ks": 3, "hs_fac": 2},
        band_layer={"n_dil_conv": 2, "ks": 3, "hs_fac": 2},
        n_att_head=0,
        attn_enc_dim=20,
        eps=1e-7,
        *args,
        **kwargs
    ):

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            cfg_eval,
            targets=target,
            sample_rate=sample_rate,
            eps=eps,
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

        # Band split module
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                )
            )

        # Separator module
        self.separator = []
        for i in range(num_repeat):
            self.separator.append(
                BSNet(
                    self.feature_dim,
                    time_layer=time_layer,
                    band_layer=band_layer,
                    n_bands=self.nband,
                    n_att_head=n_att_head,
                    attn_enc_dim=attn_enc_dim,
                    eps=self.eps,
                )
            )
        self.separator = nn.Sequential(*self.separator)

        # Mask estimation module
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

    def forward(self, input):
        # input shape: (B, nch, n_samples)

        B, nch, nsample = input.shape
        input = input.view(B * nch, -1)

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
                self.BN[i](subband_spec[i].view(B * nch, self.band_width[i] * 2, -1))
            )
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        # separator
        sep_output = self.separator(subband_feature)
        sep_output = sep_output.view(B * nch, self.nband, self.feature_dim, -1)

        # masking
        sep_subband_spec = []
        for i in range(len(self.band_width)):
            # Compute the mask, reshape, and compute GLU
            out = self.mask[i](sep_output[:, i])  # B*nch, BW*4, T
            out = out.view(B * nch, 2, 2, self.band_width[i], -1)  # B*nch, 2, 2, BW, T
            mask = out[:, 0] * torch.sigmoid(out[:, 1])  # B*nch, 2, BW, T

            # Apply the mask
            mask_real = mask[:, 0]  # B*nch, BW, T
            mask_imag = mask[:, 1]  # B*nch, BW, T
            est_spec_real = (
                subband_mix_spec[i].real * mask_real
                - subband_mix_spec[i].imag * mask_imag
            )  # B*nch, BW, T
            est_spec_imag = (
                subband_mix_spec[i].real * mask_imag
                + subband_mix_spec[i].imag * mask_real
            )  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T

        output = self.istft(est_spec, length=nsample)
        output = output.view(B, nch, -1)

        # adjust to proper sizes
        est_spec = est_spec.view(B, nch, self.enc_dim, -1)
        est_spec = est_spec.unsqueeze(1)  # B, 1, nch, F, T
        output = output.unsqueeze(1)  # B, 1, nch, n_samples

        return {"waveforms": output, "stfts": est_spec}


if __name__ == "__main__":

    cfg_optim = OmegaConf.create(
        {
            "lr": 0.001,
            "loss_type": "L1",
            "loss_domain": "t",
            "monitor_val": "sdr",
            "weight_mag": 0.1,
            "algo": "adam",
        }
    )
    cfg_scheduler = OmegaConf.create({"name": "plateau", "factor": 0.5, "patience": 3})
    cfg_eval = OmegaConf.create(
        {
            "device": "cpu",
            "segment_len": 10,
            "overlap": 0.1,
            "hop_size": None,
            "sdr_type": "usdr",
            "win_dur": 1.0,
            "verbose_per_track": True,
            "rec_dir": None,
        }
    )

    model = BSCNN(
        cfg_optim,
        cfg_scheduler,
        cfg_eval,
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        feature_dim=32,
        time_layer={"n_dil_conv": 2, "ks": 3, "hs_fac": 2},
        band_layer=None,
        num_repeat=2,
        target="vocals",
    )
    print("Number of parameters:", model.count_params())

    # Example for the forward pass
    x = torch.randn((4, 2, 10000))
    output = model(x)
    print(output["waveforms"].shape, output["stfts"].shape)

# EOF
