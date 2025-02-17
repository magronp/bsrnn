from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from helpers.transforms import mySTFT, myISTFT
from models.pl_module import PLModule
from models.bs import BSNet


class BSRNNstereo(PLModule):

    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        cfg_eval,
        target="vocals",
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        time_layer="lstm",
        band_layer="lstm",
        bidirectional=True,
        n_heads=1,
        group_num=1,
        feature_dim=128,
        num_repeat=12,
        nb_channels=2,
        fac_mask=4,
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

        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.nb_channels = nb_channels
        self.enc_dim = self.n_fft // 2 + 1
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

        # Band split module
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(
                        1, self.band_width[i] * 2 * self.nb_channels, self.eps
                    ),
                    nn.Conv1d(
                        self.band_width[i] * 2 * self.nb_channels, self.feature_dim, 1
                    ),
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
                    bidirectional=bidirectional,
                    n_heads=n_heads,
                    group_num=group_num,
                    n_att_head=n_att_head,
                    n_bands=self.nband,
                    attn_enc_dim=attn_enc_dim,
                    eps=self.eps,
                )
            )
        self.separator = nn.Sequential(*self.separator)

        self.mask = nn.ModuleList([])
        self.fac_mask = fac_mask
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, self.eps),
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
                        self.band_width[i] * 4 * self.nb_channels,
                        1,
                    ),
                )
            )

    def forward(self, input):
        # input shape: (B, nch, n_samples)

        B, nch, nsample = input.shape
        spec = self.stft(input)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 2)  # B, nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(
                spec_RI[:, :, :, band_idx : band_idx + self.band_width[i]].contiguous()
            )  # B, nch, 2, BW, T
            subband_mix_spec.append(
                spec[:, :, band_idx : band_idx + self.band_width[i]]
            )  # B, nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](subband_spec[i].view(B, self.band_width[i] * 2 * nch, -1))
            )
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        # separator
        sep_output = self.separator(subband_feature)
        sep_output = sep_output.view(B, self.nband, self.feature_dim, -1)

        # masking
        sep_subband_spec = []
        for i in range(len(self.band_width)):
            # Compute the mask, reshape, and compute GLU
            out = self.mask[i](sep_output[:, i])  # B, BW*4*nch, T
            out = out.view(B, nch, 2, 2, self.band_width[i], -1)  # B, nch, 2, 2, BW, T
            mask = out[:, :, 0] * torch.sigmoid(out[:, :, 1])  # B, nch, 2, BW, T
            # Apply the mask
            mask_real = mask[:, :, 0]  # B, nch, BW, T
            mask_imag = mask[:, :, 1]  # B, nch, BW, T
            est_spec_real = (
                subband_mix_spec[i].real * mask_real
                - subband_mix_spec[i].imag * mask_imag
            )  # B, nch, BW, T
            est_spec_imag = (
                subband_mix_spec[i].real * mask_imag
                + subband_mix_spec[i].imag * mask_real
            )  # B, nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 2)  # B, nch, F, T

        output = self.istft(est_spec, length=nsample)

        # add the "n_target" dimension
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
            "sdr_type": "global",
            "win_dur": 1.0,
            "verbose_per_track": True,
            "rec_dir": None,
        }
    )

    model = BSRNNstereo(
        cfg_optim,
        cfg_scheduler,
        cfg_eval,
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        feature_dim=16,
        num_repeat=2,
        bidirectional=True,
        n_heads=2,
        group_num=None,
        n_att_head=2,
        attn_enc_dim=16,
        target="vocals",
    )
    print("Number of parameters:", model.count_params())

    # Example for the forward pass
    x = torch.randn((4, 2, 10000))
    output = model(x)
    print(output["waveforms"].shape, output["stfts"].shape)

# EOF
