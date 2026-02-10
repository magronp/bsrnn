"""
The BSRNN main module below is adapted from the authors' version:
http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit

Main modifications:
- using our own ligthning module, and adding the stft/istft as attributes in BSRNN
- some reshaping in the forward function to allow for more flexibility in defining BSNets / ResNets / Masker
- output both the stft and the waveform, since both are needed to compute the loss mentioned in the original paper
- add several inputs to control the architecture (layer types, fac_mask, attention...)
"""

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from helpers.transforms import mySTFT, myISTFT
from models.pl_module import PLModule
from models.bs import get_bandsplit, BSNet, Masker


class BSRNN(PLModule):

    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        cfg_eval,
        targets=["vocals"],
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        nb_channels=2,
        feature_dim=128,
        joint_bandsplit=False,
        num_repeat=12,
        fac_sep=2,
        time_layer="lstm",
        band_layer="lstm",
        bidirectional=True,
        fac_mask=4,
        mask_ctxt=0,
        subtract_last_trgt=False,
        n_heads=1,
        group_num=1,
        n_att_head=0,
        attn_enc_dim=16,
        stereo=None,
        fac_tac=3,
        act_tac="tanh",
        eps=1e-7,
        *args,
        **kwargs
    ):

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            cfg_eval,
            targets=targets,
            sample_rate=sample_rate,
            eps=eps,
        )

        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.enc_dim = self.n_fft // 2 + 1
        self.feature_dim = feature_dim
        self.num_repeat = num_repeat
        self.time_layer = time_layer
        self.band_layer = band_layer
        self.n_targets = len(self.targets)
        self.nb_channels = nb_channels
        self.stereonaive = stereo == "naive"

        # Transforms
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)

        # Band split module
        trgbs = None if (joint_bandsplit or self.n_targets > 1) else self.targets[0]

        self.band_width = get_bandsplit(sample_rate, self.enc_dim, trgbs)
        self.nband = len(self.band_width)
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(
                        1, self.band_width[i] * 2 * (1 + self.stereonaive), self.eps
                    ),
                    nn.Conv1d(
                        self.band_width[i] * 2 * (1 + self.stereonaive),
                        self.feature_dim,
                        1,
                    ),
                )
            )

        # Separator module
        self.separator = []
        for i in range(num_repeat):
            self.separator.append(
                BSNet(
                    self.feature_dim,
                    fac_sep=fac_sep,
                    time_layer=time_layer,
                    band_layer=band_layer,
                    bidirectional=bidirectional,
                    n_heads=n_heads,
                    group_num=group_num,
                    n_att_head=n_att_head,
                    n_bands=self.nband,
                    attn_enc_dim=attn_enc_dim,
                    eps=self.eps,
                    tac=(stereo == "tac"),
                    nb_channels=nb_channels,
                    fac_tac=fac_tac,
                    act_tac=act_tac,
                )
            )
        self.separator = nn.Sequential(*self.separator)

        # Mask estimation module
        self.subtract_last_trgt = subtract_last_trgt
        if len(self.targets) == 1:
            self.subtract_last_trgt = False

        self.maskers = nn.ModuleList([])
        for _ in range(self.n_targets - self.subtract_last_trgt):
            m = Masker(
                self.band_width,
                self.feature_dim,
                fac_mask=fac_mask,
                mask_ctxt=mask_ctxt,
                fac_out_ch=nb_channels if self.stereonaive else 1,
                eps=eps,
            )
            self.maskers.append(m)

    def forward(self, input):
        # input: [bsize, nch, n_samples]

        bsize, nch, nsamples = input.shape
        B = bsize * nch
        input = input.view(B, -1)

        # STFT, real and imag parts
        spec = self.stft(input)  # [B, F, T]
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # [B, 2, F, T]

        # Split into subbands
        subband_mix_spec = []
        ib = 0
        for i in range(len(self.band_width)):
            sb = spec_RI[
                ..., ib : ib + self.band_width[i], :
            ].contiguous()  # [B, 2, BW, T]
            subband_mix_spec.append(sb)
            ib += self.band_width[i]

        # Normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):

            if self.stereonaive:
                sb = subband_mix_spec[i].view(
                    bsize, self.band_width[i] * 2 * self.nb_channels, -1
                )
            else:
                sb = subband_mix_spec[i].view(B, self.band_width[i] * 2, -1)

            sb = self.BN[i](sb)
            subband_feature.append(sb)
        subband_feature = torch.stack(subband_feature, 1)  # [B', nband, N, T]

        # Separator
        sep_output = self.separator(subband_feature)  # [B', nband, N, T]

        # Masker
        est_specs = []
        for masker in self.maskers:
            est_spec_cplx = masker(sep_output, subband_mix_spec)  # [B, F, T]
            est_specs.append(est_spec_cplx)
        est_specs = torch.stack(est_specs, 1)  # [B, n_targets', F, T]

        # Get the last source if it has not been estimated by the masker
        if self.subtract_last_trgt:
            last_trgt = spec.unsqueeze(1) - torch.sum(
                est_specs, dim=1, keepdim=True
            )  # B, 1, F, T
            est_specs = torch.cat((est_specs, last_trgt), dim=1)

        # Add "n_targets" dimension if needed
        if self.n_targets == 1:
            est_specs = est_specs.unsqueeze(1)

        # Reshape
        est_specs = est_specs.view(
            bsize, nch, self.n_targets, self.enc_dim, -1
        )  # bsize, nch, n_targets, F, T
        est_specs = est_specs.permute(
            0, 2, 1, 3, 4
        ).contiguous()  # bsize, n_targets, nch, F, T

        # iSTFT
        output = self.istft(
            est_specs, length=nsamples
        )  # bsize, n_targets, nch, n_samples

        return {"waveforms": output, "stfts": est_specs}


if __name__ == "__main__":

    cfg_optim = OmegaConf.create(
        {
            "algo": "adam",
            "lr": 0.001,
            "loss_type": "L1",
            "loss_domain": "t",
            "monitor_val": "sdr",
        }
    )
    cfg_scheduler = OmegaConf.create({"name": "plateau", "factor": 0.5, "patience": 3})
    cfg_eval = OmegaConf.create(
        {
            "device": "cpu",
            "verbose_per_track": True,
            "rec_dir": None,
            "segment_len": 10,
            "overlap": 0.1,
            "hop_size": None,
            "sdr_type": "usdr",
            "sdr_win": 1.0,
            "sdr_hop": 1.0,
        }
    )

    model = BSRNN(
        cfg_optim,
        cfg_scheduler,
        cfg_eval,
        # targets=["vocals", "bass", "drums", "other"],
        targets="vocals",
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        nb_channels=2,
        feature_dim=64,
        joint_bandsplit=False,
        num_repeat=8,
        fac_sep=2,
        time_layer="lstm",
        band_layer="lstm",
        bidirectional=True,
        fac_mask=4,
        mask_ctxt=0,
        subtract_last_trgt=False,
        n_heads=1,
        group_num=None,
        n_att_head=0,
        attn_enc_dim=16,
        stereo="naive",
        fac_tac=3,
        act_tac="tanh",
        eps=1e-7,
    )
    print("Number of parameters:", model.count_params())

    # Example for the forward pass
    x = torch.randn((4, 2, 10000))
    output = model(x)
    print(output["waveforms"].shape, output["stfts"].shape)

# EOF
