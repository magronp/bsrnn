from omegaconf import OmegaConf
import torch
import torch.nn as nn
from models.pl_module import PLModule
from helpers.transforms import mySTFT, myISTFT
from models.attention import AttentionBlock
from models.bs import get_bandsplit, Masker


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


class BSNetCNN(nn.Module):
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
        targets="vocals",
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        feature_dim=128,
        num_repeat=12,
        time_layer={"n_dil_conv": 2, "ks": 3, "hs_fac": 2},
        band_layer={"n_dil_conv": 2, "ks": 3, "hs_fac": 2},
        fac_mask=4,
        mask_ctxt=0,
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
            targets=targets,
            sample_rate=sample_rate,
            eps=eps,
        )

        self.sr = sample_rate
        self.win = n_fft
        self.stride = n_hop
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.num_repeat = num_repeat

        # Transforms
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)

        # Band split module
        self.band_width = get_bandsplit(sample_rate, self.enc_dim)
        self.nband = len(self.band_width)
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
                BSNetCNN(
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
        self.mask_ctxt = mask_ctxt
        self.fac_mask = fac_mask
        self.masker = Masker(
            self.band_width,
            self.feature_dim,
            fac_mask=self.fac_mask,
            mask_ctxt=self.mask_ctxt,
            eps=eps,
        )

    def forward(self, input):
        # input shape: (bsize, nch, n_samples)

        bsize, nch, nsample = input.shape
        input = input.view(bsize * nch, -1)

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
            )  # B, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](
                    subband_spec[i].view(bsize * nch, self.band_width[i] * 2, -1)
                )
            )
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        # Separator
        sep_output = self.separator(subband_feature)

        # Masker
        est_spec = self.masker(sep_output, subband_mix_spec)  # B, F, T

        # Reshape and add "n_targets" dimension
        est_spec = est_spec.view(bsize, nch, self.enc_dim, -1)  # bsize, nch, F, T
        est_spec = est_spec.unsqueeze(1)  # bsize, 1, nch, F, T

        # istft
        output = self.istft(est_spec, length=nsample)  # bsize, 1, nch, n_samples

        return {"waveforms": output, "stfts": est_spec}


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
