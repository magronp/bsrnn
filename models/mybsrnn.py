"""!
@author Yi Luo (oulyluo)
@copyright Tencent AI Lab
"""

from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from helpers.transforms import mySTFT, myISTFT
from models.pl_module import PLModule


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-7):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.norm = nn.GroupNorm(1, input_size, eps)
        self.rnn = nn.LSTM(
            input_size, hidden_size, 1, batch_first=True, bidirectional=True
        )

        # linear projection layer
        self.proj = nn.Linear(hidden_size * 2, input_size)

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(input.shape[0], input.shape[2], input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


class BSNet(nn.Module):
    def __init__(self, feature_dim, eps=1e-7):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_size = feature_dim * 2

        self.band_rnn = ResRNN(self.feature_dim, self.hidden_size, eps=eps)
        self.band_comm = ResRNN(self.feature_dim, self.hidden_size, eps=eps)

    def forward(self, input):
        # input shape: B, nband, N, T
        B, nband, N, T = input.shape

        # Sequence modeling over time
        input = input.view(B * nband, N, -1)  # [B*nband, N, T]
        band_output = self.band_rnn(input)  # [B*nband, N, T]

        # Sequence modeling over bands, need some reshaping to get to [..., N, nband]
        band_output = band_output.view(B, nband, -1, T)  # [B, nband, N, T]
        band_output = band_output.permute(0, 3, 2, 1).contiguous()  # [B, T, N, nband]
        band_output = band_output.view(B * T, -1, nband)  # [B*T, N, nband]

        output = self.band_comm(band_output)  # [B*T, N, nband]

        output = (
            output.view(B, T, -1, nband).permute(0, 3, 2, 1).contiguous()
        )  # [B, nband, N, T]

        return output


class myBSRNN(PLModule):
    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        target="vocals",
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        feature_dim=128,
        num_repeat=12,
        eps=1e-7,
        *args,
        **kwargs
    ):

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            targets=target,
            sample_rate=sample_rate,
            eps=eps,
            module_type='time'
        )

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.enc_dim = self.n_fft // 2 + 1
        self.feature_dim = feature_dim
        self.num_repeat = num_repeat
        
        # Transforms
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)

        # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_50 = int(np.floor(50 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sample_rate / 2.0) * self.enc_dim))

        # Define the frequencies bandwidth per instrument
        if target == "vocals" or target == "other":
            self.band_width = [bandwidth_100] * 10
            self.band_width += [bandwidth_250] * 12
            self.band_width += [bandwidth_500] * 8
            self.band_width += [bandwidth_1k] * 8
            self.band_width += [bandwidth_2k] * 2
        elif target == "bass":
            self.band_width = [bandwidth_50] * 10
            self.band_width += [bandwidth_100] * 5
            self.band_width += [bandwidth_500] * 6
            self.band_width += [bandwidth_1k] * 4
            self.band_width += [bandwidth_2k] * 4
        elif target == "drums":
            self.band_width = [bandwidth_50] * 20
            self.band_width += [bandwidth_100] * 10
            self.band_width += [bandwidth_250] * 8
            self.band_width += [bandwidth_500] * 8
            self.band_width += [bandwidth_1k] * 8
        else:
            print("Unknown Instrument {}".format(target))
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
            self.separator.append(BSNet(self.feature_dim, eps=self.eps))
        self.separator = nn.Sequential(*self.separator)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1),
                )
            )

    def forward(self, input):

        """
        apparemment time loss marche mieux en overfitting que loss complète...
        V0: time loss, sans norm
        V1: time loss, avec norm
        V2: 
        """

        # input: [B, nch, nsample]
        batch_size, nch, nsample = input.shape 
        
        # STFT and stack into real and imag
        mix_stft = self.stft(input)  # B, nch, F, T
        mix_stft_RI = torch.stack([mix_stft.real, mix_stft.imag], 2)  # B, nch, 2, F, T

        # Normalization
        bmean = mix_stft_RI.mean(dim=(1, 2, 3, 4), keepdim=True)
        bstd = mix_stft_RI.std(dim=(1, 2, 3, 4), keepdim=True)
        bstd[bstd == 0] = 1 # avoid probolem in case of silent chunk...
        mix_stft_RI = (mix_stft_RI - bmean) / (bstd + self.eps)

        # Reshape to assemble batch/channel dims
        mix_stft_RI = mix_stft_RI.view(batch_size * nch, 2, self.enc_dim, -1)  # [B*nch, 2, F, T]

        # split stacked-RI into subbands + normalization and bottleneck
        bandsplit_feature = []
        band_idx = 0
        for i in range(len(self.band_width)):
            current_subband = mix_stft_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous() # [B*nch, 2, BW, T]
            current_subband = current_subband.view(batch_size * nch, self.band_width[i] * 2, -1)  # [B*nch, 2*BW, T]
            current_subband = self.BN[i](current_subband)  # [B*nch, N, T]   c'est ici qu'i faudrait inclure le nch et le mag (après reshape avant)
            bandsplit_feature.append(current_subband)
            band_idx += self.band_width[i]
        bandsplit_feature = torch.stack(bandsplit_feature, 1)  # [B*nch, nband, N, T]
        
        # Separator (band / sequence modeling)
        sep_output = self.separator(bandsplit_feature)  # [B*nch, nband, N, T]

        # Mask estimation
        all_masks = []
        for i in range(len(self.band_width)):
            this_mask = self.mask[i](sep_output[:, i])  # [B*nch, 4*BW, T]
            this_mask = this_mask.view(batch_size * nch, 2, 2, self.band_width[i], -1)  # [B*nch, 2, 2, BW, T]
            this_mask = this_mask[:, 0] * torch.sigmoid(this_mask[:, 1])  # [B*nch, 2, BW, T]
            all_masks.append(this_mask)
        all_masks = torch.cat(all_masks, dim=2) # [B*nch, 2, F, T]  (RI-valued masks)

        # Apply mask
        est_stft_RI = all_masks * mix_stft_RI  # [B*nch, 2, F, T] 

        # Reshape: split batches and channels
        est_stft_RI = est_stft_RI.view(batch_size, nch, 2, self.enc_dim, -1) # [B, nch, 2, F, T] 

        # De-normalize
        est_stft_RI = est_stft_RI * bstd + bmean # [B, nch, 2, F, T] 

        # Back to the complex domain
        est_stft = torch.complex(est_stft_RI[:, :, 0], est_stft_RI[:, :, 1]) # [B, nch, F, T]

        # add extra dimension for the target
        est_stft = est_stft.unsqueeze(1)  # [B, 1, nch, F, T]
        
        # istft
        y_hat = self.istft(est_stft, length=nsample)  # [B, 1, nch, nsample]

        return {'waveforms': y_hat,
               'stfts': est_stft}


if __name__ == "__main__":

    model = myBSRNN(
        sample_rate=44100, n_fft=2048, n_hop=512, feature_dim=16, num_repeat=2, target="vocals"
    )

    # Display number of parameters
    print("Number of parameters:", model.count_params())

    # Test
    x = torch.randn((4, 2, 44100*20))
    y = torch.randn((4, 1, 2, 44100*20))

    outputs = model(x)
    print(outputs['waveforms'].shape)
    print(outputs['stfts'].shape)

    batch = (x, y, 'tg')
    train_loss = model.training_step(batch, 0)

    model.eval_device = 'cpu'
    val_loss = model.validation_step(batch, 0)
    val_loss_old = model.validation_step_old(batch, 0)

# EOF
