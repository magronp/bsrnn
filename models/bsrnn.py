"""!
@author Yi Luo (oulyluo)
@copyright Tencent AI Lab

J'ai ajouté la stft/istft, le module PL, et outputer est_spec en plus de la waveform
d'ailleurs leur version n'est pas appropriée pour la loss TF+TD... Donc soit ils mythonnes, soit ils recalculent la stft
hors de la fonction, donc c'est comme si yavait consistance...
"""


from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from models.pl_module import PLModule
from helpers.transforms import mySTFT, myISTFT


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps
        
        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=True)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*2, input_size)

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1,2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape[0], input.shape[2], input.shape[1])
        
        return input + rnn_output.transpose(1,2).contiguous()

class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband

        self.band_rnn = ResRNN(self.feature_dim, self.feature_dim*2)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim*2)

    def forward(self, input):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        band_output = self.band_rnn(input.view(B*self.nband, self.feature_dim, -1)).view(B, self.nband, -1, T)

        # band comm
        band_output = band_output.permute(0,3,2,1).contiguous().view(B*T, -1, self.nband)
        output = self.band_comm(band_output).view(B, T, -1, self.nband).permute(0,3,2,1).contiguous()

        return output.view(B, N, T)


class BSRNN(PLModule):

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

        instrument = target
        sr = sample_rate
        self.sr = sr
        self.win = n_fft
        self.stride = n_hop
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim

        # Transforms
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)
        
        # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_50 = int(np.floor(50 / (sr / 2.) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sr / 2.) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sr / 2.) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.) * self.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (sr / 2.) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.) * self.enc_dim))

        if instrument == 'vocals' or instrument == 'other':
            self.band_width = [bandwidth_100]*10
            self.band_width += [bandwidth_250]*12
            self.band_width += [bandwidth_500]*8
            self.band_width += [bandwidth_1k]*8
            self.band_width += [bandwidth_2k]*2
        elif instrument == 'bass':
            self.band_width = [bandwidth_50]*10
            self.band_width += [bandwidth_100]*5
            self.band_width += [bandwidth_500]*6
            self.band_width += [bandwidth_1k]*4
            self.band_width += [bandwidth_2k]*4
        elif instrument == "drums":
            self.band_width = [bandwidth_50]*20
            self.band_width += [bandwidth_100]*10
            self.band_width += [bandwidth_250]*8
            self.band_width += [bandwidth_500]*8
            self.band_width += [bandwidth_1k]*8
        else: 
            print("Unknown Instrument {}".format(instrument))
            raise NotImplementedError

        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(nn.GroupNorm(1, self.band_width[i]*2, self.eps),
                                         nn.Conv1d(self.band_width[i]*2, self.feature_dim, 1)
                                        )
                          )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(BSNet(self.nband*self.feature_dim, self.nband))             
        self.separator = nn.Sequential(*self.separator)
        
        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(nn.Sequential(nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                                           nn.Conv1d(self.feature_dim, self.feature_dim*4, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim*4, self.feature_dim*4, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim*4, self.band_width[i]*4, 1)
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
        input = input.view(batch_size*nch, -1)

        spec = self.stft(input)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:,:,band_idx:band_idx+self.band_width[i]].contiguous())
            subband_mix_spec.append(spec[:,band_idx:band_idx+self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(self.BN[i](subband_spec[i].view(batch_size*nch, self.band_width[i]*2, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T
        
        # import pdb; pdb.set_trace()
        # separator
        sep_output = self.separator(subband_feature.view(batch_size*nch, self.nband*self.feature_dim, -1))  # B, nband*N, T
        sep_output = sep_output.view(batch_size*nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:,i]).view(batch_size*nch, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:,0] * torch.sigmoid(this_output[:,1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:,0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:,1]  # B*nch, K, BW, T
            est_spec_real = subband_mix_spec[i].real * this_mask_real - subband_mix_spec[i].imag * this_mask_imag  # B*nch, BW, T
            est_spec_imag = subband_mix_spec[i].real * this_mask_imag + subband_mix_spec[i].imag * this_mask_real  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T
        
        output = self.istft(est_spec, length=nsample)
        output = output.view(batch_size, nch, -1)

        # ajouter pour avoir les bonnes tailles
        est_spec = est_spec.view(batch_size, nch, self.enc_dim, -1)
        est_spec = est_spec.unsqueeze(1)  # B, 1, nch, F, T
        output = output.unsqueeze(1)  # B, 1, nch, n_samples

        return {'waveforms': output,
               'stfts': est_spec}
    
    
if __name__ == '__main__':

    model = BSRNN(sample_rate=44100, n_fft=2048, n_hop=512, feature_dim=16, num_repeat=2, target='vocals')

    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print('# of parameters: '+str(s/1024.0/1024.0))
    
    x = torch.randn((1, 2, 44100*3))
    output = model(x)
    print(output.shape)

    """
    x = torch.randn((2, 44100*3))

    Xbs = torch.stft(x, n_fft=2048, hop_length=512,
                      window=torch.hann_window(2048).to(x.device).type(x.type()),
                      return_complex=True)

    stft = mySTFT(n_fft=2048, n_hop=512)
    Xmy = stft(x)
    torch.linalg.norm(Xbs - Xmy)
    """