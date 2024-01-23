import torch
from torch import Tensor
import torch.nn as nn
import torchaudio


class EncMag(nn.Module):
    def __init__(self, n_fft=4096, n_hop=1024, window=None, nb_channels=2):
        super().__init__()

        self.mono = nb_channels == 1
        #self.stft = mySTFT(n_fft, n_hop, window)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=n_hop, power=None)

    def forward(self, input: Tensor) -> Tensor:
        # take the stft magnitude
        spec = torch.abs(self.stft(input))

        # downmix in the mag domain to preserve energy
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)

        return spec


class mySTFT(nn.Module):
    def __init__(self, n_fft=4096, n_hop=1024):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.transf = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=n_hop, power=None)

    def forward(self, input: Tensor) -> Tensor:
        return self.transf(input)


class myISTFT(nn.Module):
    def __init__(self, n_fft=4096, n_hop=1024):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.transf = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=n_hop)

    def forward(self, input: Tensor, length=None) -> Tensor:
        return self.transf(input, length)



# Old implementation not good because use a default rectangular zeros (all one)
class mySTFT_old(nn.Module):
    def __init__(self, n_fft=4096, n_hop=1024):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop

    def forward(self, input: Tensor) -> Tensor:

        # reshape the input in case there are multiples source / channels /batch elements
        shape = input.size()
        input = input.view(-1, shape[-1])
        stft_tensor = torch.stft(
            input,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            return_complex=True,
        )
        # unpack batch
        stft_tensor = stft_tensor.view(shape[:-1] + stft_tensor.shape[-2:])


        return stft_tensor


class myISTFT_old(nn.Module):
    def __init__(self, n_fft=4096, n_hop=1024):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop

    def forward(self, input: Tensor, length=None) -> Tensor:

        # reshape the input in case there are multiples source / channels /batch elements
        shape = input.size()
        input = input.view(-1, shape[-2], shape[-1])

        output = torch.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            length=length,
        )

        # unpack batch
        output = output.view(shape[:-2] + output.shape[-1:])

        return output
    
# EOF
