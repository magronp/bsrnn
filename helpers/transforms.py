from torch import Tensor
import torch.nn as nn
import torchaudio


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

# EOF