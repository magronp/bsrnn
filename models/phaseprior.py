import torch
import torch.nn as nn
from helpers.transforms import mySTFT, myISTFT
from models.pl_module import PLModule
from omegaconf import OmegaConf
import lightning.pytorch as pl


# Phase model
class PhaseModel(PLModule):
    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        target="vocals",
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        hidden_size=256,
        num_conv_blocks=6,
        cutoff_freq_band=None,
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
            module_type='phase'
        )

        # Transforms
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)

        # Number of frequency channels
        self.n_freqs = n_fft // 2 + 1
        if cutoff_freq_band is not None:
            self.n_freqs = cutoff_freq_band

        self.input_feature_dim = self.n_freqs * 3
        self.output_feature_dim = self.n_freqs * 2
        self.hidden_size = hidden_size

        # Linear layers
        self.in_layer = nn.Linear(self.input_feature_dim, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, self.output_feature_dim)

        # Convolutional blocks
        self.conv_block_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Conv1d(
                        hidden_size, hidden_size, groups=hidden_size, kernel_size=5, padding=2
                    ),
                )
                for _ in range(num_conv_blocks)
            ]
        )

    def forward(self, X, est_mag):
        # X: [B, nch, F, T]
        # est_mag: [B, 1, nch, F, T] 
        batch_size, nb_channels, _, _ = X.shape

        # Prepare the inputs and crop the frequency if needed
        est_mag = est_mag[..., :self.n_freqs, :]
        est_mag = est_mag.squeeze(1) # [B, nch, F, T]

        # Normalize the magnitude
        est_mag = est_mag.reshape(batch_size, -1)
        est_mag /= est_mag.max(1, keepdim=True)[0] + self.eps
        est_mag = est_mag.view(batch_size, nb_channels, self.n_freqs, -1)

        cosX = torch.cos(torch.angle(X)[..., :self.n_freqs, :]) # [B, nch, F, T]
        sinX = torch.sin(torch.angle(X)[..., :self.n_freqs, :]) # [B, nch, F, T]

        # Concatenate the various input features
        out = torch.cat([est_mag, cosX, sinX], dim=2)

        # Reshape and Linear layer
        out = out.view(batch_size * nb_channels, self.input_feature_dim, -1)  # [B*nch, 3F, T]
        out = out.permute(0, 2, 1)  # [B*nch, T, 3F]
        out = self.in_layer(out)  # [B*nch, T, N]

        # Reshape and Conv blocks
        out = out.permute(0, 2, 1)  # [B*nch, N, T]
        for conv_block in self.conv_block_list:
            out = conv_block(out) # [B*nch, N, T]

        # Output layer
        out = out.permute(0, 2, 1)  # [B*nch, T, N]
        out = self.out_layer(out) # [B*nch, T, 2F]

        # Reshape and residual (add mixture phase)
        out = out.view(batch_size, nb_channels, -1, self.n_freqs, 2)  # [B, nch, T, F, 2]
        out = out.permute(0, 1, 3, 2, 4)  # [B, nch, F, T, 2]
        out = out + torch.stack([cosX, sinX], dim=-1) # [B, nch, F, T, 2]

        # L2 normalization (ensures it's a cos/sin), and get angle
        out = out / torch.linalg.norm(out, dim=-1, keepdim=True)
        out = torch.angle(out[..., 0] + 1j * out[..., 1]) # [B, nch, F, T]

        # Combine the estimated phase with the mixture's phase (higher frequencies)
        phi = torch.clone(torch.angle(X))
        phi[..., :self.n_freqs, :] = out # [B, nch, F, T]

        # Expand for the target dimension
        phi = phi.unsqueeze(1) # [B, 1, nch, F, T]

        return {'phases': phi}


if __name__ == "__main__":

    pl.seed_everything(1234, workers=True)
    cfg_optim = OmegaConf.create({"lr": 0.001, "loss_type": "L1", "loss_domain": "t"})
    cfg_scheduler = OmegaConf.create({"name": "plateau", "factor": 0.5, "patience": 3})

    # Data
    n_fft = 2048
    n_hop = 512
    n_freqs = n_fft // 2 + 1
    batch_size, nb_targets, nb_channels, n_samples = 4, 1, 2, 10000
    x = torch.randn((batch_size, nb_channels, n_samples))
    y = torch.randn((batch_size, nb_targets, nb_channels, n_samples))

    # Transforms
    stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
    X = stft(x)
    Y = stft(y)
    est_mag = torch.abs(Y)

    # Apply model
    model = PhaseModel(cfg_optim, cfg_scheduler, cutoff_freq_band=128)
    print('Number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        
    est_phase = model(X, est_mag)

# EOF
