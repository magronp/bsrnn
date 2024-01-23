import torch
import torch.nn.functional as F
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from openunmix.utils import bandwidth_to_max_bin, load_target_models
from models.pl_module import PLModule
from helpers.transforms import mySTFT, myISTFT
from omegaconf import OmegaConf


class UMX(PLModule):
    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        sample_rate=44100,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        hidden_size=1024,
        bandwidth=None,
        unidirectional=False,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        eps=1e-7,
        *args,
        **kwargs
    ):
        
        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            sample_rate=sample_rate,
            eps=eps,
            module_type='spectro'
        )

        # Out bins is determined by the fft size
        self.nb_output_bins = n_fft // 2 + 1

        # Transforms
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)

        # But UMX only processes some of them (and then expands)
        if bandwidth is not None:
            self.nb_bins = bandwidth_to_max_bin(sample_rate, n_fft, bandwidth)
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size
        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = BatchNorm1d(hidden_size)
        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2
        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )
        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(
            in_features=fc2_hiddensize, out_features=hidden_size, bias=False
        )
        self.bn2 = BatchNorm1d(hidden_size)
        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )
        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        # Mean and std
        if input_mean is not None:
            input_mean = -input_mean[: self.nb_bins]
        else:
            input_mean = torch.zeros(self.nb_bins)
        if input_scale is not None:
            input_scale = 1.0 / input_scale[: self.nb_bins]
        else:
            input_scale = torch.ones(self.nb_bins)
        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)
        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

        self.save_hyperparameters()

    def forward(self, x):
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, _ = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))

        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)
        
        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)
        
        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        x = x.permute(1, 2, 3, 0)

        # Add an extra dimension for the number of target
        x = x.unsqueeze(1)

        return {'magnitudes': x}

    def _load_pretrained(self, target, umxversion="umxl"):
        umxl = load_target_models(target,
                        model_str_or_path=umxversion,
                        device="cpu",
                        pretrained=True)[target]
        self.fc1 = umxl.fc1
        self.fc2 = umxl.fc2
        self.fc3 = umxl.fc3
        self.bn1 = umxl.bn1
        self.bn2 = umxl.bn2
        self.bn3 = umxl.bn3
        self.lstm = umxl.lstm
        self.input_mean = umxl.input_mean
        self.input_scale = umxl.input_scale
        self.output_mean = umxl.output_mean
        self.output_scale = umxl.output_scale


if __name__ == "__main__":
    cfg_optim = OmegaConf.create({"lr": 0.001, "loss_type": "L1", "loss_domain": "t"})
    cfg_scheduler = OmegaConf.create({"name": "plateau", "factor": 0.5, "patience": 3})

    # Instanciate model and display number of parameters
    model = UMX(cfg_optim, cfg_scheduler)
    print("Number of parameters:", model.count_params())

    # Test tensor
    x = torch.randn((4, 2, 10000))
    y = torch.randn((4, 1, 2, 10000))
    track_name = ['dummy_track']
    batch = (x, y, track_name)

    # Simulate STFT magnitude
    X = torch.abs(model.stft(x))
    print(X.shape)

    # Forward pass
    output = model(X)['magnitudes']
    print(output.shape)

    # Training and validation steps
    tr_loss = model.training_step(batch, 0)
    model.eval_device = 'cpu'
    val_loss = model.validation_step(batch, 0)

    # Load pretrained UMXL params
    model._load_pretrained('vocals')

# EOF

