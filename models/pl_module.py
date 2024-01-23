import torch
import torchaudio
import lightning.pytorch as pl
import pandas as pd
from os.path import join
from helpers.utils import rec_estimates
from helpers.eval import compute_sdr, compute_loss
from omegaconf import OmegaConf
from helpers.transforms import mySTFT, myISTFT


# Meta class that will have all the training/val/test-related methods and attributes
class PLModule(pl.LightningModule):
    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        targets=["vocals", "bass", "drums", "other"],
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        eval_segment_len=10.0,
        eval_overlap=0.1,
        sdr_type="global",
        win_dur=1.0,
        verbose_per_track=True,
        eval_device="cuda",
        rec_dir=None,
        eps=1.0e-7,
        module_type="time",
    ):
        super().__init__()

        # Module type (define how to perform forward pass) and the corrresponding function
        self.module_type = module_type
        self.shared_step = getattr(self, "_shared_step_" + module_type)

        # General useful parameters
        self.sample_rate = sample_rate
        self.eps = eps
        self.targets = targets
        if isinstance(targets, str):
            self.targets = [targets]

        # Training/optimizer attributes
        self.lr = cfg_optim.lr
        self.loss_type = cfg_optim.loss_type
        self.loss_domain = cfg_optim.loss_domain

        # Scheduler-related attributes
        self.cfg_scheduler = cfg_scheduler

        # Evaluation (valid/test) attributes
        self.eval_segment_len = eval_segment_len
        self.eval_overlap = eval_overlap
        self.sdr_type = sdr_type
        self.win_bss = int(win_dur * sample_rate)
        self.verbose_per_track = verbose_per_track
        self.eval_device = eval_device
        self.rec_dir = rec_dir

        # Transforms
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)

        # Initialize val/test results storage
        self.test_sdr_temp = []
        self.test_results = None
        self.best_val_sdr = -torch.inf

        self.save_hyperparameters()

    # dummy forward function, just to ensure proper output size
    def forward(self, input, V=None):
        if self.module_type == "time":
            # input: [batch_size, nb_channels, n_samples]
            y_hat = input.unsqueeze(1).repeat(
                1, len(self.targets), 1, 1
            )  # [batch_size, nb_targets, nb_channels, n_samples]
            return {"waveforms": y_hat}

        elif self.module_type == "spectro":
            # input : [batch_size, nb_channels, F, T]
            y_hat = input.unsqueeze(1)  # [batch_size, 1, nb_channels, F, T]
            return {"magnitudes": y_hat}

        elif self.module_type == "phase":
            # input: [batch_size, nb_channels, F, T]
            # V: doesn't matter
            phi = torch.angle(input).unsqueeze(1).repeat(
                1, len(self.targets), 1, 1
            )  # [batch_size, nb_targets, nb_channels, F, T]
            return {"phases": phi}

    def _shared_step_time(self, x, y, comp_loss=True):
        # Apply model, and get the waveform
        outputs = self(x)
        y_hat = outputs["waveforms"]

        # Compute the loss for training and validation (but no need for testing)
        if comp_loss:
            # Stack the reference waveform and STFT into a dict
            refs = {"waveforms": y, "stfts": self.stft(y)}

            # Compute the loss
            steploss = compute_loss(
                refs, outputs, loss_type=self.loss_type, loss_domain=self.loss_domain
            )
        else:
            steploss = None

        return steploss, y_hat

    def _shared_step_oracle(self, x, y, comp_loss=True):
        # Apply model, and get the waveform
        outputs = self(y)
        y_hat = outputs["waveforms"]
        steploss = None # no need to account for the particular case (comp_loss), since nothing is learned for oracle
        return steploss, y_hat
    
    def _shared_step_spectro(self, x, y, comp_loss=True):
        # Input STFT
        X = self.stft(x)

        # Apply model
        outputs = self(torch.abs(X))

        # Estimate the waveform using the mixture's phase
        y_hat = self.istft(
            outputs["magnitudes"] * torch.exp(1j * torch.angle(X).unsqueeze(1)),
            length=x.shape[-1],
        )

        # Compute the loss for training and validation (but no need for testing)
        if comp_loss:
            # Add the waveform to the outputs dict
            outputs["waveforms"] = y_hat

            # Stack the reference waveform and magnitude into a dict
            refs = {"waveforms": y, "magnitudes": torch.abs(self.stft(y))}

            # Compute the loss
            steploss = compute_loss(
                refs, outputs, loss_type=self.loss_type, loss_domain=self.loss_domain
            )
        else:
            steploss = None

        return steploss, y_hat


    def _shared_step_phase(self, x, y, comp_loss=True):
        # Get input STFT
        X = self.stft(x)
        Y = self.stft(y)
        V, phX = torch.abs(Y), torch.angle(Y)

        # Apply model
        outputs = self(X, V)

        # Estimate the waveform using the provided magnitudes
        y_hat = self.istft(V * torch.exp(1j * outputs["phases"]), length=x.shape[-1])

        # Compute the loss for training and validation (but no need for testing)
        if comp_loss:
            # Add the waveform to the outputs dict
            outputs["waveforms"] = y_hat

            # Stack the reference waveform and STFT mag/phase into a dict
            refs = {"waveforms": y, "phases": phX, "magnitudes": V}

            # Compute the loss
            steploss = compute_loss(
                refs, outputs, loss_type=self.loss_type, loss_domain=self.loss_domain
            )
        else:
            steploss = None
        
        return steploss, y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        train_loss, _ = self.shared_step(x, y)
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        return train_loss

    def on_train_epoch_end(self):
        lr_current = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("lr_epoch", lr_current, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # Load the data
        x, y, _ = batch

        # Get the estimates and validation loss
        y_hat, val_loss = self._apply_model_to_track(x, y, comp_loss=True)

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=x.shape[0],
        )

        # Validation SDR
        val_sdr = compute_sdr(
            y, y_hat, win_bss=self.win_bss, sdr_type=self.sdr_type, eps=self.eps
        )[0]
        val_sdr = torch.nanmean(val_sdr) #in case there are more than one source

        self.log(
            "val_sdr",
            val_sdr,  
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )

        return val_loss, val_sdr
    

    def _apply_model_to_track(self, mix, true_sources, comp_loss=True):

        # mix: [B, num_channels, n_samples]
        # true_sources: [B, num_sources, num_channels, n_samples]

        in_device = mix.device

        # Place model on eval device
        self.to(self.eval_device)

        bsize, nb_channels, nsamples = mix.shape

        if self.targets is not None:
            n_targets = len(self.targets)
        else:
            n_targets = 1

        # treat the case where we might want to process the whole track at once
        if self.eval_segment_len == -1:
            chunk_len = nsamples + 1
        else:
            chunk_len = int(
                self.sample_rate * self.eval_segment_len * (1 + self.eval_overlap)
            )

        # if too big chunks / too small mixture, ignore the OLA / fader
        if chunk_len >= nsamples:
            # move to eval device
            mix.to(self.eval_device)
            true_sources.to(self.eval_device)
            # apply model
            with torch.no_grad():
                loss, y_hat = self.shared_step(mix, true_sources, comp_loss=comp_loss)
            # back to the initial device
            final = y_hat.to(in_device)

        else:
            loss = []
            start = 0
            end = chunk_len
            overlap_frames = (
                self.eval_overlap * self.eval_segment_len * self.sample_rate
            )
            fade = torchaudio.transforms.Fade(
                fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear"
            )

            final = torch.zeros(
                (bsize, n_targets, nb_channels, nsamples), device=in_device
            )

            while start < nsamples - overlap_frames:
                # extract current chunk
                mix_chunk = mix[..., start:end]
                true_sources_chunk = true_sources[..., start:end]
                # move to eval device      self.test_sdr_temp.clear()  # free memory
                mix_chunk.to(self.eval_device)
                true_sources_chunk.to(self.eval_device)
                # apply model
                with torch.no_grad():
                    loss_chunk, y_hat = self.shared_step(mix_chunk, true_sources_chunk, comp_loss=comp_loss)
                # back to the initial device
                y_hat = y_hat.to(in_device)
                # apply fader and add to final signal
                final[:, :, :, start:end] += fade(y_hat)

                # Store loss over chunks
                loss.append(loss_chunk)

                # update indices
                if start == 0:
                    fade.fade_in_len = int(overlap_frames)
                    start += int(chunk_len - overlap_frames)
                else:
                    start += chunk_len
                end += chunk_len
                if end >= nsamples:
                    fade.fade_out_len = 0

            # Aggregate loss over chunks
            if comp_loss:
                loss = torch.median(torch.stack(loss))
            else:
                loss = None

        return final, loss


    def test_step(self, batch, batch_idx):
        # Load the data
        x, y, track_name = batch
        track_name = track_name[0]

        # Get the estimates
        y_hat, _ = self._apply_model_to_track(x, y, comp_loss=False)

        # Test SDR
        sdr = compute_sdr(
            y, y_hat, win_bss=self.win_bss, sdr_type=self.sdr_type, eps=self.eps
        )[0]

        # Arrange SDR into a dict
        testsdr = {}
        testsdr["track"] = track_name
        for j, trg in enumerate(self.targets):
            testsdr[trg] = sdr[j].item()

        # display results
        if self.verbose_per_track:
            print(testsdr)

        # Record the estimates
        if self.rec_dir:
            track_rec_dir = join(self.rec_dir, "test", track_name)
            rec_estimates(y_hat[0], track_rec_dir, self.targets, self.sample_rate)

        # Store it
        self.test_sdr_temp.append(testsdr)

        return testsdr

    def on_test_epoch_end(self):
        # Arrange the results as pd dataframe
        cols = self.targets.copy()
        cols.insert(0, "track")
        test_results = pd.DataFrame(columns=cols)
        for sdr in self.test_sdr_temp:
            test_results.loc[len(test_results)] = sdr

        self.test_sdr_temp.clear()  # free memory
        self.test_results = test_results

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.cfg_scheduler.name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.cfg_scheduler.factor,
                patience=self.cfg_scheduler.patience,
            )
        elif self.cfg_scheduler.name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                gamma=self.cfg_scheduler.gamma,
                step_size=self.cfg_scheduler.step_size,
            )
        else:
            raise NameError("Unknown scheduler type ")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":

    # Test tensors
    x = torch.randn((2, 2, 100000))
    y = torch.randn((2, 4, 2, 100000))
    track_name = ["dummy_track"]
    batch = (x, y, track_name)

    # Params
    cfg_optim = OmegaConf.create({"lr": 0.001, "loss_type": "L1", "loss_domain": "t"})
    cfg_scheduler = OmegaConf.create({"name": "plateau", "factor": 0.5, "patience": 3})
    module_type = 'spectro'

    # Instanciate model
    model = PLModule(cfg_optim, cfg_scheduler, eval_device="cpu", module_type=module_type, verbose_per_track=False)
    model.eval_segment_len = 1

    # Forward pass
    outputs = model(x) 
    for key in outputs:
        print('Output: ', key, '-- shape: ', outputs[key].shape)

    # Training (loss)
    train_loss = model.training_step(batch, 0)
    print('Training loss:', train_loss.item())

    # Validation step (loss and SDR)
    val_loss, val_sdr = model.validation_step(batch, 0)
    print('Validation loss:', val_loss.item(), ' --- Validation SDR: ', val_sdr)

    # Test step (SDR)
    testsdr = model.test_step(batch, 0)
    print('Test SDR:', testsdr)

# EOF
