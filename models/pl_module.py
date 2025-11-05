import torch
import torchaudio
import lightning.pytorch as pl
import pandas as pd
from os.path import join
from helpers.data import rec_estimates
from helpers.eval import compute_sdr, compute_loss
from omegaconf import OmegaConf
from helpers.transforms import mySTFT, myISTFT


# Meta class that will have all the training/val/test-related methods and attributes
class PLModule(pl.LightningModule):
    def __init__(
        self,
        cfg_optim,
        cfg_scheduler,
        cfg_eval,
        targets=["vocals", "bass", "drums", "other"],
        sample_rate=44100,
        n_fft=2048,
        n_hop=512,
        eps=1.0e-7,
    ):
        super().__init__()

        # General useful parameters
        self.sample_rate = sample_rate
        self.eps = eps
        self.targets = targets
        if isinstance(targets, str):
            self.targets = [targets]

        # Training/optimizer attributes
        self.optim_algo = cfg_optim.algo
        self.lr = cfg_optim.lr
        self.loss_type = cfg_optim.loss_type
        self.loss_domain = cfg_optim.loss_domain
        self.monitor_val = cfg_optim.monitor_val

        # Scheduler-related attributes
        self.cfg_scheduler = cfg_scheduler

        # Evaluation (valid/test) attributes
        self.eval_device = cfg_eval.device
        self.eval_segment_len = cfg_eval.segment_len
        self.eval_overlap = cfg_eval.overlap
        self.eval_hop_size = cfg_eval.hop_size
        self.verbose_per_track = cfg_eval.verbose_per_track
        self.rec_dir = cfg_eval.rec_dir

        # SDR-related
        self.sdr_type = cfg_eval.sdr_type
        self.sdr_win = int(cfg_eval.sdr_win * sample_rate)
        self.sdr_hop = int(cfg_eval.sdr_hop * sample_rate)

        # Transforms
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.stft = mySTFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = myISTFT(n_fft=n_fft, n_hop=n_hop)

        # Initialize val/test results storage
        self.test_sdr_temp = []
        self.test_results = None

        self.save_hyperparameters()

    # dummy forward function, just to ensure proper output size
    def forward(self, input):
        # input: [batch_size, nb_channels, n_samples]
        y_hat = input.unsqueeze(1).repeat(
            1, len(self.targets), 1, 1
        )  # [batch_size, nb_targets, nb_channels, n_samples]
        return {"waveforms": y_hat}

    def _shared_step(self, x, y=None, comp_loss=True):
        # Apply model, and get the waveform
        outputs = self(x)
        y_hat = outputs["waveforms"]

        # If true sources are not provided, do not compute the loss
        if y is None:
            comp_loss = False

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

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        train_loss, _ = self._shared_step(x, y)

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
        bsize = x.shape[0]

        # Get the estimates and validation loss
        y_hat, val_loss = self._apply_model_to_track(x, y, comp_loss=True)

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=bsize,
        )

        # Validation SDR
        val_sdr_trgts = compute_sdr(
            y,
            y_hat,
            win=self.sdr_win,
            hop=self.sdr_hop,
            type=self.sdr_type,
            eps=self.eps,
        )
        val_sdr = torch.nanmean(val_sdr_trgts)  # mean over targets

        self.log(
            "val_sdr",
            val_sdr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bsize,
        )

        # if there are many targets, log per-source SDR for monitoring
        if len(self.targets) > 1:
            for it, t in enumerate(self.targets):
                self.log(
                    "val_sdr_" + t,
                    val_sdr_trgts[it],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=bsize,
                )

        return val_loss, val_sdr

    def _apply_model_to_track(self, mix, true_sources=None, comp_loss=True):

        # Apply either the fader or OLA, depending if hop size is provided
        if self.eval_hop_size is None:
            output_sig = self._apply_model_to_track_fader(
                mix, true_sources, comp_loss=comp_loss
            )
        else:
            output_sig = self._apply_model_to_track_ola(
                mix, true_sources, comp_loss=comp_loss
            )

        return output_sig

    def _apply_model_to_track_ola(self, mix, true_sources=None, comp_loss=True):

        # mix: [B, num_channels, n_samples]
        # true_sources: [B, num_sources, num_channels, n_samples]

        # If true sources are not provided, do not compute the loss
        if true_sources is None:
            comp_loss = False

        # Place model on eval device
        self.to(self.eval_device)

        # Get the input signal device
        in_device = mix.device

        # Size and parameters
        bsize, nb_channels, nsamples = mix.shape
        eval_frames = int(self.sample_rate * self.eval_segment_len)
        hop_frames = int(self.sample_rate * self.eval_hop_size)
        fact_ol = self.eval_hop_size / self.eval_segment_len

        if self.targets is not None:
            n_targets = len(self.targets)
        else:
            n_targets = 1

        # treat the case where we might want to process the whole track at once
        if self.eval_segment_len == -1:
            chunk_len = nsamples + 1
        else:
            chunk_len = eval_frames

        # if too big chunks / too small mixture, ignore the OLA
        if chunk_len >= nsamples:
            # move to eval device
            mix = mix.to(self.eval_device)
            if true_sources is not None:
                true_sources = true_sources.to(self.eval_device)
            # apply model
            with torch.no_grad():
                loss, y_hat = self._shared_step(mix, true_sources, comp_loss=comp_loss)
            # back to the initial device
            final = y_hat.to(in_device)

        else:
            loss = []

            # add 0s at the begining and end
            mix = torch.cat(
                (
                    torch.zeros((bsize, nb_channels, eval_frames), device=in_device),
                    mix,
                    torch.zeros((bsize, nb_channels, eval_frames), device=in_device),
                ),
                dim=-1,
            )

            # init output sig (same as mix but with n_targets)
            final = torch.zeros(
                (bsize, n_targets, nb_channels, nsamples + 2 * eval_frames),
                device=in_device,
            )

            start = 0
            while start < nsamples + eval_frames:
                # extract current mix chunk and move to eval device
                mix_chunk = mix[..., start : start + eval_frames]
                mix_chunk = mix_chunk.to(self.eval_device)

                # Chunk the true sources and move to eval device
                if true_sources is not None:
                    true_sources_chunk = true_sources[..., start : start + eval_frames]
                    true_sources_chunk = true_sources_chunk.to(self.eval_device)
                else:
                    true_sources_chunk = None

                # apply model
                with torch.no_grad():
                    loss_chunk, y_hat = self._shared_step(
                        mix_chunk, true_sources_chunk, comp_loss=comp_loss
                    )
                # back to the initial device
                y_hat = y_hat.to(in_device)

                # OLA
                final[:, :, :, start : start + eval_frames] += y_hat * fact_ol

                # Store loss over chunks
                loss.append(loss_chunk)

                # update indices
                start += hop_frames

            # Aggregate loss over chunks
            if comp_loss:
                loss = torch.median(torch.stack(loss))
            else:
                loss = None

        # remove the meaningless samples at the beg and end
        final = final[..., eval_frames:-eval_frames]

        return final, loss

    def _apply_model_to_track_fader(self, mix, true_sources=None, comp_loss=True):

        # mix: [B, num_channels, n_samples]
        # true_sources: [B, num_sources, num_channels, n_samples]

        # If true sources are not provided, do not compute the loss
        if true_sources is None:
            comp_loss = False

        # Place model on eval device
        self.to(self.eval_device)

        # Get the input signal device
        in_device = mix.device

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

        # if too big chunks / too small mixture, ignore the fader
        if chunk_len >= nsamples:
            # move to eval device
            mix = mix.to(self.eval_device)
            if true_sources is not None:
                true_sources = true_sources.to(self.eval_device)
            # apply model
            with torch.no_grad():
                loss, y_hat = self._shared_step(mix, true_sources, comp_loss=comp_loss)
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
                # extract current mix chunk and move to eval device
                mix_chunk = mix[..., start:end]
                mix_chunk = mix_chunk.to(self.eval_device)

                # Chunk the true sources and move to eval device
                if true_sources is not None:
                    true_sources_chunk = true_sources[..., start:end]
                    true_sources_chunk = true_sources_chunk.to(self.eval_device)
                else:
                    true_sources_chunk = None

                # apply model
                with torch.no_grad():
                    loss_chunk, y_hat = self._shared_step(
                        mix_chunk, true_sources_chunk, comp_loss=comp_loss
                    )
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
            y,
            y_hat,
            win=self.sdr_win,
            hop=self.sdr_hop,
            type=self.sdr_type,
            eps=self.eps,
        )

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

        if self.optim_algo == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim_algo == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            raise NameError("Unknown optimizer type ")

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
            "monitor": "val_" + self.monitor_val,
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
    cfg_eval = OmegaConf.create(
        {
            "device": "cpu",
            "segment_len": 10,
            "overlap": 0.1,
            "hop_size": None,
            "sdr_type": "usdr",
            "win_dur": 1.0,
            "verbose_per_track": True,
            "rec_dir": None,
        }
    )
    # Instanciate model
    model = PLModule(cfg_optim, cfg_scheduler, cfg_eval)
    model.eval_segment_len = 1

    # Forward pass
    outputs = model(x)
    for key in outputs:
        print("Output: ", key, "-- shape: ", outputs[key].shape)

    # Training (loss)
    train_loss = model.training_step(batch, 0)
    print("Training loss:", train_loss.item())

    # Validation step (loss and SDR)
    val_loss, val_sdr = model.validation_step(batch, 0)
    print("Validation loss:", val_loss.item(), " --- Validation SDR: ", val_sdr)

    # Test step (SDR)
    testsdr = model.test_step(batch, 0)
    print("Test SDR:", testsdr)

# EOF
