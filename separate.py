import torch
import torchaudio
import hydra
from os.path import join
from omegaconf import DictConfig
from models.separator import Separator
from helpers.data import rec_estimates


@hydra.main(version_base=None, config_name="config", config_path="conf")
def separate(args: DictConfig):

    # Track to separate
    track_path = args.file
    orig_sr = torchaudio.info(track_path).sample_rate

    # Process only part of the track if offset and/or duration are provided
    if args.duration is None:
        nfr = -1
    else:
        nfr = int(args.duration * orig_sr)
    if args.offset is None:
        fr_offst = 0
    else:
        fr_offst = int(args.offset * orig_sr)

    # Load the mix and add batch dim [1, n_channels, n_samples]
    mix = torchaudio.load(track_path, num_frames=nfr, frame_offset=fr_offst)[
        0
    ].unsqueeze(0)

    # Resample the mixture to the target sample rate
    mix = torchaudio.functional.resample(mix, orig_sr, args.sample_rate)

    # Define the folder where the model ckpt is/are located
    args.src_mod.name_out_dir = args.model_dir

    # Load the model
    model = Separator(args)

    # Make sure to use CPU is no GPU is available
    eval_device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.eval.device == "cuda") else "cpu"
    )
    model.eval_device = eval_device

    # Estimate the sources
    max_norm = mix.abs().max()
    mix /= max_norm
    estimates = model._apply_model_to_track(mix)[0]
    mix *= max_norm
    estimates = estimates.squeeze(0)  # remove batch dimension

    # In case of SIMO model, only keep the relevant target estimates
    if args.simo:
        all_targets_sep = model.model.targets
        est_ind = [all_targets_sep.index(t) for t in args.targets]
        estimates = estimates[est_ind]

    # Record estimates
    rec_estimates(estimates, args.rec_dir, args.targets, args.sample_rate)


if __name__ == "__main__":
    separate()

# EOF
