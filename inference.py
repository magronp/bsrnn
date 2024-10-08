import hydra
from omegaconf import DictConfig
import tqdm
from models.separator import Separator
import torchaudio
import torch
from helpers.data import rec_estimates

tqdm.monitor_interval = 0


@hydra.main(version_base=None, config_name="config", config_path="conf")
def inference(args: DictConfig):

    # Additional argument: path to the track to separate
    track_path = args.file_path

    # Process only part of the track: offset and max duration
    if args.max_len is None:
        nfr = -1
    else:
        nfr = int(args.max_len * args.sample_rate)
    if args.offset is None:
        fr_offst = 0
    else:
        fr_offst = int(args.offset * args.sample_rate)

    # Load the mix and add batch dim [1, n_channels, n_samples]
    mix = torchaudio.load(track_path, num_frames=nfr, frame_offset=fr_offst)[
        0
    ].unsqueeze(0)

    # Define the folder where the model ckpt are located
    args.src_mod.name_out_dir = args.model_dir

    # Load the model
    model = Separator(args)

    # Make sure to use CPU is no GPU is available
    eval_device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.eval.device == "cuda") else "cpu"
    )
    model.eval_device = eval_device

    # Test step : estimate the sources and get the SDR
    estimates = model._apply_model_to_track(mix)[0]

    # Record estimates
    rec_estimates(estimates[0], args.rec_dir, args.targets, args.sample_rate)


if __name__ == "__main__":
    inference()

# EOF
