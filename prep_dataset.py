import torch
import torchaudio
import typing as tp
from pathlib import Path
import typing as tp
from tqdm import tqdm
import pandas as pd
from os.path import join
from helpers.data import get_track_list
import hydra
from omegaconf import DictConfig


class SAD:
    """
    SAD(Source Activity Detector)
    taken from: https://github.com/amanteur/BandSplitRNN-Pytorch/blob/main/src/data/preprocessing.py
    """

    def __init__(
        self,
        sample_rate: int,
        sad_win_size: int = 6,
        sad_overlap_ratio: float = 0.5,
        n_chunks_per_segment: int = 10,
        gamma: float = 1e-3,
        threshold_max_quantile: float = 0.15,
        threshold_segment: float = 0.5,
        eps: float = 1e-5,
    ):
        self.sample_rate = sample_rate
        self.n_chunks_per_segment = n_chunks_per_segment
        self.eps = eps
        self.gamma = gamma
        self.threshold_max_quantile = threshold_max_quantile
        self.threshold_segment = threshold_segment

        self.window_size = int(sample_rate * sad_win_size)
        self.step_size = int(self.window_size * sad_overlap_ratio)

    def chunk(self, y: torch.Tensor):
        """
        Input shape: [n_channels, n_frames]
        Output shape: []
        """
        y = y.unfold(-1, self.window_size, self.step_size)
        y = y.chunk(self.n_chunks_per_segment, dim=-1)
        y = torch.stack(y, dim=-2)
        return y

    @staticmethod
    def calculate_rms(y: torch.Tensor):
        """Calculates torch tensor rms from audio signal
        RMS = sqrt(1/N * sum(x^2))
        """
        y_squared = torch.pow(y, 2)  # need to square signal before mean and sqrt
        y_mean = torch.mean(torch.abs(y_squared), dim=-1, keepdim=True)
        y_rms = torch.sqrt(y_mean)
        return y_rms

    def calculate_thresholds(self, rms: torch.Tensor):
        """ """
        rms[rms == 0.0] = self.eps
        rms_threshold = torch.quantile(
            rms,
            self.threshold_max_quantile,
            dim=-2,
            keepdim=True,
        )
        rms_threshold[rms_threshold < self.gamma] = self.gamma
        rms_percentage = torch.mean(
            (rms > rms_threshold).float(),
            dim=-2,
            keepdim=True,
        )
        rms_mask = torch.all(rms_percentage > self.threshold_segment, dim=0).squeeze()
        return rms_mask

    def calculate_salient(self, y: torch.Tensor, mask: torch.Tensor):
        """ """
        y = y[:, mask, ...]
        C, D1, D2, D3 = y.shape
        y = y.view(C, D1, D2 * D3)
        return y

    def __call__(
        self, y: torch.Tensor, segment_saliency_mask: tp.Optional[torch.Tensor] = None
    ):
        """
        Stacks signal into segments and filters out silent segments.
        :param y: Input signal.
            Shape [n_channels, n_frames]
               segment_saliency_mask: Optional precomputed mask
            Shape [n_channels, n_segments, 1, 1]
        :return: Salient signal folded into segments of length 'self.window_size' and step 'self.step_size'.
            Shape [n_channels, n_segments, frames_in_segment]
        """
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        if segment_saliency_mask is None:
            segment_saliency_mask = self.calculate_thresholds(rms)
        y_salient = self.calculate_salient(y, segment_saliency_mask)
        return y_salient, segment_saliency_mask

    def calculate_salient_indices(self, y: torch.Tensor):
        """
        Returns start indices of salient regions of audio
        """
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        mask = self.calculate_thresholds(rms)
        indices = torch.arange(mask.shape[-1])[mask] * self.step_size
        return indices.tolist()


def get_indices_trg(
    target,
    subset,
    split,
    input_dir,
    output_dir,
    sad,
) -> None:
    # initialize pd data frame to store the results
    all_ind = pd.DataFrame(columns=["track", "ind_beg", "ind_end"])

    # get the list of tracks
    list_tracks = get_track_list(input_dir, subset=subset, split=split)

    for track_name in tqdm(list_tracks):
        # load the audio
        track_dir = join(input_dir, subset, track_name)
        src_path = join(track_dir, target + ".wav")
        y = torchaudio.load(src_path)[0]

        # find indices of salient segments
        indices = sad.calculate_salient_indices(y)

        # store the results in the df
        for curr_ind_beg in indices:
            curr_row = {
                "track": track_name,
                "ind_beg": curr_ind_beg,
                "ind_end": curr_ind_beg + sad.window_size,
            }
            all_ind.loc[len(all_ind)] = curr_row

    # Record the df as a .csv file
    subset_name = "test" if subset == "test" else split
    file_path = output_dir / f"{target}_{subset_name}.csv"
    all_ind.to_csv(file_path)

    return None


@hydra.main(version_base=None, config_name="config", config_path="conf")
def prepare_dset(args: DictConfig):
    # Define subset to process and targets
    subset = "train"
    split = "valid"
    targets = ["vocals", "bass", "drums", "other"]
    args = args.dset

    # initialize Source Activity Detector
    sad = SAD(
        sample_rate=args.sample_rate,
        sad_win_size=args.sad_win_size,
        sad_overlap_ratio=args.sad_overlap_ratio,
        n_chunks_per_segment=args.sad_n_chunks_per_segment,
        gamma=args.sad_gamma,
        threshold_max_quantile=args.threshold_max_quantile,
        threshold_segment=args.threshold_segment,
    )

    # initialize directories where to save indices
    input_dir = args.data_dir
    output_dir = Path(args.sad_dir)
    output_dir.mkdir(exist_ok=True)

    # get active indices for all targets
    for target in targets:
        print(target)
        get_indices_trg(target, subset, split, input_dir, output_dir, sad)

    return


if __name__ == "__main__":
    prepare_dset()

# EOF
