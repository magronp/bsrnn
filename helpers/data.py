import torch
from torch.utils.data import Dataset
import torchaudio
import random
import musdb
from typing import Optional
import tqdm
import os
import yaml
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

tqdm.monitor_interval = 0


def rec_estimates(estimates, track_rec_dir, targets, sample_rate):
    """
    estimates: [n_targets, n_channels, n_samples]
    """

    # Make sure the estimates tensor is detached and on cpu
    estimates = estimates.cpu().detach()

    # create the rec folder if needed
    Path(track_rec_dir).mkdir(parents=True, exist_ok=True)

    # Loop over targets
    for ind_trg, trg in enumerate(targets):
        torchaudio.save(
            os.path.join(track_rec_dir, trg + ".wav"),
            estimates[ind_trg],
            sample_rate,
        )

    return


class Augmentator(object):
    def __init__(
        self,
        transforms: list = [],
        seq_duration: float = 3.0,
        sample_rate: int = 44100,
        min_gain: float = -10.0,
        max_gain: float = 10.0,
        p_silent: float = 0.1,
        p_channelswap: float = 0.5,
    ):
        self.transforms = transforms
        self.seq_dur_samples = int(seq_duration * sample_rate)
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p_silent = p_silent
        self.p_channelswap = p_channelswap

    def _augment_randomcrop(self, audio: torch.Tensor) -> torch.Tensor:
        """Select a random chunk from the provided segment"""
        L = audio.shape[-1]
        if self.seq_dur_samples < L:
            ofst = random.randint(0, L - self.seq_dur_samples)
            return audio[..., ofst : ofst + self.seq_dur_samples]
        else:
            return audio

    def _augment_rescale_db(self, audio: torch.Tensor) -> torch.Tensor:
        """Applies a random gain between `min_gain` and `max_gain` (in dB)"""
        g = self.min_gain + torch.rand(1) * (self.max_gain - self.min_gain)
        g = 10 ** (g / 20)
        return audio * g

    def _augment_channelswap(self, audio: torch.Tensor) -> torch.Tensor:
        """Swap channels of stereo signals with a probability p"""
        if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < self.p_channelswap:
            return torch.flip(audio, [0])
        else:
            return audio

    def _augment_silentsource(self, audio: torch.Tensor) -> torch.Tensor:
        """Silent source with a probability p"""
        if torch.tensor(1.0).uniform_() < self.p_silent:
            return torch.zeros_like(audio)
        else:
            return audio

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            aug = getattr(self, "_augment_" + t)
            audio = aug(audio)

        return audio


def get_track_list(root, subset, split=None):
    # get the list of tracks in the subset
    list_tracks_subset = os.listdir(root + subset)

    # for the training subset, need to keep or filter out val tracks
    if subset == "train":
        # list of validation tracks (predefined in musdb)
        setup_path = os.path.join(musdb.__path__[0], "configs", "mus.yaml")
        with open(setup_path, "r") as f:
            list_tracks_val = yaml.safe_load(f)["validation_tracks"]

        # either keep the val tracks or remove them from the whole list of train tracks
        if split == "valid":
            list_tracks_subset = list_tracks_val
        else:
            list_tracks_subset = list(set(list_tracks_subset) - set(list_tracks_val))

    # Sort the list
    list_tracks_subset.sort()

    return list_tracks_subset


class MUSDBDatasetFulltrack(Dataset):
    def __init__(
        self,
        targets=None,
        sources=None,
        root="data/",
        subset: str = "train",
        split: str = "train",
        sample_rate: int = 44100,
        seed: int = 42,
        *args,
        **kwargs,
    ) -> None:
        if sources is None:
            sources = ["vocals", "drums", "bass", "other"]
        if targets is None:
            targets = sources
        if isinstance(targets, str):
            targets = [targets]

        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        self.sources = sources
        self.targets = targets
        self.subset = subset
        self.split = split
        self.root = root
        self.list_tracks = get_track_list(root, subset, split)

        # Sample rate (original, target), and eventually resampling fn
        self.orig_sample_rate = 44100  # musdb is fixed sample rate
        self.sample_rate = sample_rate
        self.resample_bool = not (self.sample_rate == self.orig_sample_rate)
        if self.resample_bool:
            self.resample_fn = torchaudio.transforms.Resample(
                self.orig_sample_rate, self.sample_rate
            )

    def __getitem__(self, index):
        audio_sources = []

        # Select track
        track_name = self.list_tracks[index]
        track_dir = os.path.join(self.root, self.subset, track_name)

        # Load the sources
        for source in self.sources:
            src_path = os.path.join(track_dir, source + ".wav")

            # Load the waveform
            audio = torchaudio.load(src_path)[0]

            # Resample if needed
            if self.resample_bool:
                audio = self.resample_fn(audio)

            # Add the current source to the list of all sources
            audio_sources.append(audio)

        # create stem tensor of shape (source, channel, samples)
        stems = torch.stack(audio_sources, dim=0)
        # apply linear mix over source index=0
        x = stems.sum(0)

        # Only keep the targets to be estimated
        ind_trg = [self.sources.index(x) for x in self.targets]
        y = stems[ind_trg, ...]

        return x, y, track_name

    def __len__(self):
        return len(self.list_tracks)


class MUSDBDatasetSAD(Dataset):
    def __init__(
        self,
        targets=["vocals"],
        sources=["vocals", "bass", "drums", "other"],
        data_dir="data/musdb18hq/",
        sad_dir="data/",
        n_samples=None,
        subset: str = "train",
        split: str = "train",
        sample_rate: int = 44100,
        seq_duration: Optional[float] = 3.0,
        source_augmentations: bool = None,
        min_gain: float = -10.0,
        max_gain: float = 10.0,
        p_silent: float = 0.1,
        seed: int = 42,
    ) -> None:
        # Seed for reproducibility
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Sources and targets
        if sources is None:
            sources = ["vocals", "drums", "bass", "other"]
        if isinstance(targets, str):
            targets = [targets]
        self.sources = sources
        self.targets = targets
        assert all(
            i in sources for i in targets
        ), "At least one target is not among sources"

        # Dataset general properties
        self.subset = subset
        self.split = split
        self.data_dir = data_dir
        self.n_samples = n_samples

        # musdb has a fix sample rate
        self.orig_sample_rate = 44100

        # Number of samples for the chunks
        self.seq_dur_samples = int(seq_duration * self.orig_sample_rate)

        # Augmentations
        if source_augmentations is None:
            source_augmentations = []
        self.nonrandom_crop = not ("randomcrop" in source_augmentations)
        self.source_augmentations_fn = Augmentator(
            transforms=source_augmentations,
            seq_duration=seq_duration,
            sample_rate=sample_rate,
            min_gain=min_gain,
            max_gain=max_gain,
            p_silent=p_silent,
        )

        # Load the list of valid chunks (preprocessed with SAD) and shuffle them
        # also store their lengths
        self.list_chunks, self.n_chunks = {}, {}
        for src in sources:
            self.list_chunks[src] = pd.read_csv(
                sad_dir + src + "_" + split + ".csv", index_col=0
            ).sample(frac=1)
            self.n_chunks[src] = len(self.list_chunks[src])

        # Sample rate (original, target), and eventually resampling fn
        self.sample_rate = sample_rate
        self.resample_bool = not (self.sample_rate == self.orig_sample_rate)
        if self.resample_bool:
            self.resample_fn = torchaudio.transforms.Resample(
                self.orig_sample_rate, self.sample_rate
            )

    def __getitem__(self, index):
        y = []

        # Load the sources
        for src in self.sources:
            # Get the chunk info (track name and indices)
            ind_src = index % self.n_chunks[src]
            src_row = self.list_chunks[src].iloc[ind_src]
            track_name, ind_beg, ind_end = (
                src_row["track"],
                src_row["ind_beg"],
                src_row["ind_end"],
            )

            # Get the track name to load the audio
            src_path = os.path.join(
                self.data_dir, self.subset, track_name, src + ".wav"
            )

            # Load the waveform
            audio = torchaudio.load(
                src_path, frame_offset=ind_beg, num_frames=ind_end - ind_beg
            )[0]

            # If no random cropping, crop a chunk from the beginning
            if self.nonrandom_crop:
                audio = audio[..., : self.seq_dur_samples]

            # Resample if needed
            if self.resample_bool:
                audio = self.resample_fn(audio)

            # Data augmentation (for the train subset only)
            if self.subset == self.split == "train":
                audio = self.source_augmentations_fn(audio)

            # Add the current source to the list of all sources
            y.append(audio)

        # Create stem tensor of shape (source, channel, samples)
        y = torch.stack(y, dim=0)

        # Apply linear mix over source index=0
        x = y.sum(0)

        # Only keep the targets
        ind_trg = [self.sources.index(x) for x in self.targets]
        y = y[ind_trg, ...]
        # y = y[self.sources.index(self.targets), ...].unsqueeze(0)

        # Normalization w.r.t. the norm of the mixture
        max_norm = x.abs().max()
        max_norm = (
            1 if max_norm == 0 else max_norm
        )  # avoid problem in the (rare) case where all sources are silent
        x /= max_norm
        y /= max_norm

        return x, y, "dummy_track"

    def __len__(self):
        if self.n_samples:
            return self.n_samples
        else:
            return max([self.n_chunks[t] for t in self.targets])


def build_training_samplers(targets, cfg_dset, ngpus=None, fast_tr=False):

    # If fast training (=overfit batches), remove augmentation to ensure it's always the same batch
    src_aug = cfg_dset.source_augmentations
    if fast_tr:
        src_aug = None

    # Number of GPUs
    if ngpus is None:
        ngpus = torch.cuda.device_count()

    # Train and validation datasets
    train_db = MUSDBDatasetSAD(
        targets=targets,
        sources=cfg_dset.sources,
        data_dir=cfg_dset.data_dir,
        sad_dir=cfg_dset.sad_dir,
        n_samples=cfg_dset.n_samples,
        subset="train",
        split="train",
        sample_rate=cfg_dset.sample_rate,
        seq_duration=cfg_dset.seq_duration,
        source_augmentations=src_aug,
        min_gain=cfg_dset.min_gain,
        max_gain=cfg_dset.max_gain,
        p_silent=cfg_dset.p_silent,
        seed=cfg_dset.seed,
    )

    if fast_tr:
        valid_db = train_db
    else:
        valid_db = MUSDBDatasetFulltrack(
            targets=targets,
            sources=cfg_dset.sources,
            root=cfg_dset.data_dir,
            subset="train",
            split="valid",
            sample_rate=cfg_dset.sample_rate,
            seed=cfg_dset.seed,
        )

    # Samplers
    splr_kwargs = {"num_workers": cfg_dset.nb_workers * ngpus, "pin_memory": True}
    train_sampler = DataLoader(
        train_db, batch_size=cfg_dset.batch_size, shuffle=True, **splr_kwargs
    )
    valid_sampler = DataLoader(
        valid_db,
        batch_size=1,
        shuffle=False,
        **splr_kwargs,  # val sampler batch size can be changed if using chunks
    )

    return train_sampler, valid_sampler


def build_fulltrack_sampler(targets, cfg_dset, subset="train", split="valid"):
    fulltrack_db = MUSDBDatasetFulltrack(
        targets=targets,
        sources=cfg_dset.sources,
        root=cfg_dset.data_dir,
        subset=subset,
        split=split,
        sample_rate=cfg_dset.sample_rate,
        seed=cfg_dset.seed,
    )

    splr_kwargs = {"num_workers": cfg_dset.nb_workers, "pin_memory": True}
    fulltrack_sampler = DataLoader(fulltrack_db, batch_size=1, **splr_kwargs)

    return fulltrack_sampler


if __name__ == "__main__":
    train_db = MUSDBDatasetSAD(
        targets="vocals",
        data_dir="data/musdb18hq/",
        sad_dir="data/",
        subset="train",
        split="train",
        n_samples=None,
        sample_rate=44100,
        seq_duration=3.0,
        source_augmentations=["randomcrop", "rescale", "silentsource"],
        seed=42,
    )

    print(len(train_db))

    x, y, track_name = train_db[0]
    print(x.shape, y.shape)

    x2, y2, _ = train_db[0]
    print(torch.linalg.norm(x - x2))

# EOF
