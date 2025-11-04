import torch
from torch.utils.data import Dataset
import torchaudio
import random
from typing import Optional
import os
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import musdb
import yaml


def rec_estimates(estimates, track_rec_dir, targets, sample_rate):
    """
    estimates: [n_targets, n_channels, n_samples]
    """

    if track_rec_dir is None:
        track_rec_dir = os.getcwd()

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
        aug_list: list = [],
        rescale_min: float = 0.25,
        rescale_max: float = 1.25,
        gain_dB_min: float = -10,
        gain_dB_max: float = 10,
        p_silent: float = 0.1,
        p_channelswap: float = 0.5,
        *args,
        **kwargs,
    ):
        # List of augmentations functions
        self.aug_list = aug_list
        self.augs = []
        for a in aug_list:
            current_aug = "_augment_" + a
            if hasattr(self, current_aug):
                self.augs.append(getattr(self, current_aug))

        # Parameters
        self.rescale_min = rescale_min
        self.rescale_max = rescale_max
        self.gain_dB_min = gain_dB_min
        self.gain_dB_max = gain_dB_max
        self.p_silent = p_silent
        self.p_channelswap = p_channelswap

    def _augment_rescale_db(self, audio: torch.Tensor) -> torch.Tensor:
        """Applies a random gain between `min_gain` and `max_gain` (db scale)"""
        g = self.gain_dB_min + torch.rand(1) * (self.gain_dB_max - self.gain_dB_min)
        g = 10 ** (g / 20)
        return audio * g

    def _augment_rescale(self, audio: torch.Tensor) -> torch.Tensor:
        """Applies a random gain between `min_gain` and `max_gain` (linear scale)"""
        g = self.rescale_min + torch.rand(1) * (self.rescale_max - self.rescale_min)
        return audio * g

    def _augment_channelswap(self, audio: torch.Tensor) -> torch.Tensor:
        """Randomly swap channels of stereo signals with probability p_channelswap"""
        if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < self.p_channelswap:
            return torch.flip(audio, [0])
        else:
            return audio

    def _augment_silentsource(self, audio: torch.Tensor) -> torch.Tensor:
        """Randomly silent source with a probability p_silent"""
        if torch.tensor(1.0).uniform_() < self.p_silent:
            return torch.zeros_like(audio)
        else:
            return audio

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for aug in self.augs:
            audio = aug(audio)

        return audio


def get_track_list(data_dir, subset, split=None):
    # get the list of tracks in the subset
    list_tracks_subset = os.listdir(data_dir + subset)

    # for the training subset, need to keep or filter out val tracks
    if subset == "train":
        # list of validation tracks, predefined in musdb
        setup_path = os.path.join(musdb.__path__[0], "configs", "mus.yaml")
        with open(setup_path, "r") as f:
            list_tracks_val = yaml.safe_load(f)["validation_tracks"]

        # list: see https://github.com/sigsep/sigsep-mus-db/blob/master/musdb/configs/mus.yaml)
        # list_tracks_val = [
        #    "Actions - One Minute Smile",
        #    "Clara Berry And Wooldog - Waltz For My Victims",
        #    "Johnny Lokke - Promises & Lies",
        #    "Patrick Talbot - A Reason To Leave",
        #    "Triviul - Angelsaint",
        #    "Alexander Ross - Goodbye Bolero",
        #    "Fergessen - Nos Palpitants",
        #    "Leaf - Summerghost",
        #    "Skelpolu - Human Mistakes",
        #    "Young Griffo - Pennies",
        #    "ANiMAL - Rockshow",
        #    "James May - On The Line",
        #    "Meaxic - Take A Step",
        #    "Traffic Experiment - Sirens",
        # ]

        if split == "valid":
            list_tracks_subset = list_tracks_val
        else:
            # if split=="train", remove the val tracks from the whole list of train tracks
            list_tracks_subset = list(set(list_tracks_subset) - set(list_tracks_val))

    # Sort the list
    list_tracks_subset.sort()

    return list_tracks_subset


class MUSDBDatasetUMX(Dataset):
    def __init__(
        self,
        targets=["vocals"],
        sources=["vocals", "bass", "drums", "other"],
        data_dir: str = "data/musdb18hq/",
        subset: str = "train",
        split: str = "train",
        sample_rate: int = 44100,
        seq_duration: Optional[float] = 3.0,
        samples_per_track: int = 64,
        seed: int = 42,
        # augmentations:
        aug_list: list = [],
        aug_params: dict = {},
        *args,
        **kwargs,
    ) -> None:
        # Seed for reproducibility
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Sources and targets
        if isinstance(targets, str):
            targets = [targets]
        self.sources = sources
        self.targets = targets
        assert all(
            i in sources for i in targets
        ), "At least one target is not among sources"

        # Dataset split and dir
        self.subset = subset
        self.split = split
        self.data_dir = data_dir

        # Sample rates (original and new)
        self.orig_sample_rate = 44100  # musdb has a fixed sample rate
        self.sample_rate = sample_rate

        # Data parameters
        self.seq_duration = seq_duration
        self.samples_per_track = (
            samples_per_track if seq_duration else 1
        )  # process entire tracks if no duration provided

        # List of tracks
        self.list_tracks = get_track_list(data_dir, subset, split)

        # Resample function
        self.resample_bool = self.sample_rate != self.orig_sample_rate
        if self.resample_bool:
            self.resample_fn = torchaudio.transforms.Resample(
                self.orig_sample_rate, self.sample_rate
            )

        # Augmentations
        self.random_chunk = "random_chunk" in aug_list
        self.random_track_mix = "random_track_mix" in aug_list
        self.augs_fn = Augmentator(aug_list=aug_list, **aug_params)

    def __getitem__(self, index):
        audio_sources = []

        # define the track when it's the same for all sources (valid and test)
        track_name = self.list_tracks[index // self.samples_per_track]

        # Ensure the mix is not all-zero
        silent_mix = True
        while silent_mix:
            # Load the sources
            for source in self.sources:

                # Get the track for the current source (needed for training)
                if self.random_track_mix:
                    track_name = random.choice(self.list_tracks)

                # Track folder
                track_dir = os.path.join(self.data_dir, self.subset, track_name)

                # Get the total track duration (useful if 'seq_duration' is specified)
                track_info = torchaudio.info(os.path.join(track_dir, "mixture.wav"))
                track_duration = track_info.num_frames / track_info.sample_rate

                src_path = os.path.join(track_dir, source + ".wav")

                # Check wether a duration is provided or not
                if self.seq_duration:
                    # Load random chunks of seq_duration only if not deterministic chunks
                    if self.random_chunk:
                        frame_offset = int(
                            random.uniform(0, track_duration - self.seq_duration)
                            * self.orig_sample_rate
                        )
                        num_frames = int(self.seq_duration * self.orig_sample_rate)
                    else:
                        # otherwise load from the beginning, with length seq_duration
                        frame_offset = 0
                        num_frames = int(self.seq_duration * self.orig_sample_rate)
                else:
                    # otherwise load the whole track
                    frame_offset = 0
                    num_frames = -1

                # Load the waveform
                audio = torchaudio.load(
                    src_path, frame_offset=frame_offset, num_frames=num_frames
                )[0]

                # Data augmentation (for the train subset only)
                if self.subset == self.split == "train":
                    audio = self.augs_fn(audio)

                # Add the current source to the list of all sources
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)

            # apply linear mix over source index=0
            x = stems.sum(0)

            # Check if there is energy in the mixture
            silent_mix = torch.sum(torch.square(x)) < 1e-6

        # Only keep the targets to be estimated
        ind_trg = [self.sources.index(x) for x in self.targets]
        y = stems[ind_trg, ...]

        # Resample if needed
        if self.resample_bool:
            y = self.resample_fn(y)
            x = self.resample_fn(x)

        return x, y, track_name

    def __len__(self):
        return len(self.list_tracks) * self.samples_per_track


class MUSDBDataset(Dataset):
    def __init__(
        self,
        targets=["vocals"],
        sources=["vocals", "bass", "drums", "other"],
        data_dir: str = "data/musdb18hq/",
        subset: str = "train",
        split: str = "train",
        sample_rate: int = 44100,
        seq_duration: Optional[float] = 3.0,
        n_samples: int = 20000,
        seed: int = 42,
        # augmentations:
        aug_list: list = [],
        aug_params: dict = {},
        *args,
        **kwargs,
    ) -> None:
        # Seed for reproducibility
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Sources and targets
        if isinstance(targets, str):
            targets = [targets]
        self.sources = sources
        self.targets = targets
        assert all(
            i in sources for i in targets
        ), "At least one target is not among sources"

        # Dataset split and dir
        self.subset = subset
        self.split = split
        self.data_dir = data_dir

        # Sample rates (original and new)
        self.orig_sample_rate = 44100  # musdb has a fixed sample rate
        self.sample_rate = sample_rate

        # Data parameters
        self.seq_duration = seq_duration
        self.n_samples = n_samples

        # List of tracks
        self.list_tracks = get_track_list(data_dir, subset, split)

        # Adjust n_samples if not precised
        if self.n_samples is None:
            self.n_samples = len(self.list_tracks)

        # Resample function
        self.resample_bool = self.sample_rate != self.orig_sample_rate
        if self.resample_bool:
            self.resample_fn = torchaudio.transforms.Resample(
                self.orig_sample_rate, self.sample_rate
            )

        # Augmentations
        self.random_chunk = "random_chunk" in aug_list
        self.random_track_mix = "random_track_mix" in aug_list
        self.augs_fn = Augmentator(aug_list=aug_list, **aug_params)

    def __getitem__(self, index):
        audio_sources = []

        # define the track when it's the same for all sources (valid and test)
        track_name = self.list_tracks[index % len(self.list_tracks)]

        # Load the sources
        for source in self.sources:

            # Get the track for the current source (needed for training)
            if self.random_track_mix:
                track_name = random.choice(self.list_tracks)

            # Track folder
            track_dir = os.path.join(self.data_dir, self.subset, track_name)

            # Get the total track duration (useful if 'seq_duration' is specified)
            track_info = torchaudio.info(os.path.join(track_dir, "mixture.wav"))
            track_duration = track_info.num_frames / track_info.sample_rate

            src_path = os.path.join(track_dir, source + ".wav")

            # Check wether a duration is provided or not
            if self.seq_duration:
                # Load random chunks of seq_duration only if not deterministic chunks
                if self.random_chunk:
                    frame_offset = int(
                        random.uniform(0, track_duration - self.seq_duration)
                        * self.orig_sample_rate
                    )
                    num_frames = int(self.seq_duration * self.orig_sample_rate)
                else:
                    # otherwise load from the beginning, with length seq_duration
                    frame_offset = 0
                    num_frames = int(self.seq_duration * self.orig_sample_rate)
            else:
                # otherwise load the whole track
                frame_offset = 0
                num_frames = -1

            # Load the waveform
            audio = torchaudio.load(
                src_path, frame_offset=frame_offset, num_frames=num_frames
            )[0]

            # Data augmentation (for the train subset only)
            if self.subset == self.split == "train":
                audio = self.augs_fn(audio)

            # Add the current source to the list of all sources
            audio_sources.append(audio)

        # Create stem tensor of shape (source, channel, samples)
        y = torch.stack(audio_sources, dim=0)

        # Resample if needed
        if self.resample_bool:
            y = self.resample_fn(y)

        # apply linear mix over source index=0
        x = y.sum(0)

        # Only keep the targets to be estimated
        ind_trg = [self.sources.index(x) for x in self.targets]
        y = y[ind_trg, ...]

        return x, y, track_name

    def __len__(self):
        return self.n_samples


class MUSDBDatasetSAD(Dataset):
    def __init__(
        self,
        targets=["vocals"],
        subset: str = "train",
        split: str = "train",
        sources: list = ["vocals", "bass", "drums", "other"],
        data_dir: str = "data/musdb18hq/",
        sad_dir: str = "data/",
        sample_rate: int = 44100,
        seq_duration: Optional[float] = 3.0,
        n_samples=None,
        seed: int = 42,
        # augmentations:
        aug_list: list = [],
        aug_params: dict = {},
        *args,
        **kwargs,
    ) -> None:
        # Seed for reproducibility
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Sources and targets
        if isinstance(targets, str):
            targets = [targets]
        self.sources = sources
        self.targets = targets
        assert all(
            i in sources for i in targets
        ), "At least one target is not among sources"

        # Dataset split and dir
        self.subset = subset
        self.split = split
        self.data_dir = data_dir

        # Sample rates (original and new)
        self.orig_sample_rate = 44100  # musdb has a fixed sample rate
        self.sample_rate = sample_rate

        # Data parameters
        self.seq_duration = seq_duration
        self.n_samples = n_samples
        self.chunk_samples = int(self.seq_duration * self.orig_sample_rate)

        # Load the list of valid chunks (preprocessed with SAD) and shuffle them
        # also store their lengths
        self.list_chunks, self.n_chunks = {}, {}
        for src in sources:
            chunks = pd.read_csv(sad_dir + src + "_" + split + ".csv", index_col=0)
            if (
                "shuffle_tracks" in aug_list
            ):  # even if no shuffle_tracks, tracks will occasionnally be mixed since they have diffferent number of valid chunks
                chunks = chunks.sample(frac=1, random_state=seed)
            self.list_chunks[src] = chunks
            self.n_chunks[src] = len(chunks)

        # Resampling function
        self.resample_bool = self.sample_rate != self.orig_sample_rate
        if self.resample_bool:
            self.resample_fn = torchaudio.transforms.Resample(
                self.orig_sample_rate, self.sample_rate
            )

        # Augmentations
        self.random_track_mix = "random_track_mix" in aug_list
        self.random_chunk = "random_chunk" in aug_list
        self.silenttarget = "silenttarget" in aug_list
        if self.silenttarget:
            self.p_silent = aug_params.p_silent
        self.augs_fn = Augmentator(aug_list=aug_list, **aug_params)

    def __getitem__(self, index):
        y = []

        # Load the sources
        for src in self.sources:

            # Get the chunk info (track name and indices)
            if self.random_track_mix:
                ind_src = random.randint(0, self.n_chunks[src] - 1)
            else:
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

            # If random chunk, offset inside the SAD-processed segment
            if self.random_chunk:
                ofst = random.randint(0, (ind_end - ind_beg) - self.chunk_samples)
            else:
                ofst = 0  # otherwise get a chunk from the beginning

            # Load the waveform corresponding to the chunk in the SAD-processed segment
            audio = torchaudio.load(
                src_path, frame_offset=ind_beg + ofst, num_frames=self.chunk_samples
            )[0]

            # Data augmentation (for the train subset only)
            if self.subset == self.split == "train":
                audio = self.augs_fn(audio)

            # Add the current source to the list of all sources
            y.append(audio)

        # Create stem tensor of shape (source, channel, samples)
        y = torch.stack(y, dim=0)

        # Resample if needed
        if self.resample_bool:
            y = self.resample_fn(y)

        # Apply linear mix over source index=0
        x = y.sum(0)

        # Only keep the targets to be estimated
        ind_trg = [self.sources.index(x) for x in self.targets]
        y = y[ind_trg, ...]

        # Silent the target only (only if 1 target to estimate)
        if self.silenttarget and len(self.targets) == 1:
            if torch.tensor(1.0).uniform_() < self.p_silent:
                x -= y.squeeze(0)  # remove target from mix
                y = torch.zeros_like(y)  # zero the target

        # Normalization w.r.t. the max of the max of mixture / targets
        max_norm = torch.maximum(x.abs().max(), y.abs().max())
        max_norm = (
            1 if max_norm < 1e-6 else max_norm
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

    # If fast_tr (=overfit batches), remove augmentations (including random mix/chunks), to ensure it's always the same batch
    if fast_tr:
        cfg_dset.aug_list = []

    # Number of GPUs
    if ngpus is None:
        ngpus = torch.cuda.device_count()

    # Training dataset
    if cfg_dset.sad_dir:
        DSET = MUSDBDatasetSAD
    else:
        DSET = MUSDBDataset

    train_db = DSET(
        targets=targets,
        subset="train",
        split="train",
        n_samples=cfg_dset.n_samples_tr,
        seq_duration=cfg_dset.seq_duration_tr,
        **cfg_dset,
    )

    # Validation dataset
    if fast_tr:
        valid_db = train_db
    else:
        valid_db = MUSDBDataset(
            targets=targets,
            subset="train",
            split="valid",
            sources=cfg_dset.sources,
            data_dir=cfg_dset.data_dir,
            sample_rate=cfg_dset.sample_rate,
            n_samples=cfg_dset.n_samples_eval,
            seq_duration=cfg_dset.seq_duration_eval,
            seed=cfg_dset.seed,
            aug_list=[],  # no augmentation for the validation set
        )

    # Samplers
    splr_kwargs = {"num_workers": cfg_dset.nb_workers * ngpus, "pin_memory": True}
    train_sampler = DataLoader(
        train_db, batch_size=cfg_dset.batch_size, shuffle=not fast_tr, **splr_kwargs
    )
    valid_sampler = DataLoader(
        valid_db,
        batch_size=1,  # =1 since full tracks
        **splr_kwargs,
    )

    return train_sampler, valid_sampler


def build_eval_sampler(targets, cfg_dset, subset="train", split="valid"):
    eval_db = MUSDBDataset(
        targets=targets,
        subset=subset,
        split=split,
        sources=cfg_dset.sources,
        data_dir=cfg_dset.data_dir,
        sample_rate=cfg_dset.sample_rate,
        n_samples=cfg_dset.n_samples_eval,
        seq_duration=cfg_dset.seq_duration_eval,
        seed=cfg_dset.seed,
        aug_list=[],  # no augmentation for eval sets (validation and test)
    )

    splr_kwargs = {"num_workers": cfg_dset.nb_workers, "pin_memory": True}
    eval_sampler = DataLoader(eval_db, batch_size=1, **splr_kwargs)

    return eval_sampler


if __name__ == "__main__":

    targets = ["vocals"]

    # Augmentations
    # aug_list = ["shuffle_tracks", "random_chunk", "rescale_db", "silentsource"]
    aug_list = ["random_track_mix", "random_chunk", "rescale_db", "silenttarget"]

    aug_params = OmegaConf.create(
        {
            "rescale_min": 0.25,
            "rescale_max": 1.25,
            "gain_dB_min": -10,
            "gain_dB_max": 10,
            "p_channelswap": 0.5,
            "p_silent": 0.1,
        }
    )

    cfg_dset = OmegaConf.create(
        {
            "sources": ["vocals", "bass", "drums", "other"],
            "data_dir": "data/musdb18hq/",
            # "sad_dir": "data/",
            "sad_dir": None,
            "sample_rate": 44100,
            "n_samples_tr": 5000,
            "seq_duration_tr": 3.0,
            "n_samples_eval": None,
            "seq_duration_eval": 10,
            "seed": 42,
            # augmentations
            "aug_list": aug_list,
            "aug_params": aug_params,
            # samplers
            "nb_workers": 4,
            "batch_size": 2,
        }
    )

    # Dataset
    train_db = MUSDBDatasetSAD(
        targets=targets,
        subset="train",
        split="train",
        data_dir=cfg_dset.data_dir,
        sad_dir="data/",
        sample_rate=cfg_dset.sample_rate,
        n_samples=cfg_dset.n_samples_tr,
        seq_duration=cfg_dset.seq_duration_tr,
        seed=cfg_dset.seed,
        aug_list=aug_list,
        aug_params=aug_params,
    )
    print("Training set length:", len(train_db))

    # Retrieve one training sample
    x, y, track_name = train_db[0]
    print(x.shape, y.shape)

    # Build samplers
    train_sampler, valid_sampler = build_training_samplers(
        targets, cfg_dset, fast_tr=False
    )

    # Retrieve one training batch
    x, y, _ = next(iter(train_sampler))
    print(len(train_sampler), x.shape, y.shape)
    torchaudio.save("ex_tr_mix.wav", x[0], 44100)
    torchaudio.save("ex_tr_target.wav", y[0, 0], 44100)

    # Retrieve one validation batch
    x, y, _ = next(iter(valid_sampler))
    print(len(valid_sampler), x.shape, y.shape)
    torchaudio.save("ex_val_mix.wav", x[0], 44100)
    torchaudio.save("ex_val_target.wav", y[0, 0], 44100)

# EOF
