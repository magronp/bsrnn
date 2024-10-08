import torchaudio
import pandas as pd
from os import sched_getaffinity
from helpers.data import get_track_list
import multiprocessing.pool
import functools
import torch
from os.path import join
import tqdm
from models.separator import Separator


def process_all_tracks_parallel(args, subset="test", split=None, num_cpus=None):

    # List of tracks to process
    list_tracks = get_track_list(args.data_dir, subset=subset, split=split)

    # Get number of available CPUs
    if num_cpus is None:
        num_cpus = len(sched_getaffinity(0)) // 4 - 1

    # Defined the simplified function (freeze the non track-specific arguments)
    myfun = functools.partial(process_track_and_evaluate, args=args, subset=subset)

    # If parallel, use multi-CPU to perform evaluation
    if num_cpus > 1:
        # Set torch num thread to 1 (mandatory when using multiprocessing with torch ops)
        torch.set_num_threads(1)

        # Define the pool, and run
        with multiprocessing.pool.Pool(num_cpus) as pool:
            sdr_list = list(
                pool.map(
                    func=myfun,
                    iterable=list_tracks,
                    chunksize=1,
                )
            )
    else:
        # Simple loop over tracks (applies to both CPU and GPU)
        sdr_list = []
        for track_name in tqdm.tqdm(list_tracks):
            sdr = myfun(track_name)
            sdr_list.append(sdr)

    # Arrange the results as pd dataframe
    cols = args.targets.copy()
    cols.insert(0, "track")
    test_results = pd.DataFrame(columns=cols)
    for sdr in sdr_list:
        test_results.loc[len(test_results)] = sdr

    return test_results


def process_track_and_evaluate(track_name, args, subset="test"):

    # Process only part of the track if needed (mostly for debugging)
    if args.dset.eval_seq_duration is None:
        nfr = -1
    else:
        nfr = int(args.dset.eval_seq_duration * args.sample_rate)

    # Folder where the track reference files are stored
    track_dir = join(args.data_dir, subset, track_name)

    # Load true sources, add batch dim [1, n_targets, n_channels, n_samples]
    references = torch.stack(
        [
            torchaudio.load(join(track_dir, trg + ".wav"), num_frames=nfr)[0]
            for trg in args.dset.sources
        ]
    ).unsqueeze(0)

    # Create mix
    mix = torch.sum(references, dim=1)

    # Load the model
    model = Separator(args)

    # Test step : estimate the sources and get the SDR
    test_batch = (mix, references, [track_name])
    test_sdr = model.test_step(test_batch, 0)

    return test_sdr
