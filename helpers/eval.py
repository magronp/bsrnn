import torch
import torchaudio
import museval
import pandas as pd
import math
import time
from os import path, sched_getaffinity
import tqdm
from os.path import join
from helpers.data import get_track_list, rec_estimates
import multiprocessing
import functools


def mypad(input_tensor, tot_len):
    input_tensor = torch.nn.functional.pad(
        input_tensor, (0, tot_len - input_tensor.shape[-1])
    )
    return input_tensor


def sdr_global(ref, est, eps=1e-7):
    """
    Global SDR: no distortion filter, no frame, back to basic (MDX21 style).
    Note : it's also called "new SDR" (nSDR) in DEMUCS or "utterance SDR" (uSDR) in BSRNN...
    ref / est: [batch_size, n_targets, n_channels, n_samples]
    output: [n_targets]
    """

    dif = est - ref
    num = torch.sum(torch.square(ref), dim=(2, 3)) + eps
    den = torch.sum(torch.square(dif), dim=(2, 3)) + eps
    sdr = 10 * torch.log10(num / den)

    # Reshape to [n_targets, batch_size]
    sdr = sdr.transpose(1, 0)

    return sdr


def sdr_framewise(ref, est, win_bss=44100, eps=1e-7):
    """
    Frame-wise SDR, no distortion filter allowed (useful for validation since it's fast to compute)
    median is computed outside of the fn, after stacking batches over the whole (validation) dataset
    indeed, this function can be used for other scenarios where we might want to stack more batches before median
    ref / est: [batch_size, n_targets, n_channels, n_samples]
    output: [n_targets, batch_size*n_frames]
    """

    win_bss = int(win_bss)  # make sure it's an integer

    # Get input shape
    batch_size, n_targets, n_channels, n_samples = ref.shape

    dif = est - ref

    # Crop the signals to remove the last bit that's shorter than win_bss
    # cropped_len = int((n_samples // win_bss) * win_bss)
    # ref = ref[:, :, :, :cropped_len]
    # dif = dif[:, :, :, :cropped_len]

    # Pad if needed so the signal length is a multiple of win_bss
    tot_len = math.ceil(n_samples / win_bss) * win_bss
    ref = mypad(ref, tot_len)
    dif = mypad(dif, tot_len)

    # Reshape them as (batch_size, n_targets, n_channels, n_frames, win_bss)
    ref = ref.view(batch_size, n_targets, n_channels, -1, win_bss)
    dif = dif.view(batch_size, n_targets, n_channels, -1, win_bss)

    # Sum over channels and win samples, and take the ratio
    num = torch.sum(torch.square(ref), dim=(2, 4)) + eps
    den = torch.sum(torch.square(dif), dim=(2, 4)) + eps
    sdr_frames = 10 * torch.log10(num / den)

    # Reshape to [n_targets, batch_size*n_frames]
    sdr_frames = sdr_frames.permute(1, 0, 2).reshape(n_targets, -1)

    # Get the median over frames
    sdr_med = torch.nanmedian(sdr_frames, dim=1)[0]

    return sdr_med, sdr_frames


def compute_sdr(references, estimates, win_bss=44100, sdr_type="global", eps=1e-7):
    """
    references / estimates: [batch_size, n_targets, n_channels, n_samples]
    sdr: [n_targets]
    sdr_frames: [n_targets, batch_size*n_frames]   (where n_frames = n_samples // win_bss)
    """

    win_bss = int(win_bss)  # make sure it's an integer

    if sdr_type == "framewise":
        sdr, sdr_frames = sdr_framewise(references, estimates, win_bss=win_bss, eps=eps)

    elif sdr_type == "global":
        sdr_frames = sdr_global(references, estimates, eps=eps)
        # here, we get the mean in case each sample in the batch corresponds to a different track... but in practice there will be only 1
        sdr = torch.nanmean(sdr_frames, dim=1)

    elif sdr_type == "museval":
        # To use museval, need to remove the batch dim, reshape to appropriate size [n_targets, n_samples, n_channels]
        # and back to numpy
        references = references.cpu().squeeze(0).transpose(1, 2).numpy()
        estimates = estimates.cpu().squeeze(0).transpose(1, 2).numpy()

        # SISEC 2018 style (museval)
        # TO DO: à implémenter en torch + plus de rapidité ?
        sdr_frames = museval.metrics.bss_eval(
            references,
            estimates,
            compute_permutation=False,
            window=win_bss,
            hop=win_bss,
            framewise_filters=False,
            bsseval_sources_version=False,
        )[0]
        sdr_frames = torch.from_numpy(sdr_frames)
        sdr = torch.nanmedian(sdr_frames, dim=1)[0]  # Median over frames

    else:
        raise NameError("Unknown SDR type")

    return sdr, sdr_frames


def get_loss_fn(loss_type="L1"):
    if loss_type == "L1":
        loss_fn = torch.nn.L1Loss()
    elif loss_type == "MSE":
        loss_fn = torch.nn.MSELoss()
    else:
        raise NameError("Unknown loss function")

    return loss_fn


def compute_loss(refs, est, loss_type="L1", loss_domain="t"):

    # Transform the loss-domain string into a list (based on the separator "+")
    loss_domains = loss_domain.split("+")

    # Instanciate the loss function
    loss_fn = get_loss_fn(loss_type)

    # Iterate over domains to compute the total loss
    loss = 0
    for ld in loss_domains:
        # Compute the loss depending on the domain (time-domain, TF, ...)
        if ld == "t":
            y, y_hat = refs["waveforms"], est["waveforms"]
            loss += loss_fn(y_hat, y)
        elif ld == "tf":
            Y, Y_hat = refs["stfts"], est["stfts"]
            loss += loss_fn(Y.real, Y_hat.real) + loss_fn(Y.imag, Y_hat.imag)
        else:
            raise NameError("Unknown loss domain")

    return loss


def apply_fn_to_sources(
    input_sig,
    my_fn,
    sample_rate=44100,
    eval_segment_len=10.0,
    eval_overlap=0.1,
    *args,
    **kwargs
):
    # my_fn must be a function from [B, ntrg, nch, L] to [B, ntrg, nch, L]
    # For evaluation, it takes the reference and fabricates the mix inside to avoid dim problem

    bsize, n_targets, nb_channels, nsamples = input_sig.shape

    # treat the case where we might want to process the whole track at once
    if eval_segment_len == -1:
        chunk_len = nsamples + 1
    else:
        chunk_len = int(sample_rate * eval_segment_len * (1 + eval_overlap))

    # if too big chunks / too small mixture, ignore the OLA / fader
    if chunk_len >= nsamples:
        output_sig = my_fn(input_sig, *args, **kwargs)

    else:
        start = 0
        end = chunk_len
        overlap_frames = eval_overlap * eval_segment_len * sample_rate
        fade = torchaudio.transforms.Fade(
            fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear"
        )

        output_sig = torch.zeros(
            (bsize, n_targets, nb_channels, nsamples), device=input_sig.device
        )

        while start < nsamples - overlap_frames:

            # extract current chunk
            chunk = input_sig[..., start:end]

            # apply function, detach and alocate to input device to be sure
            out_chunk = my_fn(chunk, *args, **kwargs)
            out_chunk = out_chunk.detach().to(input_sig.device)

            # fader / overlap-add
            out_chunk = fade(out_chunk)
            output_sig[..., start:end] += out_chunk

            # update indices
            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= nsamples:
                fade.fade_out_len = 0

    return output_sig


def process_all_tracks(
    my_fn,
    subset="test",
    split=None,
    parallel_cpu=True,
    targets=["vocals", "bass", "drums", "other"],
    rec_dir=None,
    data_dir="data/",
    win_dur=1.0,
    sample_rate=44100,
    max_len=None,
    verbose_per_track=True,
    eval_segment_len=10.0,
    eval_overlap=0.1,
    sdr_type="global",
    *args,
    **kwargs
):

    # List of tracks to process
    list_tracks = get_track_list(data_dir, subset=subset, split=split)

    # Get number of available CPUs and check whether parallelize or not
    num_cpus = len(sched_getaffinity(0)) // 4 - 1
    num_cpus = 10  # because the formula above doesn't really work...
    parall = parallel_cpu and num_cpus > 1
    print("Parallel CPU:", parall)

    # Defined the simplified function (freeze the non track-specific arguments)
    myfun = functools.partial(
        process_track_and_evaluate,
        my_fn=my_fn,
        subset=subset,
        targets=targets,
        rec_dir=rec_dir,
        data_dir=data_dir,
        win_dur=win_dur,
        sample_rate=sample_rate,
        max_len=max_len,
        verbose_per_track=verbose_per_track,
        eval_segment_len=eval_segment_len,
        eval_overlap=eval_overlap,
        sdr_type=sdr_type,
        *args,
        **kwargs
    )

    # If parallel, use multi-CPU to perform evaluation
    if parall:
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
    cols = targets.copy()
    cols.insert(0, "track")
    test_results = pd.DataFrame(columns=cols)
    for sdr in sdr_list:
        test_results.loc[len(test_results)] = sdr

    return test_results


def process_track_and_evaluate(
    track_name,
    my_fn,
    subset="test",
    targets=["vocals", "bass", "drums", "other"],
    rec_dir=None,
    data_dir="data/",
    win_dur=1.0,
    sample_rate=44100,
    max_len=None,
    verbose_per_track=True,
    eval_segment_len=10.0,
    eval_overlap=0.1,
    sdr_type="global",
    *args,
    **kwargs
):

    # Process only part of the track if needed (mostly for debugging)
    if max_len is None:
        nfr = -1
    else:
        nfr = int(max_len * sample_rate)

    # Folder where the track reference files are stored
    track_dir = join(data_dir, subset, track_name)

    # Load true sources, add batch dim [1, n_targets, n_channels, n_samples]
    references = torch.stack(
        [
            torchaudio.load(join(track_dir, trg + ".wav"), num_frames=nfr)[0]
            for trg in targets
        ]
    ).unsqueeze(0)

    # Estimate the sources
    estimates = apply_fn_to_sources(
        references,
        my_fn,
        sample_rate=sample_rate,
        eval_segment_len=eval_segment_len,
        eval_overlap=eval_overlap,
        *args,
        **kwargs
    )

    # SDR
    win_bss = int(win_dur * sample_rate)
    sdr_med = compute_sdr(references, estimates, win_bss=win_bss, sdr_type=sdr_type)[0]

    # Arrange SDR into a dict
    test_sdr = {}
    test_sdr["track"] = track_name
    for j, trg in enumerate(targets):
        test_sdr[trg] = sdr_med[j].item()

    # Display results
    if verbose_per_track:
        print(test_sdr)

    # Record the estimates
    if rec_dir:
        estimates = estimates[0]  # remove the batch dim
        track_rec_dir = join(rec_dir, subset, track_name)
        rec_estimates(estimates, track_rec_dir, targets, sample_rate)

    return test_sdr


def aggregate_res_over_tracks(path_or_df, method_name=None, sdr_type="global"):

    # The results are either directly provided, or as a path to a file
    if isinstance(path_or_df, str):
        test_results = pd.read_csv(path_or_df)
    else:
        test_results = path_or_df

    # Aggregate scores over tracks (mean for the global  SDR, median otherwise)
    if sdr_type == "global":
        test_results_agg = test_results.mean(numeric_only=True, axis=0)
    else:
        test_results_agg = test_results.median(numeric_only=True, axis=0)

    # Get the mean over sources
    mean_sdr = test_results_agg.mean()

    # Reform the dataframe
    test_results_agg = pd.DataFrame(test_results_agg).T
    test_results_agg["song"] = mean_sdr

    # Add the method name
    if method_name is not None:
        test_results_agg.insert(loc=0, column="method", value=method_name)

    return test_results_agg


def append_df_to_main_file(path_main_file, df):

    # Load the file containing all results if it exists
    if path.exists(path_main_file):
        all_test_results = pd.read_csv(path_main_file, index_col=0)
    # Otherwise, create it
    else:
        cols = list(df.columns.values)
        all_test_results = pd.DataFrame(columns=cols)

    # Add a new entry to the frame containing all results
    all_test_results = pd.concat([all_test_results, df], ignore_index=True)

    # Record the results
    all_test_results.to_csv(path_main_file)

    return


if __name__ == "__main__":
    # Realistic signal parameters
    sample_rate = 44100
    bsize = 1
    n_targets = 4
    nb_channels = 2
    win_bss = 1 * sample_rate
    n_samples = sample_rate * 200

    # Create ref and estimates
    references = torch.randn((bsize, n_targets, nb_channels, n_samples))
    estimates = torch.randn_like(references)

    # Compute various SDRs and display the time
    ts = time.time()
    sdr_fw = compute_sdr(references, estimates, win_bss=win_bss, sdr_type="framewise")[
        0
    ]
    print("Framewise SDR :", time.time() - ts)
    ts = time.time()
    sdr_gl = compute_sdr(references, estimates, win_bss=win_bss, sdr_type="global")[0]
    print("Global SDR :", time.time() - ts)
    ts = time.time()
    sdr_v4 = compute_sdr(references, estimates, win_bss=win_bss, sdr_type="museval")[0]
    print("Museval SDR :", time.time() - ts)

# EOF
