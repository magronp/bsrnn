import torch
import museval
import pandas as pd
import math
import time


def mypad(input_tensor, tot_len):
    input_tensor = torch.nn.functional.pad(
        input_tensor, (0, tot_len - input_tensor.shape[-1])
    )
    return input_tensor


def compute_usdr(ref, est, eps=1e-7):
    """
    The utterance SDR is a basic SNR (no distortion filter, no framewise computation), as in MDX21/23 challenges.
    (it is called "new SDR" (nSDR) in DEMUCS or "global SDR" in MDX)
    ref / est: [batch_size, n_targets, n_channels, n_samples]
    output: [batch_size, n_targets]
    """
    num = torch.sum(torch.square(ref), dim=(2, 3)) + eps
    den = torch.sum(torch.square(est - ref), dim=(2, 3)) + eps
    sdr = 10 * torch.log10(num / den)

    return sdr


def compute_csdr_fast(ref, est, win=1 * 44100, hop=1 * 44100, eps=1e-7):
    """
    cSDR: framewise, no distortion filter, and median over frames
    Here we skip the distortion filter computation (not useful for the SDR), thus it's faster than the museval
    ref / est: [batch_size, n_targets, n_channels, n_samples]
    output: [batch_size, n_targets]
    """
    dif = est - ref

    # Pad if needed so the signal length is a multiple of win_bss (crop the last bit, as in museval)
    tot_len = win + math.floor((ref.shape[-1] - win) / hop) * hop
    ref = mypad(ref, tot_len)
    dif = mypad(dif, tot_len)

    # Chunking into overlapping frames
    ref = ref.unfold(-1, win, hop)
    dif = dif.unfold(-1, win, hop)

    # SDR for each frame
    num = torch.sum(torch.square(ref), dim=(2, 4)) + eps
    den = torch.sum(torch.square(dif), dim=(2, 4)) + eps
    sdr_frames = 10 * torch.log10(num / den)

    # Get the median over frames
    sdr = torch.nanmedian(sdr_frames, dim=-1)[0]

    return sdr


def compute_csdr_museval(ref, est, win=1 * 44100, hop=1 * 44100):
    """
    cSDR: framewise, no distortion filter, and median over frames
    We use the museval toolbox, as customary in music separation papers
    ref / est: [batch_size, n_targets, n_channels, n_samples]
    output: [batch_size, n_targets]
    """

    # Reshape to [batch_size, n_targets, n_samples, n_channels], and back to cpu/numpy
    ref = ref.cpu().transpose(2, 3).numpy()
    est = est.cpu().transpose(2, 3).numpy()

    # Need to loper over batch samples
    sdr = []
    batch_size = ref.shape[0]
    for ib in range(batch_size):
        sdr_frames = museval.metrics.bss_eval(
            ref[ib],
            est[ib],
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False,
        )[0]
        sdr_frames = torch.from_numpy(sdr_frames)
        sdrb = torch.nanmedian(sdr_frames, dim=1)[0]  # Median over frames
        sdr.append(sdrb)
    sdr = torch.stack(sdr)

    return sdr


def compute_sdr(
    references, estimates, win=1 * 44100, hop=1 * 44100, type="usdr", eps=1e-7
):
    """
    references / estimates: [batch_size, n_targets, n_channels, n_samples]
    sdr: [batch_size, n_targets]
    """

    # Make sure there are integers
    win = int(win)
    hop = int(hop)

    # Compute the SDR over batches
    if type == "usdr":
        sdr = compute_usdr(references, estimates, eps=eps)
    elif type == "csdr":
        sdr = compute_csdr_museval(references, estimates, win=win, hop=hop)
    elif type == "csdr-fast":
        sdr = compute_csdr_fast(references, estimates, win=win, hop=hop, eps=eps)
    else:
        raise NameError("Unknown SDR type")

    # Aggregate results across batch samples (mean/median)
    if "csdr" in type:
        sdr = torch.nanmedian(sdr, dim=0)[0]
    else:
        sdr = torch.nanmean(sdr, dim=0)

    return sdr


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


def aggregate_res_over_tracks(path_or_df, method_name=None, sdr_type="usdr"):

    # The results are either directly provided as dataframe, or as a path to a file
    if isinstance(path_or_df, str):
        test_results = pd.read_csv(path_or_df)
    else:
        test_results = path_or_df

    # Aggregate scores over tracks (mean for the uSDR, median otherwise)
    if sdr_type == "usdr":
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


if __name__ == "__main__":
    # Realistic signal parameters
    sample_rate = 44100
    bsize = 1
    n_targets = 4
    nb_channels = 2
    win = 1.7 * sample_rate
    hop = 0.6 * sample_rate
    n_samples = int(sample_rate * 3.1)

    # Create references and estimates signals
    torch.manual_seed(0)
    references = torch.randn((bsize, n_targets, nb_channels, n_samples))
    estimates = torch.randn_like(references)

    # Compute various SDRs and display the time
    ts = time.time()
    usdr = compute_sdr(references, estimates, win=win, hop=hop, type="usdr")
    print(f"Utterance SDR : {usdr} dB --- Time: {time.time() - ts:.2f} s")

    ts = time.time()
    csdrmuseval = compute_sdr(references, estimates, win=win, hop=hop, type="csdr")
    print(f"Chunk SDR museval: {csdrmuseval} dB --- Time: {time.time() - ts:.2f} s")

    ts = time.time()
    csdr = compute_sdr(references, estimates, win=win, hop=hop, type="csdr-fast")
    print(f"Chunk SDR, fast: {csdr} dB --- Time: {time.time() - ts:.2f} s")

# EOF
