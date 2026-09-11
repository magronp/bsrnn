import torch
import museval
import math
import time


# Zero-padding function
def mypad(x, total_len):
    x_padded = torch.nn.functional.pad(x, (0, total_len - x.shape[-1]))
    return x_padded


def any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return torch.any(
        torch.all(torch.sum(sources, dim=tuple(range(2, sources.ndim))) == 0, dim=1)
    ).item()


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

    return sdr, None


def compute_csdr_fast(ref, est, win=1 * 44100, hop=1 * 44100, eps=1e-7):
    """
    cSDR: framewise, no distortion filter, and median over frames
    Here we skip the distortion filter computation (not useful for the SDR), thus it's faster than the museval
    ref / est: [batch_size, n_targets, n_channels, n_samples]
    output: [batch_size, n_targets]
    """

    # Make sure these are integers
    win = int(win)
    hop = int(hop)

    # Pad if needed so the signal length is a multiple of win_bss (crop the last bit, as in museval)
    tot_len = win + math.floor((ref.shape[-1] - win) / hop) * hop
    ref = mypad(ref, tot_len)
    est = mypad(est, tot_len)

    # Chunking into overlapping frames
    ref = ref.unfold(-1, win, hop)
    est = est.unfold(-1, win, hop)

    # SDR for each frame
    num = torch.sum(torch.square(ref), dim=(2, 4)) + eps
    den = torch.sum(torch.square(est - ref), dim=(2, 4)) + eps
    sdr_frames = 10 * torch.log10(num / den)

    # If silent ref/est, set the value at "nan" to then discard it (as in museval)
    for f in range(sdr_frames.shape[-1]):
        if any_source_silent(est[..., f, :]) or any_source_silent(ref[..., f, :]):
            sdr_frames[..., f] = torch.nan

    # Get the median over frames
    sdr = torch.nanmedian(sdr_frames, dim=-1)[0]

    return sdr, sdr_frames


def compute_csdr_museval(ref, est, win=1 * 44100, hop=1 * 44100):
    """
    cSDR: framewise, no distortion filter, and median over frames
    We use the museval toolbox, as customary in music separation papers
    ref / est: [batch_size, n_targets, n_channels, n_samples]
    output: [batch_size, n_targets]
    """

    # Make sure these are integers
    win = int(win)
    hop = int(hop)

    # Reshape to [batch_size, n_targets, n_samples, n_channels], and back to cpu/numpy
    ref = ref.cpu().transpose(2, 3).numpy()
    est = est.cpu().transpose(2, 3).numpy()

    # Need to loper over batch samples
    sdr, sdr_frames = [], []
    batch_size = ref.shape[0]
    for ib in range(batch_size):
        sdr_framesb = museval.metrics.bss_eval(
            ref[ib],
            est[ib],
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False,
        )[0]
        sdr_framesb = torch.from_numpy(sdr_framesb)
        sdrb = torch.nanmedian(sdr_framesb, dim=1)[0]  # Median over frames
        sdr_frames.append(sdr_framesb)
        sdr.append(sdrb)
    sdr = torch.stack(sdr)
    sdr_frames = torch.stack(sdr_frames)

    return sdr, sdr_frames


def compute_sdr(
    references, estimates, win=1 * 44100, hop=1 * 44100, type="usdr", eps=1e-7
):
    """
    references / estimates: [batch_size, n_targets, n_channels, n_samples]
    sdr: [batch_size, n_targets]
    type should be "usdr", "csdr" (=museval), or "csdr-fast" (= not computing the distortion filter)
    """

    # Make sure these are integers
    win = int(win)
    hop = int(hop)

    # Compute the SDR over batches
    if type == "usdr":
        sdr, sdr_frames = compute_usdr(references, estimates, eps=eps)
    elif type == "csdr":
        sdr, sdr_frames = compute_csdr_museval(references, estimates, win=win, hop=hop)
    elif type == "csdr-fast":
        sdr, sdr_frames = compute_csdr_fast(
            references, estimates, win=win, hop=hop, eps=eps
        )
    else:
        raise NameError("Unknown SDR type")

    # Aggregate results across batch samples (mean/median)
    if "csdr" in type:
        sdr = torch.nanmedian(sdr, dim=0)[0]
    else:
        sdr = torch.nanmean(sdr, dim=0)

    return sdr, sdr_frames


if __name__ == "__main__":
    # Signal parameters
    sample_rate = 44100
    bsize = 1
    n_targets = 4
    nb_channels = 2
    win = 1 * sample_rate
    hop = 1 * sample_rate
    n_samples = int(sample_rate * 10)

    # Create references and estimates
    torch.manual_seed(0)
    references = torch.randn((bsize, n_targets, nb_channels, n_samples))
    estimates = torch.randn_like(references)

    # Compute various SDRs and display the time
    ts = time.time()
    usdr = compute_sdr(references, estimates, win=win, hop=hop, type="usdr")
    print(f"Utterance SDR : {usdr} dB --- Time: {time.time() - ts:.2f} s")

    ts = time.time()
    csdrmuseval = compute_sdr(references, estimates, win=win, hop=hop, type="csdr")
    print(f"Chunk SDR, museval: {csdrmuseval} dB --- Time: {time.time() - ts:.2f} s")

    ts = time.time()
    csdr_fast = compute_sdr(references, estimates, win=win, hop=hop, type="csdr-fast")
    print(f"Chunk SDR, fast: {csdr_fast} dB --- Time: {time.time() - ts:.2f} s")

# EOF
