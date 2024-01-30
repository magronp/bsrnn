# BSRNN

<center><a href="https://arxiv.org/pdf/2209.15174.pdf">
    <img src="https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/-/raw/master/Figure/BSRNN.png"></a></center>


## Test results

We report both the *global* and the *museval* SDR:
- The global SDR is computed on the whole track and doesn't account for filtering (thus it is similar to the a basic SNR). It is used as metric in the latest MDX challenges. Then mean over tracks.
- The museval SDR allows for a global distortion filter on the track, and takes the median over segments of 1s. It's the one used in the SISEC 2018 challenge. Then median over tracks.

Global and museval SDRs are respectively refered to as utterance SDR (uSDR) and chunk SDR (cSDR) in the [BSRNN paper](https://arxiv.org/pdf/2209.15174.pdf).


| Target      |   global SDR   | museval SDR    |
|-------------|----------------|----------------|
| vocals      | 6.883          |                |
| bass        | -              | -              |
| drums       | -              | -              |
| other       | -              | -              |
| all sources | -              | -              |


## Validation results

Below we display the results on the validation set (global SDR) for different variants (all on the vocals track).









commandes utiles :
fuser 6006/tcp -k
fuser -v /dev/nvidia*

