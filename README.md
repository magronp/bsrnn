# Replicating Band-Split RNN for music separation

<div style="align: center; text-align:center;">
    <img src="https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/-/raw/master/Figure/BSRNN.png" width="500px" />
    <div class="caption"><i>Image taken from the <a href="https://arxiv.org/pdf/2209.15174.pdf">BSRNN paper</a>.</i></div>
</div>

&nbsp;

This repository is an unofficial implementation of the [BSRNN](https://arxiv.org/pdf/2209.15174.pdf) model for music separation. It complements our paper.

Our primary goal is to reproduce the original results from the BSRNN paper, but we are currently about [0.5 dB SDR bellow](#test-results), (which is why we proposed an "optimized" variant that yields competitive results). Therefore, efforts are still needed to match the original results: if you spot an error, or something that differs from the description in the paper, please feel free to reach out, send a message, or open an issue. :slightly_smiling_face:


This project is based on [PyTorch](https://pytorch.org/) ([Ligthning](https://lightning.ai/docs/pytorch/stable/)) and [Hydra](https://hydra.cc/), and uses the HQ version of the freely available [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset. We provide pretrained models on a [Zenodo repository](https://zenodo.org/records/13903584), which you can use for [separating your own song](#separation--demo).


## Contents

1. [Performance](#performance)
   1. [Test results](#test-results)
   2. [Model variants](#model-variants)
2. [How to use](#how-to-use)
   1. [Setup](#setup)
   2. [Training and evaluation](#training-and-evaluation)
   3. [Separation / demo](#separation--demo)
3. [Hardware](#hardware)
4. [Acknowledgments](#acknowledgments)
5. [Referenced repositories](#referenced-repositories)


## Performance

### Test results

Below we report the results on the MUSDB18-HQ test set in terms of signal-to-distortion ratio (SDR). As in the original BSRNN paper, we consider two variants of the SDR:

**uSDR**: The utterance SDR is used as metric in the latest [MDX challenges](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23). This SDR does not allow any distortion filter (thus it is similar to a basic signal-to-noise ratio), and it is computed on entire tracks (no chunking) and averaged over tracks.

|                      | vocals |  bass  |  drums |  other | average|
|----------------------|--------|--------|--------|--------|--------|
|  paper's results     |  10.0  |   6.8  |   8.9  |   6.0  |   7.9  |
|  our implementation  |   9.2  |   6.5  |   8.6  |   5.4  |   7.4  |
|  optimized           |   9.7  |   7.4  |   9.6  |   5.8  |   8.1  |

**cSDR**: The chunk SDR was used as metric in the [SiSEC 2018](https://sisec.inria.fr/2018-professionally-produced-music-recordings/) challenge. This SDR allows for a global distortion filter, and it is computed by taking the median over 1s-long chunks, and median over tracks. In practice, computation is performed using the [museval](https://github.com/sigsep/sigsep-mus-eval) tooblox.

|                      | vocals |  bass  |  drums |  other | average|
|----------------------|--------|--------|--------|--------|--------|
|  paper's results     |  10.0  |   7.2  |   9.0  |   6.7  |   8.2  |
|  our implementation  |   9.1  |   7.7  |   8.1  |   5.7  |   7.7  |
|  optimized           |   9.9  |   8.9  |   9.2  |   6.1  |   8.5  |


The optimized BSRNN model includes a multi-head attention mechanism, and it is trained using a non-preprocessed dataset (see [here](docs/analysis.md#optimized-model) for more details). This largely improves performance over our initial BSRNN implementation, and outperforms the paper's results by ~0.2 dB. This performance is mostly due to improvement in the bass and drums estimates, while the vocals and other results are still inferior to those in the original paper.

### Model variants

We extensively experiment with model variants, and we report and analyze the results in a [separate document](docs/analysis.md). Beyond reproducing the paper's results, we provide several suggestions to further improving the results by additional architecture variants, as well as optimizing the data preparation and training process.


## How to use

### Setup
Start by cloning this repository, creating/activating a virtual environment, and installing the required packages:

```
pip install -r requirements.txt
```

### Training and evaluation
For clarity, we provie a guide for model training and evaluation in a [separate document](docs/training.md).

### Separation / demo

If you simply want to apply BSRNN to separate your favorite song, then make sure to download the pretrained checkpoints from the [Zenodo repository](https://zenodo.org/records/13903584), and place them in the `outputs/` folder. Then, perform separation as follows:
```
python separate.py file=path/to/my/file.wav
```
You can specify:
- an offset and a maximum duration (in seconds) with the `offset` and `duration` parameters (by default, the whole song is processed).
- the directory where the separated tracks will be stored `rec_dir` (by default, it is the current working directory).
- which `targets` to estimate (by default, all four tracks `vocals`, `bass`, `drums`, and `other`).
- the folder where checkpoints are located, which is `<out_dir>/<model_dir>/`. You can change both `out_dir` (default: `outputs`) and `model_dir` (default: `bsrnn-opt`).





## Hardware

All computation were carried out using the [Grid5000](https://www.grid5000.fr) testbed, supported by a French scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations.

Most models are trained using either 4 Nvidia RTX 2080 Ti (11 GiB), except for the small [drums model with attention](docs/analysis.md#attention-mechanism), which uses 4 Nvidia Tesla T4 (15 GiB) GPUs. The [larger](docs/analysis.md#large-model) / [optimized](docs/analysis.md#optimized-model) models are trained using 2 Nvidia Tesla L40S (45 GiB) GPUs.

## Referenced repositories

Our implementation relies on code from external sources.

- The main BSRNN module definition is adapted from the [authors's repository](https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/) on the MDX 2023 challenge.
- We adapated the attention mechanism from the TFGridNet implementation in the [ESPNET toolbox](https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18).
- We implemented the BSRNN-related classes (ResNet, BSNet, BSRNN) using the authors' [repository from the MDX challenge](http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit). Note however that this code *needs to be adapted* so that it outputs both time-domain and TF domain components, which are necessary to compute the loss.
- The multi-head sequence module is adapted from the [DTTNet](https://github.com/junyuchen-cjy/DTTNet-Pytorch) code (and shares similarities with TFGridNet).
- For the source activity detector used in preparing the dataset, we largely relied on the [implementation from Amantur Amatov](https://github.com/amanteur/BandSplitRNN-Pytorch).

## Acknowledgments

We thank Jianwei Yu (author of the BSRNN paper) for trying to help us with the implementation. We also thank Christopher Landschoot for fruitful discussion related to his [own implementation](https://github.com/crlandsc/Music-Demixing-with-Band-Split-RNN). Finally, we thank Stefan Uhlich for discussions on [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) and the [MDX challenge](https://arxiv.org/pdf/2308.06979), which were helpful in improving our work overall.



# Descriptif Zenodo
Here you can download the pretrained weights (PyTorch Lightning checkpoints) of the BSRNN model for music separation. These are unofficial, and correspond to our implementation, whose code is freely available here. These models are trained using the MUSDB18HQ dataset.

We provide two sets of weights:

bsrnn-large, which correspond to our implementation of the original BSRNN paper
bsrnn-opt, which correspond to an optimized version of BSRNN (using a non preprocessed dataset, and additional attention heads)
