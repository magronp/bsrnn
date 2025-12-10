# An Open - and Optimized - Implementation of Band-Split RNN for Music Separation

<div style="align: center; text-align:center;">
    <img src="https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/-/raw/master/Figure/BSRNN.png" width="500px" />
    <div class="caption"><i>Image taken from the <a href="https://arxiv.org/pdf/2209.15174.pdf">BSRNN paper</a>.</i></div>
</div>

&nbsp;

This repository is an unofficial implementation of the [BSRNN](https://arxiv.org/pdf/2209.15174.pdf) model for music separation. It accompanies our [replication study](#reference), whose primary goal is to obtain a model that yields similar results to those of the original BSRNN paper, and to explore reproducibility issues in music separation research.

Despite our efforts, we are currently about [0.5 dB SDR bellow](#test-results) the original results, thus some work is still needed to match these. To bridge this performance gap, we proposed several variants and eventually obtained an *optimized* model that largely improves the results.

This project is based on [PyTorch](https://pytorch.org/) ([Ligthning](https://lightning.ai/docs/pytorch/stable/)) and [Hydra](https://hydra.cc/), and uses the HQ version of the freely available [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset. We provide pretrained models on a [Zenodo repository](https://zenodo.org/records/13903584), which you can use readily for [separating your own song](#separation--demo).

The goal of this project is to foster reproducible research, to allow other researchers to experiment with this model (and variants), and to provide a fully fonctionning training pipeline and checkpoints for inference. Then feel free to use it (and [cite it](#reference) if you do), and if you spot an error, or something that differs from the description in the paper, please feel free to reach out, send a message, or open an issue. :slightly_smiling_face: 


## Contents

1. [Performance](#performance)
   1. [Test results](#test-results)
   2. [Model variants](#model-variants)
2. [How to use](#how-to-use)
   1. [Setup](#setup)
   2. [Training and evaluation](#training-and-evaluation)
   3. [Separation / demo](#separation--demo)
3. [Ressources](#ressources)
   1. [Hardware](#hardware)
   2. [Related repositories](#related-repositories)
4. [Acknowledgments](#acknowledgments)
5. [Reference](#referenced-repositories)


## Performance

### Test results

The table below displays results on the MUSDB18-HQ test set in terms of signal-to-distortion ratio (SDR). More precisely, we consider the *chunk* SDR, which is computed by taking the median over 1s-long chunks, and median over tracks. In practice, computation is performed using the [museval](https://github.com/sigsep/sigsep-mus-eval) tooblox, but we provide a [more efficient implementation](https://github.com/magronp/bsrnn/blob/main/helpers/eval.py#L28) if only the SDR is needed. Complementary results in terms of *utterance* SDR are available in [our paper](#reference).

|                              |  vocals |   bass  |  drums  |  other  | average |
|------------------------------|---------|---------|---------|---------|---------|
|  BSRNN - original results    |  10.01  |   7.22  |   9.01  |   6.70  |   8.24  |
|  BSRNN - our implementation  |   9.14  |   7.72  |   8.07  |   5.68  |   7.65  |
|  **oBSRNN**                  |   **9.81**  |   **9.85**  |  **10.31**  |   **6.31**  |   **9.07**  |
 
Our optimized model (**oBSRNN**) model includes a multi-head attention mechanism, a TAC module for stereo-awareness, and it is trained using a non-preprocessed dataset (see [here](docs/analysis.md#optimized-model) for more details). This substantially improves performance over our initial BSRNN implementation, and it largely outperforms the paper's results by ~0.8 dB. This improvement is mostly due to large SDR increase in the bass and drums estimates, while the vocals and other results are still inferior to those in the original paper.

We also propose an optimized replication of the SIMO variant of BSRNN (see the [original SIMO-BSRNN paper](https://ieeexplore.ieee.org/document/10447771) and [our implementation](docs/analysis.md#simo-bsrnn) for more details).

|                                |  vocals |   bass  |  drums  |  other  | average |
|--------------------------------|---------|---------|---------|---------|---------|
|  SIMO-BSRNN - original results |   9.73  |   7.80  |  10.06  |   6.56  |   8.54  |
|  **oBSRNN-SIMO**               |  **10.66**  |   **9.73**  |  **10.98** |   **7.78**  |   **9.79**  |

This model largely outperforms the original results, and it yields state-of-the-art results without requiring extra private data. Thus we encourage to consider it as baseline, or for use if achieving maximum performance is the goal.


### Model selection

We extensively experiment with model variants, and we report and analyze the results in a [dedicated document](docs/analysis.md). Beyond reproducing the paper's results, we provide several suggestions to further improving the results by additional architecture variants, as well as optimizing the data preparation and training process.


## How to use

### Setup
Start by cloning this repository, creating/activating a virtual environment, and installing the required packages:

```
pip install -r requirements.txt
```

On linux you will also need to install ffmpeg (needed for museval / musdb):
```
sudo apt install ffmpeg
```


### Training and evaluation
For clarity, we provide a guide for model training and evaluation in a [separate document](docs/training.md).

### Separation / demo

If you simply want to use BSRNN to separate your favorite song, then make sure to download the pretrained checkpoints from the [Zenodo repository](https://zenodo.org/records/13903584), and place them in the `outputs/` folder. Then, perform separation as follows:
```
python separate.py file=path/to/my/file.wav
```
You can specify:
- an offset and a maximum duration (in seconds) with the `offset` and `duration` parameters (by default, the whole song is processed).
- the directory where the separated tracks will be stored `rec_dir` (by default, it is the current working directory).
- which `targets` to extract (by default, all four tracks `vocals`, `bass`, `drums`, and `other` are estimated).
- the folder where checkpoints are located, which is `<out_dir>/<model_dir>/`. You can change both `out_dir` (default: `outputs`) and `model_dir` (default: `bsrnn-opt`).

**Note**: if you want to use the SIMO model, you need to add an extra flag `simo=true`, so that the code loads a multi-source checkpoint named `separator.ckpt` instead of multiple single-source checkpoints named `<target>.ckpt`, e.g.:
```
python separate.py file=path/to/my/file.wav model_dir=simo-bsrnn-opt simo=true
```

## Ressources

### Hardware
All computation were carried out using the [Grid5000](https://www.grid5000.fr) testbed, supported by a French scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations.

Most models are trained using either 4 Nvidia RTX 2080 Ti (11 GiB), except for the small [drums model with attention](docs/analysis.md#attention-mechanism), which uses 4 Nvidia Tesla T4 (15 GiB) GPUs. The [larger](docs/analysis.md#large-model) / [optimized](docs/analysis.md#optimized-model) models are trained using 2 Nvidia Tesla L40S (45 GiB) GPUs.

### Related repositories

Our implementation relies on code from external sources.

- We implemented the BSRNN-related classes (ResNet, BSNet, BSRNN) using the authors' [repository from the MDX challenge](http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit). Note however that this code *needs to be adapted* so that it outputs both time-domain and TF domain components, which are necessary to compute the loss.
- We adapated the attention mechanism from the TFGridNet implementation in the [ESPNET toolbox](https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18).
- The multi-head sequence module is adapted from the [DTTNet](https://github.com/junyuchen-cjy/DTTNet-Pytorch) code (and shares similarities with TFGridNet).
- For the source activity detector used in preparing the dataset, we largely relied on the [implementation from Amantur Amatov](https://github.com/amanteur/BandSplitRNN-Pytorch).


## Acknowledgments

We thank Jianwei Yu (author of the BSRNN paper) for trying to help us with the implementation. We also thank Christopher Landschoot for fruitful discussion related to his [own implementation](https://github.com/crlandsc/Music-Demixing-with-Band-Split-RNN). Finally, we thank Stefan Uhlich for discussions on [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) and the [MDX challenge](https://arxiv.org/pdf/2308.06979), which were helpful in improving our work overall.


## Reference

If you use this code, please cite our paper:

```latex
@article{MagronBSRNN,  
  author={Paul Magron and Romain Serizel and Constance Douwes},  
  title={The Costs of Reproducibility in Music Separation Research: a Replication of Band-Split {RNN}},
  journal={under review},  
  url = {lien arxiv}
}
```

You can use the acronym **oBSRNN** (or **oBSRNN-SIMO**) to refer to our implementation, e.g., to report the corresponding [results](#test-results), where "o" stands for both "open" and "optimized". This allows to make a distinction with the original paper's model / results.
