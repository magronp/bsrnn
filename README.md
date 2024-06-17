---
google-site-verification: lLdQLxVzN75s98OondpQ8bVOY-R_Zo9hIfQdK5HrGi4
---

# Band-Split RNN for music separation - yet another unofficial implementation

This repository contains an unofficial implementation of the [BSRNN](https://arxiv.org/pdf/2209.15174.pdf) model for music separation. This project is based on [PyTorch](https://pytorch.org/) ([Ligthning](https://lightning.ai/docs/pytorch/stable/)) and [Hydra](https://hydra.cc/), and uses the HQ version of the freely available [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset.

<div style="align: center; text-align:center;">
    <img src="https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/-/raw/master/Figure/BSRNN.png" width="500px" />
    <div class="caption"><i>Image taken from the <a href="https://arxiv.org/pdf/2209.15174.pdf">BSRNN paper</a>.</i></div>
</div>

&nbsp;

The goal of this project is to reproduce the original results from the BSRNN paper. Unfortunately, we were unable to achieve the performance reported in the paper. Therefore, if you spot an error in the code, or something that differs from the description in the paper, please feel free to reach out, send a message, or open an issue. :slightly_smiling_face:

We provide pretrained models on a [Zenodo repository](https://zenodo.org/records/10992913), which you can use for [separating your own song](#inference--demo).


##  Test results

Here we report the results on the test set (check [below](#metrics) for more information on the metrics).


|             |  uSDR - paper |    uSDR - ours    |  cSDR - paper |    cSDR - ours    |
|-------------|---------------|-------------------|---------------|-------------------|
| vocals      |   10.0        |      8.5          |   10.0        |      8.1          |
| bass        |    6.8        |      6.3          |    7.2        |      7.8          |
| drums       |    8.9        |      8.5          |    9.0        |      8.2          |
| other       |    6.0        |      5.1          |    6.7        |      5.2          |
| average     |    7.9        |      7.1          |    8.2        |      7.3          |


We are about 0.8-0.9 dB SDR behind the results reported in the paper. A similar conclusion can be drawn from [another unofficial implementation](https://github.com/amanteur/BandSplitRNN-Pytorch) with worse results. Thus, some efforts are still needed to reproduce the BSRNN results. 


## Tuning the model

We report the results obtained with several variants (learning rate, architecture, training loss...) in a [separate document](tuning.md). We also provide several [suggestions](tuning.md#potential-improvements) to improve the results in order to match those of the original paper.


## Metrics

As in the original BSRNN paper, we consider two variants of the SDR:
- the *utterance SDR* (uSDR), which is used as metric in the latest MDX challenges. This SDR does not allow any distortion filter (thus it is similar to a basic SNR), and it is computed on entire tracks (no chunking) and averaged over tracks.
- the *chunk SDR* (cSDR), which is computed using the [museval](https://github.com/sigsep/sigsep-mus-eval) tooblox. It was used as metric in the [SiSEC 2018](https://sisec.inria.fr/2018-professionally-produced-music-recordings/) challenge. This SDR allows for a global distortion filter, and it is computed by taking the median over 1s-long chunks, and median over tracks.




## How to use

### Setup

Clone this repo, create and activate a virtual environment, and install the required packages:

```
pip install -r requirements.txt
```

Then, download the [MUSDB18HQ](https://zenodo.org/records/3338373) dataset and unzip it in the `data` folder (or change the strucure and path accordingly in the config file). If you want to skip training and simply perform [evaluation](#evaluation) or [inference](#inference--demo) for a quick demo, you can download pretrained models on the [Zenodo repository](https://zenodo.org/records/10992913).

To speed up data loading at training, you need to pre-process the dataset in order to extract non-silent segment indices. To that end, simply run:
```
python prep_dataset.py
```

### 


### Training

The core training function can be simply run as follows:
```
python train.py
```
This will train the default target (=vocals) using default parameters for both the optimizer and model architecture (check the conf files for more information).

For debugging / fast prototyping / overfitting on purpose, you can use the `fast_tr` flag as follows:
```
python train.py fast_tr=True
```
This enables the [overfit_batches](https://lightning.ai/docs/pytorch/stable/common/trainer.html#overfit-batches) functionnality of Ligthning to perform training and validation on a single batch (this also disables random operations when creating the dataset for ensuring it's the same batch at each step/epoch). 


### Trying multiple configurations

Thanks to the Hydra framework, you can easily change parameters (model size, number of layer, learning rate, etc.), via either the configuration files, or directly in command line, for instance:
```
python train.py optim.loss_domain=t+tf src_mod.num_repeat=10
```
Have a look at the config files to check all the parameters you can change! If you want to train all target models using default parameters, simply run:

```
python train.py -m src_mod.target=vocals,bass,drums,other
```

The list of all model/configuration variants used when presenting the [validation results](#tuning.md) are stored in the `jobs/params.txt` file. This file can be used as a parameter array when running multiple jobs using the [OAR](https://oar.imag.fr/docs/latest/user/quickstart.html) task manager (see the `jobs/book_training` script). Depending on your working environment, this script might need some adaptation. Alternatively, you can simply run each job independently as on the example above, using all the configurations in the `jobs/params.txt` file.

Lastly, we prepare a script that can easily aggregate all validation results from tensorboard logs into a csv file for comparing variants, and display them:
```
python display_tbresults.py
```

### Evaluation

Once all target models are trained, perform evaluation on the test set by running:
```
python evaluate.py
```
When creating a Separator module, the code search for target-specific checkpoints in the output folder, whose path is `outputs/bsrnn/<target>.ckpt`. If a certain checkpoint is not found, a model will be initialized from scratch with random weights instead. The function above computes the uSDR by default, but you can easily compute the cSDR:
```
python evaluate.py eval.sdr_type=museval
```

### Inference / demo

For convenience and quick demo-ing, we provide a script `inference.py` to process a mixture. The basic usage is:
```
python inference.py +file_path=path/to/my/file.wav +rec_dir=dir/to/record/estimates
```
You can specify an offset and a maximum duration (in seconds) with the `offset` and `max_len` parameters (by default, the whole song is processed). By default, it outputs all 4 tracks, but you can only compute some targets of interest, e.g.:
```
python inference.py +file_path=path/to/my/file.wav +rec_dir=dir/to/record/estimates targets=[vocals,bass]
```

The function loads checkpoints as for the evaluation process described [above](#evaluation). You can dowload pretrained checkpoints from the [Zenodo repository](https://zenodo.org/records/10992913), and place them in the `outputs/bsrnn/` folder.


## Hardware

All computation were carried out using the [Grid5000](https://www.grid5000.fr) testbed, supported by a French scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations.

In particular, for training the models we use 4 Nvidia RTX 2080 Ti (11 GiB) GPUs, except the [larger ones](#model-size) that were trained with  2 Nvidia A40 (45 GiB) GPUs


## Acknowledgments

Our implementation relies on code from external sources.

- We adapated the attention mechanism from the TFGridNet implementation in the [ESPNET toolbox](https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18)
- We implemented the BSRNN-related classes (ResNet, BSNet, BSRNN) using the authors' [repository from the MDX challenge](http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit). Note however that this code *needs to be adapted* so that it outputs both time-domain and TF domain components, which are necessary to compute the loss.
- For the source activity detector used in preparing the dataset, we largely relied on the [implementation from Amantur Amatov](https://github.com/amanteur/BandSplitRNN-Pytorch).

We would like to thank Jianwei Yu, who is an author of the BSRNN paper, for trying to help us with the implementation. We also thank Christopher Landschoot for fruitful discussion related to his [own implementation](https://github.com/crlandsc/Music-Demixing-with-Band-Split-RNN).
