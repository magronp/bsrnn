# Band-Split RNN for music separation - yet another unofficial implementation

This repository contains an unofficial Pytorch implementation of the [BSRNN](https://arxiv.org/pdf/2209.15174.pdf) model for music separation. The models are trained on the freely available [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset.

<div style="align: center; text-align:center;">
    <img src="https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/-/raw/master/Figure/BSRNN.png" width="500px" />
    <div class="caption"><i>Image taken from the <a href="https://arxiv.org/pdf/2209.15174.pdf">BSRNN paper</a>.</i></div>
</div>


##  Test results

Here we report the results in terms of global SDR, which is referred to as *utterance SDR* in the [BSRNN paper](https://arxiv.org/pdf/2209.15174.pdf) (note that our implementation allows for computing the *chunk SDR* using the museval toolbox, see below). It is used as metric in the latest MDX challenges. This SDR is computed on whole tracks and doesn't allow any filtering, thus it is similar to the a basic SNR. Then we take the mean over tracks.

|             |   BSRNN paper  | Our implementation |
|-------------|----------------|--------------------|
| vocals      |   10.04        |      7.67          |
| bass        |    6.80        |      5.77          |
| drums       |    8.92        |      8.26          |
| other       |    6.01        |      4.33          |
| all sources |    7.94        |      6.51          |

Below we also report the results in terms *chunk SDR*, computed using the museval tooblox. It was used as metric in the [SiSEC 2018](https://sisec.inria.fr/2018-professionally-produced-music-recordings/) challenge. This SDR allows for a global distortion filter, and then is computed by taking the median over 1second chunks, and median over tracks.

|             |   BSRNN paper  | Our implementation |
|-------------|----------------|--------------------|
| vocals      |   10.01        |      7.19          |
| bass        |    7.22        |      6.57          |
| drums       |    9.01        |      7.59          |
| other       |    6.70        |      4.42          |
| all sources |    8.24        |      6.44          |


Thus, we are about 1.4-1.8 dB SDR behind the baseline implementation.

## Validation results


Here we display the results on the validation set (global SDR) for different variants (all on the vocals track).

### Loss and model size

First, we check the influence of the loss. The origninal paper uses a combination of a time-domain (t) and time-frequency domain (tf) loss (t+tf). We check using each term individually. For speed, this experiment uses a model with hidden dimension of 64 and num_repeats of 8.

| loss    |   SDR   |
|---------|---------|
|  t      |   7,40  |
|  tf     |   6,41  |
|  t+tf   |   7.20  |

We observe that using a time-domain loss only is better than the loss used in the paper (t+tf). Nonetheles, let's use it anyway in what follows.


### Model size

We now increase the  architecture (hidden dimension and number of BS blocks).

| feature_dim  |  num_repeat    |   SDR   |
|--------------|----------------|---------|
|    64        |       8        |   7.20  |
|    64        |       10       |   6.95  |
|    128       |       8        |   5.42  |

We also note that unfortunately, when increasing the model size, the performance degrades, thus we can't reproduce the paper's results.



### Band and sequence modeling layers

Here we investigate on the usage of alternative layers to LSTM for band and sequence modeling. We use the t+tf loss, and the best architecture obtained above (8 repeats, 64 hidden dim).

| sequence modeling layer | band modeling layer |   SDR   |
|-------------------------|---------------------|---------|
|  lstm                   |   lstm              |   7.20  |
|  gru                    |   lstm              |   6.83  |
|  conv                   |   lstm              |   6.24  |
|  lstm                   |   gru               |   6.79  |
|  lstm                   |   conv              |   6.09  |

Nothing is better than the basic LSTM. The Conv1D layers don't work (although they are much faster to train because of memory constraints), and GRU work similarly, though they are slightly faster to train than LSTM, because slightly less parameters.

### Attention mechanism

As a prospective attempt to further boost the results, we propose to use a multi-head attention mechanism, inspired from the TFGridNet model. This model is quite similar to BSRNN, as it projects frequency bands in a deep embedding space, and then applies LSTM over both time and band dimensions.

<div style="align: center; text-align:center;">
    <img src="https://www.researchgate.net/publication/363402998/figure/fig1/AS:11431281083662730@1662694210541/Proposed-full-band-self-attention-module_W640.jpg" width="400px" />
    <div class="caption"><i>Image taken from the <a href="https://arxiv.org/abs/2209.03952">TFGridNet paper</a>.</i></div>
</div>

Below we investigate the impact of the number of attention heads, as well as the dimension of the attention encoder.

| number of heads | attention encoder dim |  SDR    |
|-----------------|-----------------------|---------|
|  0              |           -           |  7.20   |
|  1              |           4           |  7.45   |
|  1              |           10          |  7.71   |
|  1              |           20          |  7.35   |
|  2              |           4           |  7.49   |
|  2              |           10          |  7,05   |
|  2              |           20          |  7.29   |

We observe that adding one attention head may bring up to 0.5 dB improvement, which is quite noticeable. However, it should be noted that this applies to the vocals track, but not necessarily to the other tracks (in preliminary experiments, attention heads was not beneficial for the bass and other tracks). We report test results without attention, but this mechanism should be considered for further boosting the results.

## How to use

### Setup

After cloning this repo, create and activate a virtual env and install the required packages:

```
pip install -r requirements.txt
```

Then, download the [MUSDB18HQ](https://zenodo.org/records/3338373) dataset and unzip it in the `data` folder (or change the strucure and path accordingly in the config file).

Finally, to speed up data loading at training, you will need to pre-process the dataset in order to extract non-silent segment indices. To that end, simply run:
```
python prep_dataset.py
```


### Training

The core training function can be simply run as follows:
```
python train.py
```
that wil train the default target (=vocals) using default parameters (= those used for reporting [test results](#test-results)).

To have with debugging, you can use the `fast_tr` flag as follows:
```
python train.py fast_tr=True
```
This enables the [overfit_batches](https://lightning.ai/docs/pytorch/stable/common/trainer.html#overfit-batches) functionnality of Ligthning to perform training and validation on a single batch (this also disables random operations when creating the dataset for ensuring it's the same batch at each step/epoch). 


### Trying multiple configurations

This project is based on [PyTorch](https://pytorch.org/) ([Ligthning(](https://lightning.ai/docs/pytorch/stable/)) and [Hydra](https://hydra.cc/). This means you can easily change parameters (model size, number of layer, learning rate, etc.), via either the configuration files, or directly in command line, for instance:
```
python train.py optim.loss_domain=t+tf src_mod.num_repeat=10
```
Have a look at the config files to check all the parameters you can change! If you want to train all target models using default parameters, simply run:

```
python train.py -m src_mod.target=vocals,bass,drums,other
```

The list of all model/configuration variants used when presenting the [validation results](#validation-results) are stored in the `jobs/params.txt` file. This file can be used as a parameter array when running multiple jobs using the [OAR](https://oar.imag.fr/docs/latest/user/quickstart.html) task manager (see the `jobs/book_training` script). Depending on your working environment this script might need some adaptation. Alternatively, you can simply run each job independently as above:

You can then run:
```
python display_tbresults.py
```

in order to aggregate all validation results from tensorboard logs into a csv file for comparing variants (and display them).

### Evaluation

Once all target models are trained, to perform evaluation on the test set, run:
```
python evaluate.py
```
Note that when creating a Separator module, the code looks for target-specific checkpoints in the `output/bsrnn` folder. If a certain checkpoint is not found, a model will be initialized from scratch with random weights instead. The function above computes the global SDR (=uSDR) by default, but you can easily compute the museval SDR (=cSDR) as follows:
```
python evaluate.py sdr_type=museval
```


## Hardware

All computation were carried out using the [Grid5000](https://www.grid5000.fr) testbed, supported by a French scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations.

In particular, for training the models we use 4 Nvidia RTX 2080 Ti (11 GiB) GPUs, except the [larger ones](#model-size) that were trained with  2 Nvidia A40 (45 GiB) GPUs


## Acknowledgments

In our implementation we have used some code from external sources.

- We adapated the attention mechanism from the TFGridNet implementation in the [ESPNET toolbox](https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18)
- We implemented the BSRNN-related classes (ResNet, BSNet, BSRNN) using the authors' [repository from the MDX challenge](http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit) (note however that we had to adapt it so that it outputs both time-domain and TF domain components, necessary to compute the loss).
- For the source activity detector used in preparing the dataset, we largely relied on the [implementation from Amantur Amatov](https://github.com/amanteur/BandSplitRNN-Pytorch).

We would like to thank Jianwei Yu, who is an author of the BSRNN paper, for trying to help us with the implementation. We also thank Christopher Landschoot for fruitful discussion related to his [own implementation](https://github.com/crlandsc/Music-Demixing-with-Band-Split-RNN).
