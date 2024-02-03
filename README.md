# (Yet another) unofficial implementation of Band-Split RNN for music source separation

This repository contains an unofficial Pytorch implementation of the [BSRNN](https://arxiv.org/pdf/2209.15174.pdf) model for music separation.

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

Below we also report the results in terms *chunk SDR*, computed using the museval tooblox. It was used as metric in the previous SiSEC challenges. This SDR allows for a global distortion filter, and then is computed by taking the median over 1second chunks, and median over tracks.

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

We propose to further boost the results by using a multi-head attention mechanism, inspired from the TFGridNet model, which is very similar to BSRNN (projects frequency bands in a deep embedding space, and then applies LSTM over both time and band dimensions).

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

We see that adding one attention head brings some improvement, although it should be noted that this applies to the vocals track, but not necessarily to the other tracks. Thus in our test results we do not use attention, but keep in mind it might be useful.

## Reproducing the results



## Acknowledgments

In our implementation we have used some code from external sources.

- We adapated the attention mechanism from the TFGridNet implementation in the [ESPNET toolbox](https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18)
- We implemented the BSRNN-related classes (ResNet, BSNet, BSRNN) using the authors' [repository from the MDX challenge](http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit) (note however that we had to adapt it so that it outputs both time-domain and TF domain components, necessary to compute the loss).
- For the source activity detector used in preparing the dataset, we largely relied on the [implementation from Amantur Amatov](https://github.com/amanteur/BandSplitRNN-Pytorch).

We would like to thank Jianwei Yu, who is an author of the BSRNN paper, for trying to help us with the implementation. We also thank Christopher Landschoot for fruitful discussion related to his [own implementation](https://github.com/crlandsc/Music-Demixing-with-Band-Split-RNN).
