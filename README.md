# (Yet another) unofficial implementation of Band-Split RNN for music source separation

This repository contains an unofficial Pytorch implementation of the [BSRNN](https://arxiv.org/pdf/2209.15174.pdf) model for music separation.

<div style="align: center; text-align:center;">
    <img src="https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/-/raw/master/Figure/BSRNN.png" width="500px" />
    <div class="caption"><i>Image taken from the <a href="https://arxiv.org/pdf/2209.15174.pdf">BSRNN paper</a>.</i></div>
</div>



##  Test results

We report both the *global* and the *museval* SDR:
- The global SDR is computed on the whole track and doesn't account for filtering (thus it is similar to the a basic SNR). It is used as metric in the latest MDX challenges. Then mean over tracks.
- The museval SDR allows for a global distortion filter on the track, and takes the median over segments of 1s. It's the one used in the SISEC 2018 challenge. Then median over tracks.

Global and museval SDRs are respectively refered to as *utterance* SDR and *chunk* SDR in the [BSRNN paper](https://arxiv.org/pdf/2209.15174.pdf).


| Target      |   global SDR   | museval SDR    |
|-------------|----------------|----------------|
| vocals      | 6.883          |                |
| bass        | -              | -              |
| drums       | -              | -              |
| other       | -              | -              |
| all sources | -              | -              |


## Validation results


Here we display the results on the validation set (global SDR) for different variants (all on the vocals track).

### Loss and model size

First, we display below the results in terms of loss domain, and architecture (hidden dimension and number of BS blocks).

| loss    |   feature_dim  |  num_repeat    |  SDR    |
|---------|----------------|----------------|---------|
|  t      |      64        |       8        |         |
|  tf     |      64        |       8        |         |
|  t+tf   |      64        |       8        |         |
|  t+tf   |      64        |       10       |   6.96  |
|  t+tf   |      128       |       8        |   5.63  |

We observe that the loss employed in the paper is the best, but not significantly better than simply time-domain. We also note that unfortunately, when increasing the model size, the performance degrades, thus we can't reproduce the paper's results.


### Band and sequence modeling layers

Here we investigate on the usage of alternative layers to LSTM for band and sequence modeling. We use the t+tf loss, and the best architecture obtained above (8 repeats, 64 hidden dim).

| sequence modeling layer | band modeling layer |  SDR    |
|-------------------------|---------------------|---------|
|  lstm                   |   lstm              |         |
|  gru                    |   lstm              |   7.09  |
|  conv                   |   lstm              |   6.41  |
|  lstm                   |   gru               |   7.01  |
|  lstm                   |   conv              |   6.55  |

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
|  0              |           -           |         |
|  1              |           4           |  7.62   |
|  1              |           10          |  7.71   |
|  1              |           20          |  7.53   |
|  2              |           4           |  7.23   |
|  2              |           10          |  7.37   |
|  2              |           20          |  7.56   |

We see that adding one attention head brings some improvement, although it should be noted that this applies to the vocals track, but not necessarily to the other tracks. Thus in our test results we do not use attention, but keep in mind it might be useful.

## Reproducing the results



## Acknowledgments

In our implementation we have used some code from external sources.

- We adapated the attention mechanism from the TFGridNet implementation in the [ESPNET toolbox](https://github.com/espnet/espnet/blob/35c2e2b13026ba212a2ba5e454e1621d30f2d8b9/espnet2/enh/separator/tfgridnet_separator.py#L18)
- We implemented the BSRNN-related classes (ResNet, BSNet, BSRNN) using the authors' [repository from the MDX challenge](http://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit) (note however that we had to adapt it so that it outputs both time-domain and TF domain components, necessary to compute the loss).
- For the source activity detector used in preparing the dataset, we largely relied on the [implementation from Amantur Amatov](https://github.com/amanteur/BandSplitRNN-Pytorch).

We would like to thank Jianwei Yu, who is an author of the BSRNN paper, for trying to help us with the implementation. We also thank Christopher Landschoot for fruitful discussion related to his [own implementation](https://github.com/crlandsc/Music-Demixing-with-Band-Split-RNN).


commandes utiles :
fuser 6006/tcp -k
fuser -v /dev/nvidia*

