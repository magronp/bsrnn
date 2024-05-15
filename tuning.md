# Tuning the model

In this document we investigate several model configurations and hyper parameters. We consider the *vocals* target, and we compute the uSDR (which is much faster to compute than the cSDR) on the validation set (see [here](README.md#metrics) for the definitions of uSDR and cSDR).

For speed, in our experiments we use a small model with hidden dimension of 64 and num_repeats of 8.

### Learning rate

First, we quickly check the impact of the learning rate. In the original paper, the authors use a learning rate of $10^{-3}$ with 8 GPUs in parallel. Since in most of our experiments we use 4 GPUs, theoretically we need to double the batch size compared to the original paper in order to preserve the same overall batch size. However, this is not possible because of memory constraints. As a result, we need to halve the learning rate.

| learning rate |  SDR  |
|---------------|--------|
|  0.001        |   7.4  |
|  0.0005       |   7.5  |

Both learning rates yield similar results, though the small one performs slightly better and is consistent with the paper (as explained above), thus we use this value in the following experiments. 


### Loss and model size

Here we check the influence of the training loss domain. The original paper uses a combination (t+tf) of a time-domain (t) and time-frequency domain (tf) terms, which we check individually. 

| loss    |  SDR  |
|---------|--------|
|  t      |   7,6  |
|  tf     |   7.2  |
|  t+tf   |   7.5  |

We observe that using a time-domain only loss yields better results than the loss used in the paper (t+tf). Nonetheles, we use the same as in the paper in the following experiments for consistency.


### Band and sequence modeling layers

We investigate on the usage of alternative layers to LSTM for band and sequence modeling.

| sequence modeling layer | band modeling layer |   SDR   |
|-------------------------|---------------------|---------|
|  lstm                   |   lstm              |   7.5  |
|  gru                    |   lstm              |   7.2  |
|  conv                   |   lstm              |   6.7  |
|  lstm                   |   gru               |   7.4  |
|  lstm                   |   conv              |   6.9  |

LSTM layers seem to be the best choice. GRU (resp. Conv1D) layers induce a moderate (resp. large) performance drop, although they are also moderately (resp. much) faster to train because of memory constraints.

### Attention mechanism

As a prospective attempt to further boost the results, we propose to use a multi-head attention mechanism, inspired from the [TFGridNet](https://arxiv.org/abs/2209.03952) model. This model is quite similar to BSRNN, as it projects frequency bands in a deep embedding space, and then applies LSTM over both time and band dimensions, but it incorporates an additional multi-head attention mechanism. Below we investigate the impact of the number of attention heads, as well as the dimension of the attention encoder.

| number of heads | attention encoder dim |  SDR   |
|-----------------|-----------------------|---------|
|  0              |           -           |  7.5   |
|  1              |           4           |  7.6   |
|  1              |           10          |  7.8   |
|  1              |           20          |  7.4   |
|  2              |           4           |  7.1   |
|  2              |           10          |  7,4   |
|  2              |           20          |  7.5   |

We observe that adding one attention head may bring up to 0.3 dB improvement, which is quite noticeable. However, it should be noted that this applies to the vocals track, but not necessarily to the other tracks (in preliminary experiments, attention heads was not beneficial for the bass and other tracks). This mechanism is therefore an interesting direction to consider for further boosting the results.

**Note**: Following this idea, a recent approach called [Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612) replaced the LSTM+Attention mechanism with transformers, and obtained significantly better results. This is in line with a trend that consists in completely replacing RNNs with transformers.


### Model size

We now use the original paper's architecture by increasing the model size. Since the model uses significantly more parameters, we We used larger GPUs for training this model in order to accomodate with its size; however, in this case we can only use 2 GPUs in parallel. Therefore, following our [discussion above](#learning-rate), we use a learning rate of $2.5*10^4$ for training this model (the performance is very poor with larger values).

| feature_dim  /  num_repeat    |  SDR  |
|------------------------------|--------|
|    64        /       8        |   7.5  |
|    128       /       12       |   8.4  |

We observe a significant improvement when using this larger model.


### Potential improvements

**Dataset**

In our experiments, we use a similar data preprocessing as suggested in the paper, based on source activity detection (SAD). Note that we also trained the model without such preprocessing, and we obtain similar results, which suggests that this SAD preprocessing implementation can be improved (we used one based on [another repo](https://github.com/amanteur/BandSplitRNN-Pytorch)).

**Monitoring criterion**

It is not clear from the paper which quantity is monitored on the validation set (the authors mention that "early stopping is applied when the *best validation* is not found in 10 consecutive epochs"). We therefore assume that the validation loss is minimized. However, note that if instead one maximizes the validation SDR, the large model above yields a  8.7 dB (instead of 8.4). This is another possible way of improving overall results.

**Chunking for evaluation**

For evaluation (both validation and test), songs are divided into small segments for performing separation, and then the chunked estimates are assembled to form whole songs/sources estimates. To do that, we use chunks of 10 seconds with a 1s overlap and a [linear fader](https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html#configure-the-application-function).

However, this does not correspond to what is done in BSRNN, which is unfortunately hard to understand,  and therefore reproduce. Indeed:
- In the paper, the authors mention some zero-padding at the beginning and end of each chunk before applying the model, and then some overlap-add (OLA) in order to smooth the outputs. However, they don't specify which OLA procedure (e.g., which window is used? does it ensure perfect reconstruction?).
- During our discussion with an author, he suggests to use the [bytedance music separation repo](https://github.com/bytedance/music_source_separation), which includes a [chunk-level separator](https://github.com/bytedance/music_source_separation/blob/master/bytesep/separator.py#L122). But the output chunks are simply cropped and concatenated, which does not involve any OLA, and it uses a fixed overlap ratio of 50%, which makes it not applicable to other ratios.
- In the paper, test results are displayed using 3s chunks with 0.5s hop size, but when discussing with the author he mentioned using a 1s hop size for validation. Thus it is unclear whether the same setup was used for validation and testing, and which value is used exactly.

We can reasonably assume that changing the chunking/OLA procedure will not yield a different optimal model during training: the validation score will be different, but yield the same local minimum. However, the results could be very different at testing: the authors observe a 0.3 dB variation depending on the hop size (*cf* Table II in the paper), which is more than the difference between some methods. 

In this code we implement a simple OLA method which uses rectangular windowing, and we compute the test results (for the vocals track) using a 3s window and 0.5s hop size. We don't observe any significant difference with the 10s-fader based chunking method. Nonotheless, considering the observations above, there might be some room for further improvement.

