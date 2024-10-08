# Testing various model configurations

In this document we investigate several model configurations and hyper parameters. Our primary goal is to reproduce the paper's results, but we conduct several additional experiments to study the behavior of the model and training process in more depth. We also seek to optimize the model in order to eventually improve the performance over the original BSRNN.

For speed, we display results on the validation set in terms of uSDR, as it is much faster to compute than the cSDR (see [here](README.md#test-results) for the definitions of uSDR and cSDR). We consider a small model with a hidden dimension of 64 and a number of repeats of 8, except when trying the [larger](#large-model) or [optimized](#optimized-model) models.

**Note about the learning rate**: In the original paper, the model is trained using a learning rate of $10^{-3}$, a batch size of 2, and 8 GPUs in parallel, yielding an effective batch size of 16. We don't have access to enough (large) GPUs, so theoretically we should increase the batch size in order to yield the same effective batch size. However, this is not possible because of memory constraints. As a result, we [adjust the learning rate](https://github.com/magronp/bsrnn/blob/main/train.py#L47) to compensate for the drop in effective batch size, so that the effective learning rate is the same as in the paper. Note that we also tried [accumulating gradients](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#accumulate-gradients) to increase the effective batch size instead, but this yielded similar or slightly worse results overall.


## Preliminary tests

### Loss domain

The original paper uses a combination (t+tf) of a time-domain (t) and time-frequency domain (tf) terms. First, we study the influence of the training loss domain by checking these terms individually (and their combination).

| loss    | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
|  t+tf   |   7.5  |   6.4  |   9.3  |  4.8   |   7.0  |
|  t      |   7.6  |   6.3  |   9.4  |  4.7   |   7.0  |
|  tf     |   7.2  |   6.0  |   9.3  |  3.6   |   6.5  |

We observe that using the time-domain only loss yields similar results to the complete t+tf loss. Both approaches perform much better than the tf-only loss, which confirms the [importance of time-domain training](https://arxiv.org/pdf/1911.08895). In the following experiments, we use the t+tf loss in order to be consistent with the paper.

### FFT size

The original BSRNN model uses a fixed FFT size of 2048, while the value 4096 is common for music separation model (UMX, demucs). Thus, we suggest to investigate it since it only moderately increases the model size, while reducing the memory requirements (as the number of time frames is reduced). This allows to increase the batch size and therefore to speed up training.

| n_fft   | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
|  2048   |   7.5  |   6.4  |   9.3  |   4.8  |   7.0  |
|  4096   |   7.0  |   5.3  |   9.1  |   4.4  |   6.4  |

However, it does not yield good results, which we can explain as follows. While the change in frequency resolution is not really impactful since the STFT is projected into an embedding of fixed dimension, the time resolution is diminished, which has a much stronger impact on the results. This also explains why the drop is more significant for the bass track, which requires a refined time resolution, but not on the drums, which are localized events in time.


### Monitoring criterion

It is not clear from the paper which quantity is monitored on the validation set (the authors mention that "early stopping is applied when the *best validation* is not found in 10 consecutive epochs"). Thus, we assumed previously that the validation *loss* was minimized. However, we can instead maximize the validation *SDR*.

|          | vocals |  bass  |  drums |  other | average|
|----------|--------|--------|--------|--------|--------|
| min loss |   7.5  |   6.4  |   9.3  |   4.8  |   7.0  |
| max SDR  |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |

Doing so allows training to continue for more epochs, and yields an average 0.3 dB improvement. We use this strategy in the next experiments.


## Large model

We now use the original paper's architecture by increasing the model size, i.e., a hidden dimension of 128 and a number of repeat of 12 (vs. 64 and 8 for the small model used above).

|           | vocals |  bass  |  drums |  other | average|
|-----------|--------|--------|--------|--------|--------|
|   small   |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|   large   |   9.2  |   7.7  |   9.9  |   6.2  |   8.3  |

We observe a significant improvement when using this larger model. We report the test results using this model in the main [readme document](README.md) as our implementation of the original BSRNN model.


## Further improvements

We now suggest several potential directions for further improving the performance of BSRNN.


### Band and sequence modeling layers

Here we investigate alternative layers to LSTM for band and sequence modeling (GRU or Conv1D instead of LSTM), for the vocal track.

| sequence modeling layer | band modeling layer |   SDR   |
|-------------------------|---------------------|---------|
|  lstm                   |   lstm              |   7.8  |
|  gru                    |   lstm              |   8.0  |
|  conv                   |   lstm              |   6.8  |
|  lstm                   |   gru               |   7.9  |
|  lstm                   |   conv              |   7.0  |

We observe that GRUs yield a slight performance improvement, and might be considered as an interesting alternative to LSTMs (in what follows we still use LSTMs for consistency with the original paper). On the other hand, Conv1D layers induce a large performance drop, although they are also much faster to train because of memory constraints. Note that the usage of Conv1D layers is quite naïve, and instead one could replace recurrent networks with [dilated convolutions](https://arxiv.org/pdf/2306.05887).


### Attention mechanism

We propose here to use a multi-head attention mechanism, inspired from the [TFGridNet](https://arxiv.org/abs/2209.03952) model. Indeed, TFGridNet is quite similar to BSRNN, as it projects frequency bands in a deep embedding space, and then applies LSTM over both time and band dimensions. However, it incorporates an additional multi-head attention mechanism, which we propose to incorporate to BSRNN here. We investigate the impact of the number of attention heads ($H$), as well as the dimension of the attention encoder ($E$) onto performance for the vocal track.

| $H$ \ $E$ |    4   |    8   |   16   |   32   |
|-----------|--------|--------|--------|--------|
|     1     |   7.9  |   8.5  |   8.4  |   8.3  |
|     2     |   7.7  |   8.2  |   8.6  |   8.3  |
|     4     |   8.4  |   8.4  |   8.5  |        |

We observe that using 2 attention heads with a dimension of 16 yields a large 0.8 dB improvement over no attention mechanism, at the cost of very few additional parameters. We evaluate this attention mechanism on other tracks as well:

|                 | vocals |  bass  |  drums |  other | average|
|-----------------|--------|--------|--------|--------|--------|
|  no attention   |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|  with attention |   8.6  |   6.7  |  10.7  |   5.3  |   7.8  |

Overall, attention yields an average 0.5 dB improvement. This is mostly due to the vocals and drums tracks, for which this small model with attention outperforms a [large model](#large-model) with no attention. The mechanism also benefits to the other track, to some extent, but not importantly to the bass track.

**Note**: Following this idea, a recent approach called [Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612) replaced the LSTM+Attention mechanism with transformers, and obtained significantly better results. This is in line with a trend that consists in completely replacing recurrent networks with transformers.


### Data preprocessing

In our experiments, we use a similar data preprocessing as suggested in the paper, based on source activity detection (SAD), thus we compare it with no preprocessing.

|                    | vocals |  bass  |  drums |  other | average|
|--------------------|--------|--------|--------|--------|--------|
|  SAD preprocessing |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|  no preprocessing  |   8.1  |   7.1  |   9.7  |   5.2  |   7.5  |

We obtain slightly better results with no preprocessing. This suggests that the SAD preprocessing implementation we used (taken from [another repository](https://github.com/amanteur/BandSplitRNN-Pytorch)) can probably be improved.


## Optimized model

Following the results above, we train a large network (faeture dim 128 and num_repeats 12) that is optimized considering the experiments above, that is, using the non-preprocessed dataset, monitoring with validation SDR, adding some attention heads.

|                   | vocals |  bass  |  drums |  other | average|
|-------------------|--------|--------|--------|--------|--------|
|  as in the paper  |   9.2  |   7.7  |   9.9  |  6.2   |   8.3  |
|  optimized        |   9.6  |   8.4  |  10.8  |  6.6   |   8.9  |

It should be noted that using this large model for the drums track yields roughly the same performance as a [lighter model](#attention-mechanism). Thus, the optimization for the drums model is mostly due to using the attention mechanism, rather than a large model. This should be considered if reducing the computational cost is important.


## A note on the chunking process for evaluation

For evaluation (both validation and test), songs are divided into small segments for performing separation, and then the chunked estimates are assembled to form whole songs/sources estimates. To do that, we use chunks of 10 seconds with a 1s overlap and a [linear fader](https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html#configure-the-application-function).

However, this does not correspond to what is done in BSRNN, which is unfortunately hard to understand, and therefore hard to reproduce. Indeed:
- In the paper, the authors mention some zero-padding at the beginning and end of each chunk before applying the model, and then some overlap-add (OLA) in order to smooth the outputs. However, they don't specify which OLA procedure, i.e., which windowing function is used (does it ensure perfect reconstruction?).
- It was suggestsed by an author of the BSRNN paper to use the [bytedance music separation repo](https://github.com/bytedance/music_source_separation), which includes a [chunk-level separator](https://github.com/bytedance/music_source_separation/blob/master/bytesep/separator.py#L122). However, the output chunks are cropped and concatenated, which does not involve any OLA / fader. Besides, it uses a fixed overlap ratio of 50%, thus it is not applicable to other ratios (e.g., those considered in the paper).
- While test results reported in the paper use 3s chunks with 0.5s hop size, one of the paper's authors mentioned using a 1s hop size for validation. Thus it is unclear whether the same setup was used for validation and testing, and which value is used exactly.

While this matter might seem of limited importance at first glance (it is reasonable to assume that changing the chunking/OLA procedure will yield the same optimal model during training), it could actually be significant in terms of test results. Indeed, adjusting the hop size yield differences up to 0.3 dB (*cf* Table II in the paper), which is larger than the score difference between some competing methods. Thus, this chunking/OLA procedure needs to be clarfied.

We experimented with a simple OLA method which uses rectangular windowing (implemented in this repo), and we compute the test results (for the vocals track) using a 3s window and 0.5s hop size. We don't observe any significant difference with the 10s-fader based chunking method. Nonetheless, considering the observations above, there might be some room for further improvement.
