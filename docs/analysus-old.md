# Analyzing various model configurations

In this document we investigate several model configurations and hyper parameters. Our primary goal is to reproduce the paper's results, but we conduct several additional experiments to study the behavior of the model and training process in more depth. We also seek to optimize the model in order to eventually improve the performance over the original BSRNN.

For speed, we display results on the validation set in terms of uSDR, as it is much faster to compute than the cSDR (see [here](README.md#test-results) for the definitions of uSDR and cSDR). We consider a small model with a hidden dimension of 64 and a number of repeats of 8, except when trying the [larger](#large-model) or [optimized](#optimized-model) models.


## Preliminary tests

This first series of test investigate basic parameters to train the model. Unless specified otherwise, the model is trained by minimizing the same loss as in the paper, using an adjusted learning rate as described [below](#learning-rate--batch-size), and training is monitored by maximizing the uSDR on the validation set.


### Learning rate / batch size

In the original paper, the model is trained using a learning rate of $10^{-3}$, a batch size of 2, and 8 GPUs in parallel, yielding a global batch size of 16. Unfortunately, we do not have access to enough (large) GPUs, so theoretically we should increase the batch size in order to yield the same global batch size. However, this is not possible because of memory constraints (larger batches do not fit into such GPUs). As a result, two strategies can be employed to compensate for this drop in global batch size:
- [Accumulating gradients](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#accumulate-gradients), such that each SGD descent step is performed after considering the same amount of data samples.
- Adjusting the learning rate (see [here](https://github.com/magronp/bsrnn/blob/main/train.py#L47)), such that the *effective* learning rate (= learning rate / global batch size) is the same (*cf*. [this paper](https://arxiv.org/pdf/1706.02677))


|                           | vocals |  bass  |  drums |  other | average|
|---------------------------|--------|--------|--------|--------|--------|
|  accumulating gradients   |   8.0  |   5.8  |   9.6  |   4.9  |   7.1  |
|  adjusting learning rate  |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |


The results above show that adjusting the learning rate performs better than accumulating gradients, therefore this is the strategy we retain.

Note that we draw a similar conclusion when training the [large](#large-model) model, with an even greater gap.


### Monitoring criterion


We study here the impact of the monitoring criterion on the validation set. Indeed, it is not clear from the paper which quantity is used: the authors mention that "early stopping is applied when the *best validation* is not found in 10 consecutive epochs". Then we consider either minimizing the validation loss, or maximizing the validation SDR (here: the uSDR, as computing the cSDR at each epoch is very time-consumming).

|          | vocals |  bass  |  drums |  other | average|
|----------|--------|--------|--------|--------|--------|
| min loss |   7.5  |   6.4  |   9.3  |   4.8  |   7.0  |
| max SDR  |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |

Overall, maximizing the validation SDR allows training to continue for more epochs, and yields an average 0.3 dB improvement.


### Training loss

The original paper uses a combination (t+tf) of a time-domain (t) and time-frequency domain (tf) terms:
$$
\text{(t)}: \quad |s-\hat{s}|_1
$$
$$
\text{(tf)}: \quad |\Re{(S)}-\Re{(\hat{S})}|_1 + |\Im{(S)}-\Im{(\hat{S})}|_1
$$
Here we study the influence of the training loss domain by checking these terms individually (and their combination). First, we display below the results when minimizing the validation loss:

| loss    | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
|  t+tf   |   7.5  |   6.4  |   9.3  |  4.8   |   7.0  |
|  t      |   7.6  |   6.3  |   9.4  |  4.7   |   7.0  |
|  tf     |   7.2  |   6.0  |   9.3  |  3.6   |   6.5  |

We observe that using the time-domain only loss yields similar results to the complete t+tf loss. Both approaches perform better than the tf-only loss, which is in line with the [importance of time-domain training](https://arxiv.org/pdf/1911.08895) that has been previously reported.


We also display the results when maximizing the validation SDR, which was shown [above](#monitoring-criterion) to be a more effective monitoring criterion:

| loss    | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
|  t+tf   |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|  t      |   7.9  |   6.1  |   9.4  |   4.9  |   7.1  |
|  tf     |   7.9  |   6.4  |   9.6  |   4.9  |   7.2  |

Interestingly, a different trend is observed, as here the (tf) domain training is competitive with other approaches. This is likely due to the fact that since complex-domain modeling is used, (t) and (tf) are more or less equivalent. Besides, the trend in terms of SDR is similar for both model, so it's probably just the validation loss that is less stable, and prone to stop earlier for (t) or (tf) domain only approaches, while this effect disappears when monitoring with SDR.

In the following experiments, we use the t+tf loss in order to be consistent with the paper, and we monitor validation via maximizing the SDR.


## Basic variants


### FFT size

The original BSRNN model uses a fixed FFT size of 2048, while an FFT size of 4096 is common for music separation models (*cf.* UMX, demucs). Thus, we suggest to investigate it since it only moderately increases the model size, while reducing the memory requirements (as the number of time frames is reduced). This allows to increase the batch size and therefore to speed up training.


| n_fft / n_hop   | vocals |  bass  |  drums |  other | average|
|-----------------|--------|--------|--------|--------|--------|
|  2048 / 512     |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|  4096 / 1024    |   7.3  |   5.9  |   8.7  |   4.4  |   6.6  |


However, it does not yield good results, which we can explain as follows. While the change in frequency resolution is not really impactful since the STFT is projected into an embedding of fixed dimension, the time resolution is diminished, which has a much stronger impact on the results. This also explains why the drop is more significant for the bass track, which requires a refined time resolution, but not on the drums, which are localized events in time.

Note that an even larger FFT size of 6144 along with a hop size of 1024 (to match the setup of [DTTNet](https://arxiv.org/abs/2309.08684)) yields even worse results.


### Masker size

|            | vocals |  bass  |  drums |  other | average|
|------------|--------|--------|--------|--------|--------|
| fac_mask=4 |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
| fac_mask=2 |   7.9  |   6.8  |   9.4  |   4.4  |   7.1  |

Note that further decreasing `fac_mask` to 1 further decreases performance.


### Sequence modeling (time/band layers)

Here we investigate alternative layers to LSTM for time and band modeling (GRU or Conv1D instead of LSTM), for the vocal track.

| timelayer | band layer |  uSDR  |
|-------------------------|---------------------|--------|
|  lstm                   |   lstm              |   7.8  |
|  gru                    |   lstm              |   8.0  |
|  conv                   |   lstm              |   6.8  |
|  lstm                   |   gru               |   7.9  |
|  lstm                   |   conv              |   7.0  |

|  gru                    |   gru               |   7.8  |
|  conv                   |   conv              |   6.4  |


We observe that GRUs yield a slight performance improvement, and might be considered as an interesting alternative to LSTMs; though in what follows we still use LSTMs for consistency with the original paper. On the other hand, Conv1D layers induce a large performance drop, although they are also much faster to train because of memory constraints. We propose a [more refined](#bscnn) alternative to basic 1D-conv.


## Large model

We now use the original paper's architecture by increasing the model size, i.e., a hidden dimension of 128 and a number of repeat of 12 (vs. 64 and 8 for the small model used above).

|           | vocals |  bass  |  drums |  other | average|
|-----------|--------|--------|--------|--------|--------|
|   small   |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|   large   |   9.2  |   7.7  |   9.9  |   6.2  |   8.3  |

We observe a significant improvement when using this larger model for most sources except for the drums track. We report the test results using this model in the main [readme document](README.md) as our implementation of the original BSRNN model.


## Further architecture variants

We now suggest several potential directions for further improving the performance of BSRNN.

### BSCNN

The usage of Conv1D layers as done [before](#sequence-modeling-timeband-layers) is quite naïve. Instead we can replace RNNs with stacked dilated convolutions, as it increases the receptive field in each block, thus being more suitable replacements for recurrent networks. We propose to implement the architecture [in this paper](https://arxiv.org/pdf/2306.05887), yielding a fully convolutional model we call band-split CNN ([BSCNN](models/bscnn.py)).

We determine the optimal parameters via preliminary experiments on the vocals track. For speed, we considered time modeling only (no intra-band modeling layer), and a small architecture (`feature_dim`=32 and `num_repeat`=4). We found that the best model uses 4 dilation blocks, a kernel size of 3, and a hidden factor of 2.

We then use these parameters along with a larger architecture (`feature_dim`=64, `num_repeat`=8, and using a band layer as well) in order to compare it to the BSRNN model considered above, but this unfortunately yields a too large model. Therefore, we consider 3 dilation blocks and a factor of 1 instead. Validation results are displayed below:

|                 | vocals |  bass  |  drums |  other | average|
|-----------------|--------|--------|--------|--------|--------|
|  BSRNN (lstm)   |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|  BSCNN          |   7.3  |   5.9  |   9.0  |   4.2  |   6.6  |

We observe that an RNN-based network significantly outperforms the CNN-based alternative. We leave exploring more refined convolutional architectures as well as selecting the optimal conv parameters for each instrument specifically to future work.


### Attention mechanism

We propose here to use a multi-head attention mechanism, inspired from the [TFGridNet](https://arxiv.org/abs/2209.03952) model. Indeed, TFGridNet is quite similar to BSRNN, as it projects frequency bands in a deep embedding space, and then applies LSTM over both time and band dimensions. However, it incorporates an additional multi-head attention mechanism, which we propose to incorporate to BSRNN here. We investigate the impact of the number of attention heads ($H$), as well as the dimension of the attention encoder ($E$) onto performance for the vocal track.

| $H$ \ $E$ |    4   |    8   |   16   |   32   |
|-----------|--------|--------|--------|--------|
|     1     |   7.9  |   8.5  |   8.4  |   8.3  |
|     2     |   7.7  |   8.2  |   8.6?  |   8.3 ? |
|     4     |   8.4  |   8.4  |   8.5  |    -   |
résultats pas méga fiables car poursuite d'un entrainement avec monitor loss, et sur les "?" il semble y avoir des problèmes...


We observe that using 2 attention heads with a dimension of 16 yields a large 0.8 dB improvement over no attention mechanism, at the cost of very few additional parameters. We evaluate this attention mechanism on other tracks as well:

|                 | vocals |  bass  |  drums |  other | average|
|-----------------|--------|--------|--------|--------|--------|
|  no attention   |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|  with attention |   8.6  |   6.8  |  10.7  |   5.3  |   7.8  |
pas ultra clair ces résultats, car tous semblent avoir continué training à partir d'une version pré convergée...


| from scratch 2/16 |   8.2  |   7.7  |  10.4 |   4.9  |        |
| from scratch 1/8  |   7.7  |   7.4  |  10.4 |   4.8  |        |


donc ya un souci... peut-être dû au fait que précédemment on a fait la val sur min loss puis ensuite poursuivi le training..;



Overall, attention yields an average 0.5 dB improvement. This is mostly due to the vocals and drums tracks, for which this small model with attention outperforms a [large model](#large-model) with no attention. The mechanism also benefits to the other track, to some extent, but not importantly to the bass track.

**Note**: Following this idea, a recent approach called [Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612) replaced the LSTM+Attention mechanism with transformers, and obtained significantly better results. This is in line with a trend that consists in completely replacing recurrent networks with transformers.


### Stereo modeling

Even though BSRNN is applied to stereo music, it is inherently a monochannel modeling technique, since it is apply to each channel individually. We propose instead a first naive extension to [stereo modeling](models/bsrnnstereo.py) by jointly projecting the two channels into a common latent representation, rather than treating each channel independently. 

|              | vocals |  bass  |  drums |  other | average|
|--------------|--------|--------|--------|--------|--------|
|   "mono"     |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|   stereo     |   7.7  |   6.6  |   8.4  |   4.0  |   6.7  |

Unfortunately, this approach results in a performence decrease, especially for the drums and other tracks, and at the cost of an increase in terms of model size (9.2M vs. 8.0M parameters for the vocals track). One way to bridge this gap is by increasing the masker size via `fac_mask` (to be consistent with the subsequent increase in number of outputs - two channels need to be recovered instead of 1). This bridges the performance gap and yields a 7.9 dB SDR for the vocal tracks, but the model becomes too large (20M parameters) for usage when increasing feature_dim and num_repeat subsequently. Overall, this approach is not effective, although one advantage is the possibility to double the batch size for faster training.


### Multi-head sequence module

We took inspiration from the [DTTNet](https://arxiv.org/abs/2309.08684) model who proposed a so-called "improved" sequence module, based on splitting the latent representation into several heads for parallel processing. This allows for reducing the number of parameters and performance improvement.

|                | vocals |  bass  |  drums |  other | average|
|----------------|--------|--------|--------|--------|--------|
|    Original    |   8.1  |   7.1  |   9.7  |   5.2  |   7.5  |
|    Multi-head  |   7.7  |   6.2  |   9.2  |   4.7  |   7.0  |


Note that changing the number of groups yields even worse results.



dataset SAD :
|                | vocals |  bass  |  drums |  other | average|
|----------------|--------|--------|--------|--------|--------|
|    Original    |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|    Multi-head  |   7.6  |   5.5  |   9.1  |   4.0  |   6.6  |

ngroups ="null" (donc automatiquement =3ou 4), vocals=7.7, donc ne change pas grand chose.
si on réduit num_repeats=4 as suggested dans DTTNet, worse results (7.5 dB vocals, même si 8M -> 5M parameters) 

Thus, this scheme seems to be effective only when using the conv layers as in DTTNet, instead of the band-split mechanism here.


## Dataset and data preprocessing

In our experiments, we use a similar data preprocessing as suggested in the paper, based on source activity detection (SAD), thus we compare it with no preprocessing.

|                    | vocals |  bass  |  drums |  other | average|
|--------------------|--------|--------|--------|--------|--------|
|  SAD preprocessing |   7.8  |   6.6  |   9.7  |   4.9  |   7.3  |
|  no preprocessing  |   8.1  |   7.1  |   9.7  |   5.2  |   7.5  |
|  slightly less data|   8.0  |   7.0  |   9.7  |   5.1  |   7.4  |


We obtain slightly better results with no preprocessing. This suggests that the SAD preprocessing implementation we used (taken from [another repository](https://github.com/amanteur/BandSplitRNN-Pytorch)) can probably be improved.

#TODO: il faut régler ce truc de la data, peut-être que SAD en fait c'est juste qu'il utilise les mêmes random track mix contrairement à umx qui emploie de nouvelles combinaisons de tracks à chaque epoch...


|  SAD + nouvelles aug | 7.7 / 7.9   |   6.6  |  9.5   |  4.4   |     |


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


#TODO: penser à faire une note sur les random seed ?