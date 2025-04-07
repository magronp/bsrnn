# Analyzing various model configurations

In this document we investigate several model configurations and hyper parameters. Our primary goal is to reproduce the paper's results, but we conduct several additional experiments to study the behavior of the model and training process in more depth. We also seek to optimize the model in order to eventually improve the performance over the original BSRNN.

For speed, we display results on the validation set in terms of uSDR, as it is much faster to compute than the cSDR (see [here](README.md#test-results) for the definitions of uSDR and cSDR). We consider a small model with a hidden dimension of 64 and a number of repeats of 8, except when trying the [larger](#large-model) or [optimized](#optimized-model) models.


## Preliminary tests

This first series of test consists of basic experiments to train the model. Unless specified otherwise, the model is trained by minimizing the same loss as in the paper, using an adjusted learning rate as described [below](#learning-rate), and training is monitored by maximizing the uSDR on the validation set. We set the maximum number of epochs at 200 (which is larger than in the paper, but needed to ensure convergence).

### Randomness, convergence, and patience

The random seed is important, as it might strongly impact the results. We run 3 different training and display the results below.

|         | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
|  run 1  |   7.7  |   5.9  |   9.5  |   4.9  |   7.1  |
|  run 2  |   8.1  |   6.0  |   9.7  |   4.8  |   7.1  |
|  run 3  |   7.4  |   6.4  |   9.8  |   4.6  |   7.1  |
|  mean   |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  std    |   0.3  |   0.2  |   0.1  |   0.1  |   0.1  |

We observe some variability in the results, which could become problematic when comparing model variants (as done below). A good practice would consist in [tuning the random seed as a hyperparamer](https://arxiv.org/pdf/2210.13393) in all experiements, but this would be way too computationally demanding. Alternatively, one can increase the patience parameter to ensure proper convergence and reduce the impact of the random seed. We run 3 other training with a patience parameter of 30 on the vocals track. We observe that the mean SDR is increased from 7.7 to 8.3, but more importantly, the std is reduced to 0.03. This shows that increasing patience is effective to reduce the variance between runs. Unfortuntately, this is computationaly demanding, thus we keep it to 10 as in the paper.

We report hereafter the mean results over these 3 runs above which serves as a reference, and will perform only one run of each experiment to save some computational time. We will outline when a variant performs significantly better or worse based on the overall trend of the validation score (rather than the "best" value).


### Learning rate

In the original paper, the model is trained using a learning rate of $10^{-3}$, a batch size of 2, and 8 GPUs in parallel, yielding a global batch size of 16. Unfortunately, we do not have access to enough (large) GPUs, so theoretically we should increase the batch size in order to yield the same global batch size. However, this is not possible because of memory constraints (larger batches do not fit into such GPUs). As a result, two strategies can be employed to compensate for this drop in global batch size:
- [Accumulating gradients](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#accumulate-gradients), such that each SGD descent step is performed after considering the same amount of data samples.
- Adjusting the learning rate (see [here](https://github.com/magronp/bsrnn/blob/main/train.py#L47)), such that the *effective* learning rate (= learning rate / global batch size) is the same (*cf*. [this paper](https://arxiv.org/pdf/1706.02677))


|                           | vocals |  bass  |  drums |  other | average|
|---------------------------|--------|--------|--------|--------|--------|
|  adjusting learning rate  |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  accumulating gradients   |   8.0  |   5.8  |   9.6  |   4.9  |   7.1  |


The results above show that both strategies yield similar results. We therefore retain adjusting the learning rate, since when training the [large](#large-model) model, we noted a more pronounced gap in favor of this strategy.


### Monitoring criterion

We study here the impact of the monitoring criterion on the validation set. Indeed, it is not clear from the paper which quantity is used: the authors mention that "early stopping is applied when the *best validation* is not found in 10 consecutive epochs". Then we consider either minimizing the validation loss, or maximizing the validation uSDR.

|          | vocals |  bass  |  drums |  other | average|
|----------|--------|--------|--------|--------|--------|
| min loss |   7.5  |   6.4  |   9.3  |   4.8  |   7.0  |
| max SDR  |   7.7  |   6.1  |   9.6  |   4.8  |   7.1  |

While results are similar, overall maximizing the validation SDR allows training to continue for more epochs, which in general is a better strategy for obtaining a larger validation SDR.


### Training loss

The original paper uses a combination (t+tf) of a time-domain (t) and time-frequency domain (tf) terms:
$$
\text{(t)}: \quad |s-\hat{s}|_1
$$
$$
\text{(tf)}: \quad |\Re{(S)}-\Re{(\hat{S})}|_1 + |\Im{(S)}-\Im{(\hat{S})}|_1
$$
We study the influence of the training loss domain by checking these terms individually (and their combination)

| loss    | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
|  t+tf   |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  t      |   7.9  |   6.1  |   9.4  |   4.9  |   7.1  |
|  tf     |   7.9  |   6.4  |   9.6  |   4.9  |   7.2  |

Interestingly, we don't observe any major difference between losses. While previous works have outline the [importance of time-domain training](https://arxiv.org/pdf/1911.08895) (rather than end-to-end time-domain modeling), it seems that if the whole complex-valued STFT is modeled, than TF-domain training works equally well.


## Basic variants

### FFT size

The original BSRNN model uses a fixed FFT size of 2048, while an FFT size of 4096 is common for music separation models (*cf.* UMX, demucs). Thus, we suggest to investigate it since it only moderately increases the model size, while reducing the memory requirements (as the number of time frames is reduced). This allows to increase the batch size and therefore to speed up training.


| n_fft / n_hop   | vocals |  bass  |  drums |  other | average|
|-----------------|--------|--------|--------|--------|--------|
|  2048 / 512     |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  4096 / 1024    |   7.3  |   5.9  |   8.7  |   4.4  |   6.6  |


However, it does not yield good results, which we can explain as follows. While the change in frequency resolution is not really impactful since the STFT is projected into an embedding of fixed dimension, the time resolution is diminished, which has a much stronger impact on the results. This also explains why the drop is more significant for the bass track, which requires a refined time resolution, but not on the drums, which are localized events in time.

Note that an even larger FFT size of 6144 along with a hop size of 1024 (to match the setup of [DTTNet](https://arxiv.org/abs/2309.08684)) yields even worse results. Finally, a smaller FFT size (1024) with hop size of 256 doesn't fit because of memory constraints.


### Masker size

|            | vocals |  bass  |  drums |  other | average|
|------------|--------|--------|--------|--------|--------|
| fac_mask=4 |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
| fac_mask=2 |   7.9  |   6.8  |   9.4  |   4.4  |   7.1  |

We observe that while similar results are obtained on average, it depends on the source: while reducing the masker size might negatively affect performance for drums and other, it might slightly (vocals) or more substantially (bass) improve performance for some tracks. Note however that further decreasing `fac_mask` to 1 decreases performance more significantly.


## Large model

We now use the original paper's architecture by increasing the model size, i.e., a hidden dimension of 128 and a number of repeat of 12 (vs. 64 and 8 for the small model used above).

|                        | vocals |  bass  |  drums |  other | average|
|------------------------|--------|--------|--------|--------|--------|
|   small                |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|   large                |   9.2  |   7.3  |  10.3  |   5.9  |   8.2  |
|   large (patience=30)  |   9.5  |   7.8  |  10.3  |   6.3  |   8.5  |


We observe a large improvement of 1.1 dB on average when using this larger model. This improvement is slightly more important for the vocals track, and less for the drums track.

When checking at the validation SDR over epoch, we observe that the model has not fully converged (as in this [previous experiment](#randomness-convergence-and-patience)). As a result, we increase the patience parameter at 30 in order to allow training to continue and ensure convergence. However, note that the maximum number of epochs is set at 200 to prevent from excessive computation time. The drums and bass models converge before reaching that limit, and the vocals and other models reach a plateau by 200 epochs. While the drums track does not benefit from this larger training time (overfitting is observed sooner), the remaining models exhibit some improvement. We report the test results using this model in the [readme document](README.md) as our implementation of the original BSRNN model.


## Further architecture variants

We now suggest several potential directions for further improving the performance of BSRNN.


### Stereo modeling

Even though BSRNN is applied to stereo music, it is inherently a monochannel modeling technique, since it is apply to each channel individually. We propose instead a first naive extension to [stereo modeling](models/bsrnnstereo.py) by jointly projecting the two channels into a common latent representation, rather than treating each channel independently. 

|              | vocals |  bass  |  drums |  other | average|
|--------------|--------|--------|--------|--------|--------|
|   "mono"     |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|   stereo     |   7.7  |   6.6  |   ???  |   4.0  |   6.7  |

Unfortunately, this approach results in a performance decrease, especially for the drums and other tracks (on the other hand it might be interesting for the bass track), and at the cost of an increase in terms of model size (9.2M vs. 8.0M parameters for the vocals track). One way to bridge this gap is by increasing the masker size via `fac_mask` (to be consistent with the subsequent increase in number of outputs - two channels need to be recovered instead of 1). This bridges the performance gap and yields a 7.9 dB SDR for the vocal tracks, but the model becomes too large (20M parameters) for usage when increasing feature_dim and num_repeat subsequently. Overall, this approach is not effective, although one advantage is the possibility to double the batch size for faster training.


### Sequence and band modeling layers

Here we investigate alternative layers to LSTM for time and band modeling (GRU or Conv1D instead of LSTM), for the vocal track.


| time / band layer |  lstm  |   gru  |  conv  |
|-------------------|--------|--------|--------|
|  lstm             |   7.7  |   7.9  |   7.0  |
|  gru              |   8.0  |   7.8  |   -    |
|  conv             |   6.8  |    -   |   6.4  |

GRUs yield similar results and might be considered as an interesting alternative to LSTMs for slightly lighter networks, though in what follows we still use LSTMs for consistency with the original paper. On the other hand, Conv1D layers induce a large performance drop, although they are also much faster to train because of memory constraints. 


### BSCNN

Here we propose to use  stacked dilated convolutions instead of naive Conv1D layers as done above. Indeed, they increase the receptive field in each block, thus being more suitable replacements for RNNs. We propose to implement the architecture [in this paper](https://arxiv.org/pdf/2306.05887), yielding a fully convolutional model we call band-split CNN ([BSCNN](models/bscnn.py)).

We determine the optimal parameters via preliminary experiments on the vocals track. For speed, we considered time modeling only (no intra-band modeling layer), and a small architecture (`feature_dim`=32 and `num_repeat`=4). We found that the best model uses 4 dilation blocks, a kernel size of 3, and a hidden factor of 2.

We then use these parameters along with a larger architecture (`feature_dim`=64, `num_repeat`=8, and using a band layer as well) in order to compare it to the BSRNN model considered above, but this unfortunately yields a too large model. Therefore, we consider 3 dilation blocks and a factor of 1 instead. Validation results are displayed below:

|                 | vocals |  bass  |  drums |  other | average|
|-----------------|--------|--------|--------|--------|--------|
|  BSRNN (lstm)   |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  BSCNN          |   7.3  |   5.9  |   9.0  |   4.2  |   6.6  |

We observe that an RNN-based network outperforms the CNN-based alternative. We leave exploring more refined convolutional architectures as well as selecting the optimal conv parameters for each instrument specifically to future work.


### Attention mechanism

We propose here to use a multi-head attention mechanism, inspired from the [TFGridNet](https://arxiv.org/abs/2209.03952) model. Indeed, TFGridNet is quite similar to BSRNN, as it projects frequency bands in a deep embedding space, and then applies LSTM over both time and band dimensions. However, it incorporates an additional multi-head attention mechanism, which we propose to incorporate to BSRNN here.

|                       | vocals |  bass  | drums |  other | average|
|-----------------------|--------|--------|-------|--------|--------|
| no attention          |   7.7  |   6.1  |  9.7  |   4.8  |   7.1  |
| 1 head, emb_dim = 8   |   7.7  |   7.4  |  10.4 |   4.8  |   7.6  |
| 2 heads, emb_dim = 16 |   8.2  |   7.7  |  10.4 |   4.9  |   7.8  |


Overall, using attention is beneficial, except for the other track. In particular, this small model with attention outperforms a [large model](#large-model) with no attention for the drums track.

**Note**: Following this idea, a recent approach called [Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612) replaced the LSTM+Attention mechanism with transformers, and obtained significantly better results. This is in line with a trend that consists in completely replacing recurrent networks with transformers.


### Multi-head sequence module

We took inspiration from the [DTTNet](https://arxiv.org/abs/2309.08684) model who proposed a so-called "improved" sequence module, based on splitting the latent representation into several heads for parallel processing. This allows for reducing the number of parameters and performance improvement.

|                | vocals |  bass  |  drums |  other | average|
|----------------|--------|--------|--------|--------|--------|
|    Original    |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|    Multi-head  |   7.6  |   5.5  |   9.1  |   4.0  |   6.6  |

ngroups ="null" (donc automatiquement =3 ou 4), vocals=7.7, donc ne change pas grand chose.
si on réduit num_repeats=4 as suggested dans DTTNet, worse results (7.5 dB vocals, même si 8M -> 5M parameters) 

Thus, this scheme seems to be effective only when using the conv layers as in DTTNet, instead of the band-split mechanism here.


## Dataset and data preprocessing

In our experiments, we use a similar data preprocessing as suggested in the paper, based on source activity detection (SAD), thus we compare it with no preprocessing.

|                    | vocals |  bass  |  drums |  other | average|
|--------------------|--------|--------|--------|--------|--------|
|  SAD preprocessing |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  SAD +  alt   aug  |   7.9  |   6.6  |   9.5  |   4.4  |   7.1  |
|  no SAD            |   8.2  |   6.9  |   9.5  |   5.3  |   7.5  |

We obtain slightly better results with no preprocessing. This suggests that the SAD preprocessing implementation we used (taken from [another repository](https://github.com/amanteur/BandSplitRNN-Pytorch)) can probably be improved.


## Optimized model

Following the results above, we train a large network (faeture dim 128 and num_repeats 12) that is optimized considering the experiments above, that is, using the non-preprocessed dataset, incorporating attention heads, and an increased patience for ensuring convergence.

|                   | vocals |  bass  |  drums |  other | average|
|-------------------|--------|--------|--------|--------|--------|
|  as in the paper  |   9.5  |   7.8  |  10.3  |   6.3  |   8.5  |
|  optimized        |   10.1 |   9.1  |  11.0  |   6.7  |   9.2  |

This so-called optimized version of the model largely improves performance over our implementation of the paper's model. It should be noted that the drums model mostly benefits from using the attention mechanism, rather than increasing the model size via the feature_dim and num_repeats parameters. This should be considered if reducing the computational cost is important.


## A note on the chunking process for evaluation

For evaluation (both validation and test), songs are divided into small segments for performing separation, and then the chunked estimates are assembled to form whole songs/sources estimates. However, the exact procedure employed in the BSRNN paper is hard to understand (and therefore to reproduce). Indeed:
- In the paper, the authors mention some zero-padding at the beginning and end of each chunk before applying the model, and then some overlap-add (OLA) in order to smooth the outputs. However, they don't specify which OLA procedure, i.e., which windowing function is used (does it ensure perfect reconstruction?).
- It was suggested by an author of the BSRNN paper to use the [bytedance music separation repo](https://github.com/bytedance/music_source_separation), which includes a [chunk-level separator](https://github.com/bytedance/music_source_separation/blob/master/bytesep/separator.py#L122). However, the output chunks are cropped and concatenated, which does not involve any OLA / fader. Besides, it uses a fixed overlap ratio of 50%, thus it is not applicable to other ratios (e.g., those considered in the paper).
- While test results reported in the paper use 3s chunks with 0.5 s hop size, one of the paper's authors mentioned using a 1s hop size for validation. Thus it is unclear whether the same setup was used for validation and testing, and which value is used exactly.


To do that, we use chunks of 10 s with a 1 s overlap and a [linear fader](https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html#configure-the-application-function).

While this matter might seem of limited importance at first glance (it is reasonable to assume that changing the chunking/OLA procedure will yield the same optimal model during training), it could actually be significant in terms of test results. Indeed, adjusting the hop size yield differences up to 0.3 dB (*cf* Table II in the paper), which is larger than the score difference between some competing methods. We implement a simple OLA method which uses rectangular windowing, and we compute the test results using a 3 s window and 0.5 s / 1.5 hop size. We don't observe any significant difference with the 10s-fader based chunking method. Nonetheless, considering the observations above, there might be some room for further improvement.

Thus, this chunking/OLA procedure needs to be clarfied.


