# Analyzing various model configurations

In this document we investigate several model configurations and hyper parameters. Our primary goal is to reproduce the paper's results, but we conduct several additional experiments to study the behavior of the model and training process in more depth. We also seek to optimize the model in order to eventually improve the performance over the original BSRNN.

For speed, we display results on the validation set in terms of *utterance* SDR, as it is much faster to compute than the [chunk SDR](README.md#test-results) using museval. The utterance SDR is equal to a basic signal-to-noise ratio on entier tracks (no chunking into frames, no distortion filter). It is used as evaluation metric in the latest [MDX challenges](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23). We report the mean over songs.

We consider a small model with a hidden dimension of 64 and a number of repeats of 8, except when trying the [larger](#large-model) or [optimized](#optimized-model) models.


## Preliminary tests

This first series of test consists of basic experiments to train the model. Unless specified otherwise, the model is trained by minimizing the same loss as in the paper, using an adjusted learning rate as described [below](#learning-rate), and training is monitored by maximizing the SDR on the validation set. We set the maximum number of epochs at 200 (which is larger than in the paper, but needed to ensure convergence).

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

We study here the impact of the monitoring criterion on the validation set. Indeed, it is not clear from the paper which quantity is used: the authors mention that "early stopping is applied when the *best validation* is not found in 10 consecutive epochs". Then we consider either minimizing the validation loss, or maximizing the validation SDR.

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
| fac_mask=2 |   7.5  |   6.1  |   9.8  |   4.7  |   7.0  |

We observe that average results are similar, but the bahavior might depend on the source: while reducing the masker size might negatively affect performance for drums and other, it improves performance slightly for the vocals or more substantially for the bass. Further decreasing `fac_mask` to 1 slightly decreases the overall performance, although the model becomes much lighter in terms of number of parameters.


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

BSRNN is inherently a monochannel model, since it is applied to each channel of the input music song independently, as these two channels were from 2 different songs. We propose a naive stereo extension to by projecting the two channels into a joint latent representation.

|                              | vocals |  bass  |  drums |  other | average|
|------------------------------|--------|--------|--------|--------|--------|
|   "mono"                     |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|   stereo - naive             |   7.7  |   6.6  |   8.4  |   4.0  |   6.7  |
|   stereo - naive, fac_mask=8 |   7.9  |   6.1  |   8.7  |   4.3  |   6.7  |
|   stereo - TAC, act=TanH     |   7.6  |   6.0  |   9.6  |   4.3  |   6.8  |
|   stereo - TAC, act=PReLU    |   7.9  |   6.5  |   10.0 |   4.7  |   7.3  |

Unfortunately, this approach results in a performance decrease, especially for the drums and other tracks (on the other hand it might be interesting for the bass track), and at the cost of an increase in terms of model size (9.2M vs. 8.0M parameters for the vocals track). One way to bridge this gap is by increasing the masker size via `fac_mask` (to be consistent with the subsequent increase in number of outputs - two channels need to be recovered instead of 1). This bridges the performance gap and yields a 7.9 dB SDR for the vocal tracks, but the model becomes too large (20M parameters) for usage when increasing `feature_dim` and `num_repeat` subsequently. Overall, this approach is not effective, although one advantage is the possibility to double the batch size for faster training.

A more refined approach to stereo modeling consists in leveraging the TAC module that accounts for cross-channel information. If using a TanH activation, as proposed in the [SIMO-BSRNN variant](https://ieeexplore.ieee.org/document/10447771), the perforamnce is not improved. However, if we use the PReLU activation, as proposed in [the original TAC paper](https://arxiv.org/abs/1910.14104), the performance is improved compared to the basic version, especially for the bass and drums tracks.


### Sequence and band modeling layers

Here we investigate alternative layers to LSTM for time and band modeling (GRU or Conv1D instead of LSTM), for the vocal track.


| time / band layer |  lstm  |   gru  |  conv  |
|-------------------|--------|--------|--------|
|  lstm             |   7.7  |   7.9  |   7.0  |
|  gru              |   8.0  |   7.8  |   -    |
|  conv             |   6.8  |    -   |   6.4  |

GRUs yield similar results and might be considered as an interesting alternative to LSTMs for slightly lighter networks, though in what follows we still use LSTMs for consistency with the original paper. On the other hand, Conv1D layers induce a large performance drop, although they are also much faster to train because of memory constraints. 


### BSCNN

We propose to use stacked dilated convolutions instead of naive Conv1D layers as done above. Indeed, they increase the receptive field in each block, thus being more suitable replacements for RNNs. We propose to implement the architecture [in this paper](https://arxiv.org/pdf/2306.05887), yielding a fully convolutional model we call band-split CNN ([BSCNN](models/bscnn.py)).

We determine the optimal parameters via preliminary experiments on the vocals track. For speed, we considered time modeling only (no intra-band modeling layer), and a small architecture (`feature_dim`=32 and `num_repeat`=4). We found that the best model uses 4 dilation blocks, a kernel size of 3, and a hidden factor of 2.

We then use these parameters along with a larger architecture (`feature_dim`=64, `num_repeat`=8, and using a band layer as well) in order to compare it to the BSRNN model considered above, but this unfortunately yields a too large model. Therefore, we consider 3 dilation blocks and a factor of 1 instead. Validation results are displayed below:

|                 | vocals |  bass  |  drums |  other | average|
|-----------------|--------|--------|--------|--------|--------|
|  BSRNN (lstm)   |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  BSCNN          |   7.3  |   5.9  |   9.0  |   4.2  |   6.6  |

We observe that an RNN-based network outperforms the CNN-based alternative. We leave exploring more refined convolutional architectures as well as selecting the optimal conv parameters for each instrument specifically to future work.


### Attention mechanism

Here we consider a multi-head attention mechanism, inspired from the [TFGridNet](https://arxiv.org/abs/2209.03952) model. Indeed, TFGridNet is quite similar to BSRNN, as it projects frequency bands in a deep embedding space, and then applies RNNs over both time and band dimensions, following a dual-path like architecture. It incorporates an additional multi-head attention mechanism, which we propose to incorporate to BSRNN here.

|                       | vocals |  bass  | drums |  other | average|
|-----------------------|--------|--------|-------|--------|--------|
| no attention          |   7.7  |   6.1  |  9.7  |   4.8  |   7.1  |
| 1 head, emb_dim = 8   |   7.7  |   7.4  |  10.4 |   4.8  |   7.6  |
| 2 heads, emb_dim = 16 |   8.2  |   7.7  |  10.4 |   4.9  |   7.8  |


Overall, using attention is beneficial, except for the other track. In particular, this small model with attention outperforms a [large model](#large-model) with no attention for the drums track.

**Note**: Following this idea, a recent approach called [Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612) replaced the LSTM+Attention mechanism with transformers, and obtained significantly better results. This is in line with a trend that consists in completely replacing recurrent networks with transformers.


### Multi-head sequence module

We took inspiration from the [DTTNet](https://arxiv.org/abs/2309.08684) model, where a so-called "improved" sequence module is used. This module is based on splitting the latent representation into several heads for parallel processing: the RNNs then process a smaller representation, which reduces the number of parameters.

|                | vocals |  bass  |  drums |  other | average|
|----------------|--------|--------|--------|--------|--------|
|    Original    |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|    Multi-head  |   7.6  |   5.5  |   9.1  |   4.0  |   6.6  |

However this approach yields a significant performance drop. If we further reduce `num_repeats=4` as suggested in the DTTNet paper, the results get worse, even though the model becomes much lighter (7.5 dB vocals and 5M parameters, vs. 8M for the base one). Thus, this mechanism seems to be effective only when using in cunjonction with other architecture aspects of DTTNet, e.g., not the band-split scheme of BSRNN considered here.


## Dataset and data preprocessing

In our experiments, we use a similar data preprocessing as suggested in the paper, based on source activity detection (SAD), thus we compare it with no preprocessing.

|                    | vocals |  bass  |  drums |  other | average|
|--------------------|--------|--------|--------|--------|--------|
|  SAD preprocessing |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  SAD +  alt   aug  |   7.9  |   6.6  |   9.5  |   4.4  |   7.1  |
|  no SAD            |   8.2  |   6.9  |   9.5  |   5.3  |   7.5  |

We obtain slightly better results with no preprocessing. This suggests that the SAD preprocessing implementation we used (largely based on [another repository](https://github.com/amanteur/BandSplitRNN-Pytorch)) can probably be improved.


## Optimized model

Following the results above, we consider an optimzed model that consists of a large network (`feature_dim=128` and `num_repeats=12`), is trained using the non-preprocessed dataset and a large patience for ensuring convergence, and it incorporates attention heads and a TAC module with PReLU activation.

|                      | vocals |  bass  |  drums |  other | average|
|----------------------|--------|--------|--------|--------|--------|
|  as in the paper     |   9.5  |   7.8  |  10.3  |   6.3  |   8.5  |
|  optimized (no TAC)  |   10.1 |   9.1  |  10.9  |   6.7  |   9.2  |
|  optimized           |   10.2 |  10.2  |  11.3  |   6.9  |   9.6  |


This optimized version of the model largely improves performance over our implementation of the paper's model. We report the results without the TAC module to show that the drums model mostly benefits from using the attention mechanism, rather than increasing the model size (as compared to the results for the [attention mechanism](#attention-mechanism)). This should be considered if reducing the computational cost is important. The further addition of the TAC module adds an extra 0.4 dB on average, mostly due to improvements for the bass and drums tracks, which really benefit from stereo modeling.


## SIMO-BSRNN

As complementary experiments, we implement variants that correspond to [SIMO-BSRNN](https://ieeexplore.ieee.org/document/10447771), a model that builds upon BSRNN with additional variants. In particular, it enables stereo modelind using a TAC module, which we already tested [above](#stereo-modeling). Another feature of SIMO-BSRNN is to use a masker that exploits adjacent frequencies as context. We implemented it but it was shown to yield poor performance in our experiments. Below we test the additional features of SIMO-BSRNN. Note that these (as well as the TAC module) are not evaluated specifically in the paper, thus these experiments might yield new insight about which one(s) contributes the most to the performance improvement.

### Basic and proposed variants

|                                        | vocals |  bass  |  drums |  other | average|
|----------------------------------------|--------|--------|--------|--------|--------|
|  base BSRNN                            |   7.7  |   6.1  |   9.7  |   4.8  |   7.1  |
|  joint refined bandsplit               |   8.6  |   6.7  |   9.5  |   5.0  |   7.4  |
|  shared encoder, subtract last source  |   8.1  |   6.7  |   8.7  |   5.5  |   7.3  |
|  shared encoder, one masker per source |   8.2  |   6.7  |   8.9  |   5.6  |   7.4  |

We first consider a finer-grain and joint band-split scheme, i.e., using the same frequency split for all instruments. This feature mostly improves the performance of the vocals and bass tracks, which is relevant since the refined scheme affects frequency regions where these tracks have some energy content, which is less the case for other / drums.

We then consider using a shared encoder (band-split and sequence/band modeling modules) across sources, hence the name SIMO - *single-input-multiple-outputs*. The paper considers one masker per source for the vocals, bass, and drums track; and the last (=other) source is obtained by subtracting these 3 from the mixture. Note that this approch implies to use the joint bandsplit scheme for all source described above. This yields similar results to a model with a source-specific encoder, but the advantage of this approach is to reduce the number of parameters.

Alternatively, we assume 4 maskers instead of 3 (i.e., one per source and no subtraction). This approach yields a slightly better performance overall, while keeping a reasonable total model size since the encoder is still shared among sources.

### Optimized SIMO-BSRNN

Based on these results, we finally consider a large (`feature_dim=128` and `num_repeats=12`) and optimzed SIMO model (called oBSRNN-SIMO). It combines the features of the [optimized BSRNN model described above](#optimized-model), as well as a joint encoder using a refined band-split scheme shared across sources, and one masker per source.

|                | vocals |  bass  |  drums |  other | average|
|----------------|--------|--------|--------|--------|--------|
|  oBSRNN        |   10.2 |  10.2  |  11.3  |   6.9  |   9.6  |
|  oBSRNN-SIMO   |   11.3 |   9.8  |  11.8  |   8.4  |  10.3  |

This model yields a large performance improvement over the non-SIMO model, and will yield state-of-the-art performance in terms of chunk SDR on the [test set]. An interesting point is that the actual performance improvement of the SIMO model over BSRNN is mostly due to the TAC module for stereo modeling and to the finer band-split scheme - not so much from the actual *SIMO* aspect. Be that as it might, using a shared encoder allows to maintain performance and to reduce model size, which is a significant improvement in itself.
