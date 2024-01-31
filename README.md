# BSRNN

<center><a href="https://arxiv.org/pdf/2209.15174.pdf">
    <img src="https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit/-/raw/master/Figure/BSRNN.png" width="400"></a>
    Image taken from the <a href="https://arxiv.org/pdf/2209.15174.pdf">BSRNN paper.</a>
</center>


<center><a href="https://www.researchgate.net/publication/363402998_TF-GridNet_Making_Time-Frequency_Domain_Models_Great_Again_for_Monaural_Speaker_Separation">
    <img src="https://www.researchgate.net/publication/363402998/figure/fig1/AS:11431281083662730@1662694210541/Proposed-full-band-self-attention-module_W640.jpg" width="300"></a>
    Image taken from the <a href="https://arxiv.org/abs/2209.03952">TFGridNet paper.</a>
</center>



## Test results

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

Here we display the results on the validation set (global SDR) for different variants (all on the vocals track). First, we display below the results in terms of loss domain, and architecture (hidden dimension and number of BS blocks).

| loss    |   feature_dim  |  num_repeat    |  SDR    |
|---------|----------------|----------------|---------|
|  t      |      64        |       8        |         |
|  tf     |      64        |       8        |         |
|  t+tf   |      64        |       8        |         |
|  t+tf   |      64        |       10       |   6.96  |
|  t+tf   |      128       |       8        |   5.63  |



Band split layers (using t+tf, and the 64+8 architecture)


| time layer | band layer |  SDR    |
|------------|------------|---------|
|  lstm      |   lstm     |         |
|  gru       |   lstm     |   7.09  |
|  conv      |   lstm     |   6.41  |
|  lstm      |   gru      |   7.01  |
|  lstm      |   conv     |   6.55  |


Attention mechanism (using t+tf loss, the basic 64+8 architecture, and lstm layers)

| number of heads | attention encoder dim |  SDR    |
|-----------------|-----------------------|---------|
|  0              |           -           |         |
|  1              |           4           |  7.62   |
|  1              |           10          |  7.71   |
|  1              |           20          |  7.53   |
|  2              |           4           |  7.23   |
|  2              |           10          |  7.37   |
|  2              |           20          |  7.56   |

We see that adding one attention head brings some improvement, although it should be noted that this applies to the vocals track, but not necessarily to the other tracks.


commandes utiles :
fuser 6006/tcp -k
fuser -v /dev/nvidia*

