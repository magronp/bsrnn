# Comparison with other methods

On this page we summarize the performance of most state-of-the-art music separation models. We display results on the [MUSDB18-HQ test set](#musdb18-hq), as well as generalization on the [MoisesDB dataset](#generalization-capability-on-moisesdb). Importantly, we explain how we [obtain and report](#reporting-results) these results.

All results displayed here are expressed in terms of [chunk SDR](README.md#note-on-the-metric) (dB).


## MUSDB18-HQ

Below are the results on the MUSDB18-HQ test set.

|         | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
| [CWS-PResUNet](https://arxiv.org/abs/2112.04685) | 8.92 | 5.93 | 6.38 | 5.84 | 6.77|
| [KUIELab-MDX-Net](https://arxiv.org/abs/2111.12203)   |  8.97 | 7.83 | 7.20 | 5.90 | 7.47|
| [Hybrid Demucs](https://arxiv.org/abs/2111.03600)   | 8.35 | 8.43 | 8.12  | 5.65 | 7.64|
| [HT Demucs](https://arxiv.org/abs/2211.08553)  | 7.93 | 8.48  | 7.94 | 5.72 | 7.52|
| [BSRNN](https://arxiv.org/abs/2209.15174) | 10.01  | 7.22 | 9.01  | 6.70 | 8.24|
| [SIMO-BSRNN](https://ieeexplore.ieee.org/document/10447771/) | 9.73 | 7.80 | 10.06 | 6.56 | 8.54|
| [BS-RoFormer](https://arxiv.org/abs/2309.02612) | 10.66  | 11.31  | 9.49 | 7.73 | 9.80|
| [DTTNet](https://arxiv.org/abs/2309.08684) | 10.12  | 7.45  | 7.74 | 6.92 | 8.06|
| [TFC-TDF UNet v3](https://arxiv.org/abs/2306.09382) |  9.59 | 8.45  | 8.44 | 6.86  | 8.34 |
| [SCNet](https://arxiv.org/pdf/2401.13276) | 9.89	| 8.82	|10.51|	6.76	|9.00   |
| [SCNet-large](https://arxiv.org/pdf/2401.13276) | 10.86	| 9.49	|10.98|	7.44	|9.69   |
| oBSRNN | 9.81	|9.85	|10.31|	6.31	|9.07  |
| oBSRNN-SIMO | 10.66 |	9.73 |	10.98 |	7.78 |	9.79 |


oBSRNN outperforms the original paper's results, therefore it is an interesting alternative to BSRNN, as it is openly available and yields better results. Besides, our oBSRNN-SIMO version largely outperforms SIMO-BSRNN and actually performs on par with BS-RoFormer. Since this model reaches state-of-the-art music separation quality, one might consider using it if the goal is to achieve maximum performance.


### Training using extra data

For reference, here is the performance on the same test set, but where models are trained using extra data (more information about the dedicated dataset can be found in each corresponding paper). Note that  BSRNN only utilizes mixtures, while the others employ multi-track songs. SCNet variants use the openly available MoisesDB dataset as extra material, while the others employ private datasets.

|         | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
| [Hybrid Demucs](https://arxiv.org/abs/2111.03600) | 8.75 | 9.13  | 9.31  | 6.18 | 8.34  |
| [HT Demucs](https://arxiv.org/abs/2211.08553) | 9.37 | 10.47  | 10.83  | 6.41 | 9.27  |
| [BSRNN](https://arxiv.org/abs/2209.15174)  | 10.47  | 8.16 | 10.15  | 7.08 | 8.97  |
| [BS-RoFormer](https://arxiv.org/abs/2309.02612) | 12.72  | 13.32  | 12.91 | 9.01 | 11.99  |
| [SCNet](https://arxiv.org/pdf/2401.13276)   | 10.17  | 9.21 | 10.78  | 6.84 | 9.25 |
| [SCNet-large](https://arxiv.org/pdf/2401.13276)  | 11.10  | 9.86  | 11.23 | 7.51 | 9.92  |

These results provide insight about how much gain can be made by fine-tuning a specific model (e.g., about 2.2 dB for BS-RoFormer), but most of these are by design nonreplicable since most datasets are private. Besides, they allow neither to assess one model's best / maximum potential nor to perform actual comparison between models, since these extra datasets are different from one paper to another.



## Generalization capability on MoisesDB

Below we report results on the MoisesDB dataset, which comprises 235 songs (240 in total, but 5 of them lack one of the 4 stems used here).
Note that here SCNet is not trained on MoisesDB unlike [above](#training-using-extra-data), therefore is only uses the MUSDB18-HQ training set, as for oBSRNN-SIMO. On the other hand, HT Demucs results are those using an extra private dataset.

|         | vocals |  bass  |  drums |  other | average|
|---------|--------|--------|--------|--------|--------|
| [HT Demucs](https://arxiv.org/abs/2211.08553) | 10.05 | 11.64  | 10.94  | 7.00 | 9.91  |
| [SCNet](https://arxiv.org/pdf/2401.13276)   | 10.74  | 11.65 | 11.40  | 7.54 | 10.33 |
| oBSRNN-SIMO | 9.50 |	11.71 |	10.79 |	8.61 |	10.16 |

The results demonstrate that oBSRNN-SIMO performs similarly to SCNet. Both models exhibit superior generalization even with limited data compared to HT Demucs. In particular, while SCNet performs better on the drums and vocals stems, our oBSRNN-SIMO exhibits higher performance on the bass, and more importantly on the other sources. Depending on the application scenario, one might therefore consider it as an interesting alternative.


## Reporting results

Here we describe the method for collecting the test results in the tables above. This allows one to avoid ambiguities or inconsistencies that might occur when cross-comparing papers, especially for model ranking. This also explains why we do not report some popular models (ResUNet, Open-Unmix, Spleeter, D3Net).


### From the papers

The results corresponding to the most models (CWS-PResUNet, TFC-TDF UNet v3, BSRNN, SIMO-BSRNN, BS-RoFormer, DTTNet, SCNet) are reported from the related original publications. The [results on the MoisesDB dataset](#generalization-capability-on-moisesdb) are reported from the  [SCNet](https://arxiv.org/pdf/2401.13276) paper, although the performance for HT Demucs is consistent with the [benchmark](https://github.com/moises-ai/moises-db/blob/main/benchmark/htdemucs4.csv) on the official MoisesDB repository.


### MUSDB18: HQ vs. nonHQ

Let us recall that the MUSDB18 dataset comprises a high-quality version denoted [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html), and a *compressed* version, herein denoted ``[nonHQ](https://zenodo.org/records/1117372)''. The co-existence of these two versions, although beneficial to the community as it provides more flexibility, might complicate reporting.

- In its original paper, the [ResUNet](https://arxiv.org/abs/2109.05418) model is evaluated on MUSDB18-nonHQ. Nevertheless, it is commonly reported in separation benchmarks along with other models tested on MUSDB18-HQ (*cf*. Table 3 in the [Hybrid Demucs paper](https://arxiv.org/abs/2111.03600)), which is somewhat confusing; and sometimes reported as computed on the HQ test set (*cf*. Table 3 in the [DTTNet paper](https://arxiv.org/pdf/2309.08684)), which is erroneous. We do not include this model here since its performance on MUSDB18-HQ is not available.
- Open-Unmix (UMX) official [repository](github.com/sigsep/open-unmix-pytorch) reports results on both datasets. A [follow-up paper](https://arxiv.org/abs/2010.04228) reports performance on MUSDB18-nonHQ, but the numbers correspond to the MUSDB18-HQ performance from the official repository. We do not include this model because of this confusion, as well as its overall low performance compared to more recent methods.
- Similarly, the [original paper](https://arxiv.org/abs/2111.12203) accompanying the KUIELab-MDX-Net model reports performance on MUSDB18-nonHQ, while these same numbers are then reported as performance on MUSDB18-HQ in the [HT Demucs paper](https://arxiv.org/abs/2211.08553). Then, the [BSRNN paper](https://arxiv.org/abs/2209.15174) reports performance on both datasets, thus we extract the HQ results from this source here.
- Lastly, we do not include other popular models such as [Spleeter](https://github.com/deezer/spleeter), or [D3Net](https://arxiv.org/abs/2010.01733), since to the best of our knowledge, their performance is solely reported on MUSDB18-nonHQ.


### Model variants

Lastly, another source of confusion is the existence of several model variants. For instance, the [Hybrid Demucs paper](https://arxiv.org/abs/2111.03600) reports the performance of both a *basic model*, and that of an *optimized bag of models*. Subsequent papers then report either [one](https://arxiv.org/abs/2211.08553) or [the other](https://arxiv.org/abs/2209.15174), without specifying which one explicitly, nor providing justification. 

In Table~\ref{tab:sdr_comparison} we report results that correspond to the optimized bag of models, since it yields the largest SDR. The same applies to HT Demucs, for which we report the best results from the [original paper](https://arxiv.org/abs/2211.08553), except for the average SDR of the model trained with extra data (we replace the erroneous $9.20$ value with the actual average of $9.27$ dB).


