# Evaluation

## Analyzing results

This project uses tensorboard for logging and monitoring training and validation. In particular, each run of the training script will create a name for the experiment by aggregating all input parameters to the function (see [here](https://github.com/magronp/bsrnn/blob/main/helpers/utils.py#L9)), and store it in an file `<out_dir>/exp_infos.csv`, along with the tensorboard version and folder, and the number of parameters.

Then, to analyze validation results, run:

```
python process_val_results.py
```
This will aggregate results into several CSV files, including summaries of SDRs, epochs, and energy over targets and experiments.

You can also run the notebook `vizualization.ipynb` to produce plots as in the papers. Note that several figures can only be plot if you have [tracked energy](../docs/training.md#tracking-energy) when training your models.


## Testing

### Basic usage

To perform evaluation on the test set, simply run the `test.py` script, optionally specifying the source model (default: `bsrnn`) and SDR type (default: `usdr`, see the [note below](#note-on-the-metrics) for more information about the metrics):
```
python test.py model=bsrnn-large eval.sdr_type=csdr
```
The code will create a `Separator` module, for which it will search for target-specific checkpoints with the following path: `<out_dir>/<model.name_out_dir>/<target>.ckpt`. If a certain checkpoint is not found, a model will be initialized from scratch with random weights instead. You can change the checkpoint location by overriding the `out_dir` and `model.name_out_dir` parameters.

**Note**: if you want to use a SIMO model, you need to add an extra flag `simo=true`, so that the code loads a multi-source checkpoint named `separator.ckpt` instead of multiple single-source checkpoints named `<target>.ckpt`, e.g.:
```
python test.py model=simo-bsrnn-opt simo=true
```


### Inference procedure

As detailed in the paper (Section 3.5), instead of the default linear fader, it is possible to use an OLA procedure to handle whole songs. To do so, simply change the parameters in the `eval` configuration, e.g.:
```
python test.py eval.segment_len=3 eval.hop_size=1.5
```
By default, `eval.hop_size=null`, which uses the linear fader. Setting a value for `eval.hop_size` will trigger the OLA inference procedure instead.


## Test on MoisesDB

To perform evaluation on the MoisesDB dataset, just change the path to the dataset, by indicating where the 4-stem preprocessed MoisesDB is located: the `Dataloader` object will adapt to it.
```
python test.py data_dir=data/moisesdb_4stems
```
In particular, since there are 235 songs in this dataset, we strongly advise to use the additional flag `sdr_type=csdr-fast`, since computing the cSDR this way is much faster than using the [museval](https://github.com/sigsep/sigsep-mus-eval) tooblox (see [note on the metric](README.md#note-on-the-metric)).


## Launching jobs

You can perform multiple testing by editing the `jobs/params/test.txt` file (just like for [training](../docs/training.md#launching-jobs)), and running the following command:
```
jobs/book test <CLUSTER_NAME>
```

## GPU vs. (parallel) CPUs

By default, testing will be performed on GPU if available (you can change it via the `eval.device` parameter). If you'd rather use several CPUs in parallel, you can simply set:
```
python test.py model=bsrnn-large parallel_cpu=True
```
and you can adjust the number of CPUs with the `num_cpus` parameter (if null, then all available CPUs will be used).



## Note on the metrics

We consider two variants of the signal-to-distortion ratio (SDR), which is common is modern source separation papers:
- the *utterance* SDR (uSDR), computed by taking the mean SDR across whole songs.
- the *chunk* SDR (cSDR), computed via the [museval](https://github.com/sigsep/sigsep-mus-eval) toolbox by taking the median SDR across 1 s-long chunks and across songs.

Note that most of the museval function's computational cost comes from calculating a distortion filter which does not actually affect the SDR. Indeed, when using default parameters as per SiSEC guidelines, the distortion filter only affects the signal-to-interference and -artifact ratios (SIR and SAR), which are not considered here nor in most recent MSS papers (see [this thread](https://github.com/sigsep/sigsep-mus-eval/issues/101) on the museval project).

Therefore, we propose an [efficient implementation](https://github.com/magronp/bsrnn/blob/main/helpers/eval.py#L34) if only the cSDR is needed (i.e., no SIR/SAR). However, there are some discrepancies between this implementation and the museval results, which stem from several frames being set at "NaN" when a source is silent - this is still under investigation.