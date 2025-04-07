# Training and evaluation

## Setup

To train and/or evaluate the model as per our paper, you need to download the [MUSDB18-HQ](https://zenodo.org/records/3338373) dataset. Unzip it in the `data` folder; if you want to use a different folder structure, remember to change the path accordingly [in the config file](https://github.com/magronp/bsrnn/blob/main/conf/config.yaml#L32).

To speed up data loading at training, you need to pre-process the dataset in order to extract non-silent segment indices using a source activity detector. To that end, run:
```
python prep_dataset.py
```
Note that if you want to skip training and only perform [evaluation](#evaluation), you can download pretrained models on the [Zenodo repository](https://zenodo.org/records/13903584) (and you can skip applying the preprocessing script above).

## Training

### Basic usage

The core training function can be simply run as follows:
```
python train.py targets=xxx
```
where `xxx` can be `vocals`, `bass`, `drums`, or `other`. You can specify the source model via the `src_mod` parameter as follows:
```
python train.py targets=vocals src_mod=bsrnn-opt
```
By default, `src_mod=bsrnn`, which corresponds to the base model in the paper (i.e., a small-size BSRNN).

To resume training, provide a checkpoint path as follows:
```
python train.py targets=vocals ckpt_path=path/to/checkpoint.ckpt
```

### Fast prototyping

We incorporated a `fast_tr` flag, which is very useful for debugging / fast prototyping / overfitting on purpose.
```
python train.py targets=vocals fast_tr=True
```
It enables the [overfit_batches](https://lightning.ai/docs/pytorch/stable/common/trainer.html#overfit-batches) functionnality of Ligthning to perform training and validation on a single batch. Besides, it disables random operations / augmentations when creating the dataset for ensuring it's the same batch at each step/epoch.


### Training variants

This project uses the hydra framework for structured configuration files, thus changing parameters (e.g., model size, number of layers, learning rate), is quite straightforward:
```
python train.py targets=bass,vocals optim.loss_domain=t+tf src_mod.num_repeat=10
```
Feel free to check the conf files to see all possible parameters and default values.

### Lauching jobs

In practice, you will likely performing training (and testing) using a cluster of GPUs. Here, we use the [Grid5000](https://www.grid5000.fr/w/Grid5000:Home) testbed, which operates under the [OAR](https://oar.imag.fr/) task manager. You should be able to adapt it to the slurm job manager or another testbed with minor adjustments.

For lauching jobs, simply run:
```
jobs/book train <CLUSTER_NAME> <PARAM_ARRAY>
```
where `<CLUSTER_NAME>` is the name of the cluster (this depends on the available hardware), and `<PARAM_ARRAY>` is the txt file that stores the configuration(s) you want to run, whose path is `jobs/params/<PARAM_ARRAY>.txt`. You can adjust the default `<CLUSTER_NAME>` as well as the walltime depending on your hardware in the `jobs/book` file.


## Analyzing results

This project uses tensorboard for logging and monitoring training and validation. In particular, each run of the training script will create a name for the experiment by aggregating all input parameters to the function (see [here](https://github.com/magronp/bsrnn/blob/main/helpers/utils.py#L9)), and store it in an file `<out_dir>/exp_infos.csv`, along with the tensorboard version and folder, and the number of parameters.

Then, to analyze validation results, run:

```
python get_val_results.py
```
This will aggregate results into several csv, including a summary of SDRs over targets and experiments, and a summary of energy consumption (see [below](#tracking-energy)). Note that if you didn't estimate energy consumption beforehand, you should comment the [corresponding line in this script](https://github.com/magronp/bsrnn/blob/main/models/get_val_results.py#L197). These results correspond to Table II in the paper.

You can also run the notebook `vizualization.ipynb` to produce plots (including Figure 1 from the paper)



## Testing

### Basic usage

To perform evaluation on the test set, simply run the `test.py` script, optionally specifying the source model (default: `bsrnn`) and SDR type (default: `usdr`)
```
python test.py src_mod=bsrnn-large eval.sdr_type=csdr
```
Note that when creating a Separator module, the code searches for target-specific checkpoints with the following path: `<out_dir>/<src_mod.name_out_dir>/<target>.ckpt`. If a certain checkpoint is not found, a model will be initialized from scratch with random weights instead. You can change the checkpoint location by overriding th `out_dir` and `src_mod.name_out_dir` parameters.

### Inference procedure

As detailed in the paper (Section 3.5), instead of the default linear fader, it is possible to use an OLA procedure to handle whole songs. To do so, simply change the parameters in the `eval` configuration, e.g.:
```
python test.py eval.segment_len=3 eval.hop_size=1.5
```
By default, `eval.hop_size=null`, which uses the linear fader. Setting a value for `eval.hop_size` will trigger the OLA inference procedure instead.

### Launching jobs

You can perform multiple testing by editing the `jobs/params/test.txt` file (just like for training), and running the following command:
```
jobs/book test <CLUSTER_NAME>
```

### GPU vs. (parallel) CPUs

By default, testing will be performed on GPU if available (you can change it via the `eval.device` parameter). If you'd rather use several CPUs in parallel, you can simply set:
```
python test.py src_mod=bsrnn-large parallel_cpu=True
```
and you can adjust the number of CPUs with the `num_cpus` parameter (if null, then all available CPUs will be used).


## Tracking energy

To track the consummed energy with the [codecarbon](https://codecarbon.io/) toolbox, you need to set the corresponding flag when training:
```
python train.py targets=vocals track_emissions=true
```
which will save the energy (along with the experiment name) in a `<out_dir>/emissions.csv` file.

In the paper, we track emission separately by running additional jobs (listed in the `jobs/params/carbon.txt` file), for a specific number of epochs set at `track_epochs=3`. Then, we estimate the global energy for each considered experiment by accounting for the corresponding total number of epochs (see [here](https://github.com/magronp/bsrnn/blob/main/models/get_val_results.py#L151)).

If you prefer to estimate the consumption directly when training a model (rather than in separate experiments), then feel free to set `track_emissions=true` in the [config file](https://github.com/magronp/bsrnn/blob/main/conf/config.yaml#L45). Then, `<out_dir>/emissions.csv` will directly contain the overall estimated energy, thus you can edit the `get_val_results.py` script by [commenting the energy estimation line](https://github.com/magronp/bsrnn/blob/main/models/get_val_results.py#L197).


## Use your own model

Lastly, even though this project's primary goal is not to be a universal framework for music separation (as [UMX](https://github.com/sigsep/open-unmix-pytorch) or [Asteroid](https://github.com/asteroid-team/asteroid)), it is rather easy to add a custom model:
- add a script, e.g., in the `models/` folder, that defines your model class
- import the model class from your script so you can use it when [instanciating a model](https://github.com/magronp/bsrnn/blob/main/models/instanciate_src.py#L9)
- add a corresponding yaml configuration file in the `conf/src_mod/` folder (e.g., `mycustommodel`)

Then, training and testing your model is a simple as 
```
python train.py targets=vocals src_mod=mycustommodel
```
