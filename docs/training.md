# Training

## Basic usage

The core training function can be simply run as follows:
```
python train.py targets=xxx
```
where `xxx` can be `vocals`, `bass`, `drums`, or `other`. You can specify the model type via the `model` parameter as follows:
```
python train.py targets=vocals model=bsrnn-opt
```
By default, `model=bsrnn`, which corresponds to a small-size BSRNN.

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

This project uses the Hydra framework for structured configuration files, thus changing parameters (e.g., model size, number of layers, learning rate), is quite straightforward:
```
python train.py targets=bass,vocals optim.loss_domain=t+tf model.num_repeat=10
```
Feel free to check the conf files (e.g., `conf/model/<model>.yaml` for the model types) to see all possible parameters and default values, or to change these directly.


## Launching jobs

In practice, you will likely performing training (and testing) using a cluster of GPUs. Here, we use the [Grid5000](https://www.grid5000.fr/w/Grid5000:Home) testbed, which operates under the [OAR](https://oar.imag.fr/) task manager. Adapting our script to operate with the [SLURM](https://slurm.schedmd.com/overview.html) job manager or another testbed should only require minor adjustments.

For lauching jobs, simply run:
```
jobs/book train <CLUSTER_NAME> <PARAM_ARRAY>
```
where `<CLUSTER_NAME>` is the name of the cluster (this depends on the available hardware), and `<PARAM_ARRAY>` is the txt file that stores the configuration(s) you want to run, whose path is `jobs/params/<PARAM_ARRAY>.txt`. You can adjust the default `<CLUSTER_NAME>` as well as the walltime depending on your hardware in the `jobs/book` file.


## Tracking energy

To track the consummed energy with the [codecarbon](https://codecarbon.io/) toolbox, you need to set the corresponding flag when training:
```
python train.py targets=vocals track_emissions=true
```
which will save the energy (along with the experiment name) in a `<out_dir>/emissions.csv` file.

In the paper, we track emission separately by running additional jobs (listed in the `jobs/params/carbon.txt` file), for a specific number of epochs set at `track_epochs=3`. Then, we estimate the global energy for each considered experiment by accounting for the actual total number of epochs?

If you prefer to estimate the consumption directly when training a model (rather than in separate experiments), then feel free to set `track_emissions=true` in the [config file](https://github.com/magronp/bsrnn/blob/main/conf/config.yaml#L45). Then, `<out_dir>/emissions.csv` will directly contain the overall estimated energy.

## Use your own model

Lastly, even though this project's primary goal is not to be a universal framework for music separation (as [UMX](https://github.com/sigsep/open-unmix-pytorch) or [Asteroid](https://github.com/asteroid-team/asteroid)), it is rather easy to add a custom model:
- add a script, e.g., in the `models/` folder, that defines your model class
- add a corresponding yaml configuration file in the `conf/model/` folder (e.g., `mycustommodel`)
- import the model class from your script so you can use it when [instanciating a model](https://github.com/magronp/bsrnn/blob/main/models/instanciate_src.py#L9)

Then, training (and further testing) your model is a simple as: 
```
python train.py targets=vocals model=mycustommodel
```
