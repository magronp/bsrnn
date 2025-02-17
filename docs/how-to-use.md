# How to use

## Setup

Clone this repo, create and activate a virtual environment, and install the required packages:

```
pip install -r requirements.txt
```

Then, download the [MUSDB18HQ](https://zenodo.org/records/3338373) dataset and unzip it in the `data/` folder (or change the strucure and path accordingly in the config file). If you want to skip training and simply perform [evaluation](#evaluation) or [inference](#inference--demo) for a quick demo, you can download pretrained models on the [Zenodo repository](https://zenodo.org/records/13903584).

To speed up data loading at training, you need to pre-process the dataset in order to extract non-silent segment indices using a [source activity detector](https://github.com/amanteur/BandSplitRNN-Pytorch). To that end, simply run:
```
python prep_dataset.py
```

## Training

The core training function can be simply run as follows:
```
python train.py
```
This will train the default target (=vocals) using default parameters for both the optimizer and model architecture (check the conf files for more information).

For debugging / fast prototyping / overfitting on purpose, you can use the `fast_tr` flag as follows:
```
python train.py fast_tr=True
```
This enables the [overfit_batches](https://lightning.ai/docs/pytorch/stable/common/trainer.html#overfit-batches) functionnality of Ligthning to perform training and validation on a single batch (this also disables random operations / augmentations when creating the dataset for ensuring it's the same batch at each step/epoch). 


## Trying multiple configurations

Thanks to the Hydra framework, you can easily change parameters (model size, number of layer, learning rate, etc.), via either the configuration files, or directly in command line, for instance:
```
python train.py optim.loss_domain=t+tf src_mod.num_repeat=10
```
Have a look at the config files to check all the parameters you can change! If you want to train all target models using default parameters, simply run:

```
python train.py -m src_mod.target=vocals,bass,drums,other
```

The list of configurations/parameters used for training various [models configurations](#analysis.md) are stored in the `jobs/params_train.txt` file. This file can be used as a parameter array when running multiple jobs using the [OAR](https://oar.imag.fr/docs/latest/user/quickstart.html) task manager (see the `jobs/book_train` script). Depending on your working environment, this script might need some adaptation. Alternatively, you can simply run each job independently as on the example above, using all the configurations in the `jobs/params_train.txt` file.

Lastly, we prepare a script that can easily aggregate all validation results from tensorboard logs into a csv file for comparing variants, and display them:
```
python display_tbresults.py
```

## Evaluation

Once all target models are trained, perform evaluation on the test set by running:
```
python evaluate.py
```
When creating a Separator module, the code search for target-specific checkpoints in the output folder: the default path is `outputs/bsrnn/<target>.ckpt`. If a certain checkpoint is not found, a model will be initialized from scratch with random weights instead. You can change the checkpoint location by overriding th `out_dir` and `src_mod.name_out_dir` parameters (in general, the checkpoints are expected to be in the `<out_dir>/<src_mod.name_out_dir>/` folder).

The function above computes the uSDR by default, but you can easily compute the cSDR:
```
python evaluate.py eval.sdr_type=museval
```
You can also perform multiple evaluation by editing the `jobs/params_test.txt` file (just like for training), and running the `jobs/book_test` script accordingly.

## Inference / demo

For convenience and quick demo-ing, we provide a script `inference.py` to process a mixture. The basic usage is:
```
python inference.py +file_path=path/to/my/file.wav
```
You can specify an offset and a maximum duration (in seconds) with the `offset` and `max_len` parameters (by default, the whole song is processed), and the output directory `rec_dir` (by default, it is the current working directory). By default, it outputs all 4 tracks, but you can only compute some targets of interest, e.g.:
```
python inference.py +file_path=path/to/my/file.wav targets=[vocals,bass]
```
The function loads checkpoints as for the evaluation process described [above](#evaluation), but for convenience at inference, the checkpoints are expected to be found in the `<out_dir>/<model_dir>/` folder, where by default `model_dir=bsrnn-opt`. You can change it accordingly, e.g., :
```
python inference.py +file_path=path/to/my/file.wav model_dir=bsrnn-large
```
Keep in mind that pretrained checkpoints are available from the [Zenodo repository](https://zenodo.org/records/13903584) to experiment right away, using either our implementation of BSRNN, or the optimized model :)
