# Get and Prepare the Datasets

### MUSDB18-HQ

To train and/or evaluate models, download the [MUSDB18-HQ](https://zenodo.org/records/3338373) dataset. Unzip it in the `data` folder; if you want to use a different folder structure, change the path accordingly [in the config file](https://github.com/magronp/bsrnn/blob/main/conf/config.yaml#L32).

To comply with the original BSRNN pipeline, you need to pre-process the dataset in order to extract non-silent segment indices using a source activity detector (SAD). To that end, run:
```
python prep_dataset_sad.py
```

You can change the SAD parameters via the config file located in the `conf/sad` folder (or in command line, using Hydra).

Note that if you want to skip training and only perform [evaluation](#evaluation), you can download pretrained models on the [Zenodo repository](https://zenodo.org/records/17516442) (and you can skip applying the preprocessing script above).


### MoisesDB

To perform evaluation on the MoisesDB dataset, you'll need to download the dataset (follow instructions [here](https://music.ai/research/)), unzip it in the `data` folder (or change the path accordingly, as above), and install the dedicated [package](https://github.com/moises-ai/moises-db/) as follows:
```
pip install git+https://github.com/moises-ai/moises-db.git
```

Then, run the dedicated script to create 4-stem dataset (same format as MUSDB18):
```
python prep_moisesdb.py
```
You can change the default base/output folders:
```
python prep_moisesdb.py --dir_base=path/to/original/db --dir_4stems=path/where/to/record/4stems/songs
```
