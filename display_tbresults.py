import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import yaml


# Define the TB log dirs to scan
tb_logs_display_dir = "tb_logs/bsrnn-vocals/"
path_res_file = "outputs/val_bsrnn.csv"

# Init result df
cols = [
    "loss_domain",
    "feature_dim",
    "num_repeat",
    "time_layer",
    "band_layer",
    "n_att_head",
    "attn_enc_dim",
    "val_sdr",
]
all_results = pd.DataFrame(columns=cols)

# Iterate over experiments
all_tb_dirs = os.listdir(tb_logs_display_dir)
for tbdir in all_tb_dirs:
    curr_dir = os.path.join(tb_logs_display_dir, tbdir)
    tbfiles = os.listdir(curr_dir)
    tbfiles = [os.path.join(curr_dir, tbfiles[i]) for i in [0, 1]]

    # TODO: un peu dégeu, à faire clean
    for f in tbfiles:
        if 'yaml' in f:
            yamlfile = f
        else:
            tbfile = f

    curr_res = {}

    # Load the TB log
    event_acc = EventAccumulator(tbfile)
    event_acc.Reload()
    # Store the val SDR into a pandas dataframe and extract the max
    df_tmp = pd.DataFrame(event_acc.Scalars("val_sdr_epoch"))
    curr_res["val_sdr"] = df_tmp["value"].max()

    # Load the hyperparameters
    with open(yamlfile, "r") as f:
        prime_service = yaml.safe_load(f)
    # Add the relevent ones in the df
    for k in cols[1:-1]:
        curr_res[k] = prime_service[k]
    curr_res["loss_domain"] = prime_service["cfg_optim"]["loss_domain"]

    # Add a new entry to the frame containing all results
    all_results.loc[len(all_results)] = curr_res

# Record and display the results
all_results.to_csv(path_res_file)
print(all_results)
