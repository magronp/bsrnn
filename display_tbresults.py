import os
import yaml
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def display_from_tb(model_to_validate, tblogdir='tb_logs/', outdir='outputs/'):

    if 'bsrnn' in model_to_validate:
        optim_sched_params = ["loss_domain"]
        model_params =  ["feature_dim", "num_repeat","time_layer","band_layer", "n_att_head", "attn_enc_dim"]
    else:
        raise NameError("Unknown model type")

    # Path to scan / record results
    tblogdir_model = os.path.join(tblogdir, model_to_validate)
    path_res_file = os.path.join(outdir, 'val_' + model_to_validate + ".csv")

    # Init result dataframe
    all_results = pd.DataFrame(columns=optim_sched_params + model_params + ["val_sdr"])

    # Iterate over experiments
    all_tb_dirs = os.listdir(tblogdir_model)
    for tbdir in all_tb_dirs:

        # Get the TB and hparams (yaml) files
        curr_dir = os.path.join(tblogdir_model, tbdir)
        tbfiles = os.listdir(curr_dir)
        tbfiles = [os.path.join(curr_dir, tbfiles[i]) for i in [0, 1]]
        for f in tbfiles:
            if 'yaml' in f:
                yamlfile = f
            else:
                tbfile = f

        # Initialize the result df
        curr_res = {}

        # Load the TB log
        event_acc = EventAccumulator(tbfile)
        event_acc.Reload()
        # Store the val SDR into a pandas dataframe and extract the max
        df_tmp = pd.DataFrame(event_acc.Scalars("val_sdr_epoch"))
        curr_res["val_sdr"] = df_tmp["value"].max()

        # Load the hyperparameters
        with open(yamlfile, "r") as f:
            hparams = yaml.safe_load(f)
        # Add the relevent ones in the df
        for k in model_params:
            curr_res[k] = hparams[k]
        curr_res["loss_domain"] = hparams["cfg_optim"]["loss_domain"]
        curr_res["scheduler"] = hparams["cfg_scheduler"]["name"]

        # Add a new entry to the frame containing all results
        all_results.loc[len(all_results)] = curr_res

    # Record the results
    all_results.to_csv(path_res_file)

    return all_results


model_to_validate = 'bsrnn-vocals'
all_results = display_from_tb(model_to_validate)
print(all_results)
