import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
from os import path, walk
import fnmatch


def find_files(directory, pattern="events*"):
    """Recursively finds all files matching the pattern."""
    filespaths = []
    filenames_list = []
    for root, _, filenames in walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            filespaths.append(path.join(root, filename))
            filenames_list.append(filename)

    return filespaths, filenames_list


def display_from_tb(method_name, tblogdir="tb_logs/", outdir="outputs/", monitor_val='loss'):
    # Path to scan / record results
    tblogdir_model = path.join(tblogdir, method_name)
    path_res_file = path.join(outdir, "val_" + method_name + ".csv")

    # Init result dataframe
    param_keys = [
        "cfg_optim/loss_domain",
        "cfg_optim/lr",
        "feature_dim",
        "num_repeat",
        "time_layer",
        "band_layer",
        "n_att_head",
        "attn_enc_dim",
    ]
    all_results = pd.DataFrame(columns=param_keys + ["val_sdr", "exp_name",  "tb_version"])

    # Iterate over experiments
    all_tb_paths, all_tb_names = find_files(tblogdir_model)

    for i, tbpath in enumerate(all_tb_paths):

        # Initialize the result df
        curr_res = {}
        expname = all_tb_names[i]
        curr_res["exp_name"] = expname.replace("events.out.tfevents.", "")
        curr_res["tb_version"] = int(tbpath.replace(expname,'').replace(tblogdir_model,'').replace('/','').replace('version_',''))

        # Load the TB log
        event_acc = EventAccumulator(tbpath)
        event_acc.Reload()

        # Get the SDR corresponding to the optimal model, depending on the monitoring strategy
        df_valsdr = pd.DataFrame(event_acc.Scalars("val_sdr_epoch"))
        if monitor_val == 'loss':
            df_valloss = pd.DataFrame(event_acc.Scalars("val_loss_epoch"))
            idx_opt = df_valloss.idxmin()['value']
        elif monitor_val == 'sdr':
            idx_opt = df_valsdr.idxmax()['value']
        else:
            raise NameError("Unknown monitoring type")
        curr_res["val_sdr"] = df_valsdr["value"][idx_opt]

        # Load the hyperparameters
        data = event_acc._plugin_to_tag_to_content["hparams"][
            "_hparams_/session_start_info"
        ]
        hparam_data = HParamsPluginData.FromString(data).session_start_info.hparams
        hparam_dict = {
            key: hparam_data[key].ListFields()[0][1] for key in hparam_data.keys()
        }

        # Add the relevent ones in the df
        for k in param_keys:
            curr_res[k] = hparam_dict[k]

        # Add a new entry to the frame containing all results
        all_results.loc[len(all_results)] = curr_res

    # Sort by exp name
    all_results.sort_values(by=["tb_version"], inplace=True)

    # Record and display the results
    all_results.to_csv(path_res_file)
    print(all_results.drop(columns=["exp_name"]))

    return all_results


if __name__ == "__main__":
    method_name = "bsrnn-vocals"
    all_results = display_from_tb(method_name)

# EOF
