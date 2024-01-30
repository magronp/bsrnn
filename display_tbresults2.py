import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
from os import path, walk
import fnmatch


def find_files(directory, pattern='events*'):
    """Recursively finds all files matching the pattern."""
    filespaths = []
    filenames_list = []
    for root, _, filenames in walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            filespaths.append(path.join(root, filename))
            filenames_list.append(filename)

    # sort the list, to avoid mismatch in the output files
    filespaths = sorted(filespaths)
    filenames_list = sorted(filenames_list)

    return filespaths, filenames_list


def display_from_tb(method_name, tblogdir='tb_logs/', outdir='outputs/'):

    if 'bsrnn' in method_name:
        param_keys =  ["cfg_optim/loss_domain","feature_dim", "num_repeat","time_layer","band_layer", "n_att_head", "attn_enc_dim"]
    else:
        raise NameError("Unknown model type")

    # Path to scan / record results
    tblogdir_model = path.join(tblogdir, method_name)
    path_res_file = path.join(outdir, 'val_' + method_name + ".csv")

    # Init result dataframe
    all_results = pd.DataFrame(columns=param_keys + ["val_sdr", "exp_name"])

    # Iterate over experiments
    all_tb_paths, all_tb_names = find_files(tblogdir_model)

    for i, tbpath in enumerate(all_tb_paths):

        # Initialize the result df
        curr_res = {}
        curr_res["exp_name"] = all_tb_names[i].replace('events.out.tfevents.','')

        # Load the TB log
        event_acc = EventAccumulator(tbpath)
        event_acc.Reload()

        # Store the val SDR into a pandas dataframe and extract the max
        df_tmp = pd.DataFrame(event_acc.Scalars("val_sdr_epoch"))
        curr_res["val_sdr"] = df_tmp["value"].max()

        # Load the hyperparameters
        data = event_acc._plugin_to_tag_to_content["hparams"]["_hparams_/session_start_info"]
        hparam_data = HParamsPluginData.FromString(data).session_start_info.hparams
        hparam_dict = {key: hparam_data[key].ListFields()[0][1] for key in hparam_data.keys()}

        # Add the relevent ones in the df
        for k in param_keys:
            curr_res[k] = hparam_dict[k]

        # Add a new entry to the frame containing all results
        all_results.loc[len(all_results)] = curr_res

    # Record and display the results
    all_results.to_csv(path_res_file)
    print(all_results.drop(columns=['exp_name']))

    return all_results


if __name__ == "__main__":

    method_name = 'bsrnn-vocals'
    all_results = display_from_tb(method_name)
    
# EOF
