import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
from os import path, walk
import fnmatch
import hydra
from omegaconf import DictConfig


def find_files(directory, pattern="events*"):
    """Recursively finds all files matching the pattern."""
    filespaths = []
    filenames_list = []
    for root, _, filenames in walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            filespaths.append(path.join(root, filename))
            filenames_list.append(filename)

    return filespaths, filenames_list


@hydra.main(version_base=None, config_name="config", config_path="conf")
def process_tblog(args: DictConfig):

    # Path to scan / record results
    method_name = args.src_mod.name_tblog_dir + "-" + args.src_mod.target
    tblogdir_model = path.join(args.tblog_dir, method_name)
    path_res_file = path.join(args.out_dir, "val_" + method_name + ".csv")

    # Init result dataframe
    if "bscnn" in method_name:
        param_keys = ["time_layer/n_dil_conv", "time_layer/ks", "time_layer/hs_fac"]
    else:
        param_keys = [
            "cfg_optim/loss_domain",
            "cfg_optim/lr",
            "cfg_optim/accumulate_grad_batches",
            "feature_dim",
            "num_repeat",
            "time_layer",
            "band_layer",
            "n_att_head",
            "attn_enc_dim",
            "n_fft",
            "fac_mask",
            "n_heads",
            "group_num",
        ]

    all_results = pd.DataFrame(
        columns=["tb_version"] + param_keys + ["val_sdr", "exp_name"]
    )

    # Iterate over experiments
    all_tb_paths, all_tb_names = find_files(tblogdir_model)

    for i, tbpath in enumerate(all_tb_paths):

        print(tbpath)
        # Initialize the result df
        curr_res = {}
        expname = all_tb_names[i]
        curr_res["exp_name"] = expname.replace("events.out.tfevents.", "")
        curr_res["tb_version"] = int(
            tbpath.replace(expname, "")
            .replace(tblogdir_model, "")
            .replace("/", "")
            .replace("version_", "")
        )

        # Load the TB log
        event_acc = EventAccumulator(tbpath)
        event_acc.Reload()

        # Load the hyperparameters
        data = event_acc._plugin_to_tag_to_content["hparams"][
            "_hparams_/session_start_info"
        ]
        hparam_data = HParamsPluginData.FromString(data).session_start_info.hparams
        hparam_dict = {
            key: hparam_data[key].ListFields()[0][1] for key in hparam_data.keys()
        }

        # Check whether monitoring is done with SDR or loss
        if "cfg_optim/monitor_val" in hparam_dict.keys():
            monitor_val = hparam_dict["cfg_optim/monitor_val"]
        else:
            # by default, assume the ckpt was monitored with val loss
            monitor_val = "loss"

        # Get the optimal model's SDR
        df_valsdr = pd.DataFrame(event_acc.Scalars("val_sdr_epoch"))
        if monitor_val == "loss":
            df_valloss = pd.DataFrame(event_acc.Scalars("val_loss_epoch"))
            idx_opt = df_valloss.idxmin()["value"]
        elif monitor_val == "sdr":
            idx_opt = df_valsdr.idxmax()["value"]
        else:
            raise NameError("Unknown monitoring type")
        curr_res["val_sdr"] = df_valsdr["value"][idx_opt]

        # Add the relevent ones in the df
        for k in param_keys:
            if k in hparam_dict.keys():
                curr_res[k] = hparam_dict[k]

        # Add a new entry to the frame containing all results
        all_results.loc[len(all_results)] = curr_res

    # Sort by exp name
    all_results.sort_values(by=["tb_version"], inplace=True)

    # Record and display the results
    all_results.to_csv(path_res_file, index=False)
    print(all_results.drop(columns=["exp_name"]).to_string(index=False))

    return


if __name__ == "__main__":
    process_tblog()

# EOF
