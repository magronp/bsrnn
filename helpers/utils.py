from codecarbon import OfflineEmissionsTracker
import sys
from os.path import join, exists
import pandas as pd
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_exp_params_str():

    # Define the exp name from the input prompt
    exp_params = ""
    list_params = sys.argv[1:]
    for s in list_params:
        if not (
            any(
                substring in s
                for substring in [
                    "src_mod=",
                    "targets=",
                    "tblog_dir",
                    "track_emissions",
                    "ckpt_path=",
                    "hydra.",
                ]
            )
        ):
            exp_params += "-" + s

    # For better readability, remove the "src_mod." and "optim." strings when there are subparams
    exp_params = exp_params.replace("src_mod.", "")
    exp_params = exp_params.replace("optim.", "")

    ## Get a bool to see if it's a new run or a training that is resumed
    # is_resume_tr = True if any("ckpt_path=" in s for s in list_params) else False

    return exp_params


def get_emission_tracker(exp_name, country_iso_code="FRA", pue=1.5, out_dir="outputs/"):

    # Instanciate a tracker object
    tracker = OfflineEmissionsTracker(
        project_name=exp_name,
        country_iso_code=country_iso_code,
        pue=pue,
        output_dir=out_dir,
    )
    return tracker


def append_df_to_main_file(path_main_file, df):

    # Load the file containing all results if it exists
    if exists(path_main_file):
        all_results = pd.read_csv(path_main_file, index_col=False)
    # Otherwise, create it
    else:
        cols = list(df.columns.values)
        all_results = pd.DataFrame(columns=cols)

    # Add a new entry to the frame containing all results
    all_results = pd.concat([all_results, df], ignore_index=True)

    # Record the updated df with all results
    all_results.to_csv(path_main_file, index=False)

    return


def store_exp_info(exp_name, target, vnum, nparams, tblog_dir, out_dir="outputs/"):

    df = pd.DataFrame(
        data={
            "target": target,
            "tb_version": vnum,
            "exp_name": exp_name,
            "num_params": nparams,
            "tblog_dir": tblog_dir,
        },
        index=[0],
    )

    path_main_file = join(out_dir, "exp_infos.csv")
    append_df_to_main_file(path_main_file, df)

    return


def load_val_log(version_list, tblogdir_model, monitor_val_loss=False):

    if isinstance(version_list, int):
        version_list = [version_list]

    df_valsdr, df_valloss = [], []
    for v in version_list:
        # Get the path to the TB log, and load it
        tbpath = glob.glob(join(tblogdir_model, "version_" + str(v), "events*"))[0]
        event_acc = EventAccumulator(tbpath)
        event_acc.Reload()

        # Get the validation SDR and loss over epoch
        df_valsdr.append(pd.DataFrame(event_acc.Scalars("val_sdr_epoch")))
        df_valloss.append(pd.DataFrame(event_acc.Scalars("val_loss_epoch")))

    # Concatenate SDR / loss over many runs (useful when resume training)
    df_valsdr = pd.concat(df_valsdr, ignore_index=True)
    df_valloss = pd.concat(df_valloss, ignore_index=True)

    # Get the index corresponding to the optimal value
    if monitor_val_loss:
        idx_opt = df_valloss.idxmin()["value"]
    else:
        idx_opt = df_valsdr.idxmax()["value"]

    # Get the SDR corresponding to the index
    best_sdr = df_valsdr["value"][idx_opt]

    total_epochs = len(df_valsdr)

    return best_sdr, idx_opt, df_valsdr, df_valloss, total_epochs


# EOF
