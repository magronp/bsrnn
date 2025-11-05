import sys
import numpy as np
from os.path import join, exists
import pandas as pd
import glob
from codecarbon import OfflineEmissionsTracker
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
                    "name_out_dir",
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


def aggregate_res_over_tracks(path_or_df, method_name=None, sdr_type="usdr"):

    # The results are either directly provided as dataframe, or as a path to a file
    if isinstance(path_or_df, str):
        test_results = pd.read_csv(path_or_df)
    else:
        test_results = path_or_df

    # Aggregate scores over tracks (mean for the uSDR, median otherwise)
    if sdr_type == "usdr":
        test_results_agg = test_results.mean(numeric_only=True, axis=0)
    else:
        test_results_agg = test_results.median(numeric_only=True, axis=0)

    # Get the mean over sources
    mean_sdr = test_results_agg.mean()

    # Reform the dataframe
    test_results_agg = pd.DataFrame(test_results_agg).T
    test_results_agg["song"] = mean_sdr

    # Add the method name
    if method_name is not None:
        test_results_agg.insert(loc=0, column="method", value=method_name)

    return test_results_agg


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


def store_exp_info(exp_name, targets, vnum, nparams, tblog_dir, out_dir="outputs/"):

    targets = "-".join(targets)

    df = pd.DataFrame(
        data={
            "target": targets,
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


def load_val_log(
    version_list, tblogdir_model, monitor_val_loss=False, monitor_target="epoch"
):

    # Assume a list of TB logs (in case training was resumed)
    if isinstance(version_list, int):
        version_list = [version_list]

    df_valsdr_monitor, df_valloss, df_valsdr_trg = [], [], []
    for v in version_list:
        # Get the path to the TB log, and load it
        tbpath = glob.glob(join(tblogdir_model, "version_" + str(v), "events*"))[0]
        event_acc = EventAccumulator(tbpath)
        event_acc.Reload()

        # Get the validation SDR and loss over epoch
        df_valsdr_monitor.append(pd.DataFrame(event_acc.Scalars("val_sdr_epoch")))
        df_valloss.append(pd.DataFrame(event_acc.Scalars("val_loss_epoch")))

        if monitor_target != "epoch":
            df_valsdr_trg.append(
                pd.DataFrame(event_acc.Scalars("val_sdr_" + monitor_target))
            )

    if monitor_target == "epoch":
        df_valsdr_trg = df_valsdr_monitor

    # Concatenate SDR / loss over many runs (in case training was resumed)
    df_valsdr_monitor = pd.concat(df_valsdr_monitor, ignore_index=True)
    df_valsdr_trg = pd.concat(df_valsdr_trg, ignore_index=True)
    df_valloss = pd.concat(df_valloss, ignore_index=True)

    # Get the index corresponding to the optimal value
    if monitor_val_loss:
        idx_opt = df_valloss.idxmin()["value"]
    else:
        idx_opt = df_valsdr_monitor.idxmax()["value"]

    # Get the SDR corresponding to the index
    best_sdr = df_valsdr_trg["value"][idx_opt]

    total_epochs = len(df_valsdr_monitor)

    return best_sdr, idx_opt, df_valsdr_trg, df_valloss, total_epochs


def get_pareto_mask(x):
    """
    Find the Pareto-efficient points
    :input: x = [n_points, n_costs] array
    :output: m = [n_points,] boolean array
    adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.arange(x.shape[0])
    n_points = x.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(x):
        nondominated_point_mask = np.any(x < x[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        x = x[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    m = np.zeros(n_points, dtype=bool)
    m[is_efficient] = True
    return m


# EOF
