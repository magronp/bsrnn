import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from os.path import join
from IPython.display import display
from datetime import datetime
import json

# Global file names
FILE_EMISSIONS = "emissions.csv"
FILE_VAL_RESULTS_EPOCHS = "val_results_epochs.csv"
NAME_VAL_RESULTS_ENERGY = "val_results_energy"
G5K_INFO_FILE = "data/grid5k.json"

# Experiments corresponding to the base model in terms of emissions
EMISSIONS_BASE_MODEL = [
    "bsrnn-seed=1",
    "bsrnn-seed=2",
    "bsrnn-seed=3",
    "bsrnn-patience=30-seed=1",
    "bsrnn-patience=30-seed=2",
    "bsrnn-patience=30-seed=3",
    "bsrnn-acc_grad=2",
    "bsrnn-monitor_val=loss",
    "bsrnn-loss_domain=t",
    "bsrnn-loss_domain=tf",
    "bsrnn-dset.aug_list=[random_chunk,random_track_mix,rescale_db,silenttarget]",
    "bsrnn-sad_dir=null",
    "bsrnn-dset=musdb18hq",
]

# Experiment corresponding to the large model in terms of emissions
EMISSIONS_LARGE_MODEL = [
    "bsrnn-large-patience=30",
    "bsrnn-large-dset=musdb18hq-patience=30"
]

# Load Grid5k info
with open(G5K_INFO_FILE, "r") as fp:
    GRID5K_INFO = json.load(fp)


def get_epoch_per_exp(path_epochs, exp_list=None):

    df = pd.read_csv(path_epochs)

    if exp_list is None:
        exp_list = pd.unique(df["exp_name"]).tolist()

    exp_list_epochs = pd.DataFrame(columns=["exp_name", "target", "total_epochs"])

    epoch_large_t = {}
    for exp_name in exp_list:

        # Current exp sub-frames
        df_exp = df.loc[df["exp_name"] == exp_name]
        curr_targets = pd.unique(df_exp["target"]).tolist()

        # Number of epochs for the current exp and target
        for t in curr_targets:

            epoch_t = df_exp.loc[df_exp["target"] == t]["total_epochs"].iloc[0]

            # Handle the case of the "large" BSRNN model, for which (only) additional epochs are reported if extra patience
            if exp_name == "bsrnn-large":
                epoch_large_t[t] = epoch_t
            if exp_name == "bsrnn-large-patience=30":
                epoch_t += epoch_large_t[t]

            # Store the results into the df
            curr_res = {
                "exp_name": exp_name,
                "target": t,
                "total_epochs": epoch_t,
            }

            exp_list_epochs.loc[len(exp_list_epochs)] = curr_res

    return exp_list_epochs


def energy_green_algorithm(cluster, duration, occupation, pue=1.5):
    mem_factor = 0.3725
    cluster_info = GRID5K_INFO["CLUSTERS_INFO_GPU"][cluster]
    tdp_gpu = cluster_info["tdp_gpu"]
    tdp_cpu = cluster_info["tdp_cpu"]
    nb_gpu = cluster_info["nb_gpu"]
    cores_per_cpu = cluster_info["cores_per_cpu"]
    tdp_core = tdp_cpu / cores_per_cpu
    nb_cores = cluster_info["nb_cpu"] * cores_per_cpu
    memory = cluster_info["mem_gb"]

    duration_h = duration / 3600
    power_gpu = tdp_gpu * nb_gpu * occupation
    power_cpu = tdp_core * nb_cores * occupation
    power_mem = memory * mem_factor

    energy_green_algorithm = (
        (power_gpu + power_cpu + power_mem) * duration_h / 1000
    )  # express energy in kWh
    energy_green_algorithm = energy_green_algorithm * pue
    return energy_green_algorithm


def energy_mlco2(cluster, duration, occupation, pue=1.5):
    cluster_info = GRID5K_INFO["CLUSTERS_INFO_GPU"][cluster]
    tdp_gpu = cluster_info["tdp_gpu"]
    nb_gpu = cluster_info["nb_gpu"]

    duration_h = duration / 3600
    power_gpu = tdp_gpu * nb_gpu * occupation
    energy_mlco2 = power_gpu * duration_h / 1000  # express energy in kWh
    energy_mlco2 = energy_mlco2 * pue

    return energy_mlco2


def energy_codecarbon(emissions, nb_epochs, nb_max_epochs, pue=1.5):
    energy_codecarbon = emissions["energy_consumed"].sum() * nb_max_epochs / nb_epochs
    energy_codecarbon = energy_codecarbon * pue
    return energy_codecarbon


def energy_board_management_controller():
    energy_board_management_controller = 0
    return energy_board_management_controller


def get_energies_single_model(
    emissions, cluster, nb_epochs_codecarbon, nb_epochs_total, occupation
):
    # emissions_csv (str) : csv name to find the emissions
    # cluster (str) name of the cluster that ran the training
    # nb_epochs_codecarbon (int) : nb of epochs of the codecarbon run
    # nb_epochs_total (int) : total nb of epochs that the training need
    # occupation (float) : between 0 and 1 ; 1 is full occupation, 0.5 means that half gpus are used etc...

    duration = emissions["duration"].sum() * nb_epochs_total / nb_epochs_codecarbon
    energy_GA = energy_green_algorithm(cluster, duration, occupation)
    energy_MCO2 = energy_mlco2(cluster, duration, occupation)
    energy_CC = energy_codecarbon(emissions, nb_epochs_codecarbon, nb_epochs_total)
    energy_BMC = energy_board_management_controller()

    energies = {
        "MLCO2": energy_MCO2,
        "GA": energy_GA,
        "BMC": energy_BMC,
        "CC": energy_CC,
    }

    return energies


def average_over_seeds(energies, targets):

    bsrnn_av = []
    for t in targets:

        # Locate rows corresponding to the seed exp, and compute average
        indx_seeds = energies["exp_name"].str.contains("bsrnn-seed") & energies[
            "target"
        ].str.contains(t)
        bsrnnseeds = energies.loc[indx_seeds].mean(numeric_only=True)

        # Add info and store that
        bsrnnseeds["exp_name_full"] = "bsrnn-" + t
        bsrnnseeds["exp_name"] = "bsrnn"
        bsrnnseeds["target"] = t
        bsrnn_av.append(bsrnnseeds)

        # Remove seed-specific results
        energies = energies[~indx_seeds]

    # Assemble bsrnn info in a df, and concatenate it to the main results
    bsrnnseeds = pd.DataFrame.from_dict(bsrnn_av)
    energies = pd.concat((bsrnnseeds, energies), ignore_index=True)

    return energies


def get_energies_from_exp(
    targets, exp_list_epochs, emissions_log, nb_epochs_codecarbon_list, occupation
):

    # Initialize dict to store everything
    energies_project = {}
    outs = []
    emissions_exp_list = emissions_log["project_name"].tolist()

    # Iterate over experiments
    for row in exp_list_epochs.index:

        # Get current exp information, and extract exp name, target, and total number of epochs
        exp = exp_list_epochs.loc[row]

        total_epochs = exp["total_epochs"]
        exp_name = exp["exp_name"]
        target = exp["target"]

        # Exp name corresponding to the emissions file
        exp_name_emissions = exp_name
        if exp_name_emissions in EMISSIONS_BASE_MODEL:
            exp_name_emissions = "bsrnn"
        if exp_name_emissions in EMISSIONS_LARGE_MODEL:
            exp_name_emissions = "bsrnn-large"

        # Full exp name (model-target)
        if "simo" in exp_name:
            exp_name_full = exp_name + "-vocals-bass-drums-other"
            exp_name_emissions_full = exp_name_emissions + "-vocals-bass-drums-other"
        else:
            exp_name_full = exp_name + "-" + target
            exp_name_emissions_full = exp_name_emissions + "-" + target

        # Skip it if it's an exp for which there is no emission log
        if not (exp_name_emissions_full in emissions_exp_list):
            continue

        # Get the emission data corresponding to this exp
        emissions_model = emissions_log.loc[
            emissions_log["project_name"] == exp_name_emissions_full
        ]

        # Additional cluster data for this exp
        cluster = GRID5K_INFO["HARDWARE_TO_NODE"][emissions_model["gpu_model"].item()]

        # Number of epochs for getting the codecarbon estimate
        nb_epochs_codecarbon = nb_epochs_codecarbon_list[row]

        # Compute energies
        out = get_energies_single_model(
            emissions_model, cluster, nb_epochs_codecarbon, total_epochs, occupation
        )

        # Divide the energy by the number of targets for SIMO models (since it's counted multiple times)
        if "simo" in exp_name:
            for m in out.keys():
                out[m] /= len(targets)

        # Add some extra info and store it
        out["exp_name"] = exp_name
        out["target"] = target
        out["exp_name_full"] = exp_name_full

        outs.append(out)

    # Convert list of dicts into a dataframe
    energies_project = pd.DataFrame.from_dict(outs)

    # Average results (if desired) across seeds for the base model
    energies_project = average_over_seeds(energies_project, targets)

    # Re-order a bit
    energies_project = energies_project[
        ["exp_name_full", "exp_name", "target", "CC", "MLCO2", "GA", "BMC"]
    ]  # re-order

    return energies_project


def get_energies_models(
    targets, out_dir="outputs/", pue=1.5, track_epochs=3, exp_list=None
):

    path_epochs = join(out_dir, FILE_VAL_RESULTS_EPOCHS)
    path_emissions = join(out_dir, FILE_EMISSIONS)

    # Get the list of experiments of interest + total number of epochs
    exp_list_epochs = get_epoch_per_exp(path_epochs, exp_list=exp_list)

    # Emissions (codecarbon log)
    emissions_log = pd.read_csv(path_emissions)
    emissions_log["energy_consumed"] = (
        emissions_log["energy_consumed"] / pue
    )  # pue was already accounted for

    # Number of epochs for codecarbon run
    nb_epochs_codecarbon_list = [track_epochs] * len(
        exp_list_epochs
    )  # for simplicity, use the same nb epochs for codecarbon for all exp (as done in practice anyway)

    # Get all experiments energies
    energies = get_energies_from_exp(
        targets, exp_list_epochs, emissions_log, nb_epochs_codecarbon_list, 1
    )

    return energies


def get_energy_summary(energies, metric="CC"):

    # Pivot to get a summary for exp/target
    energies = energies.pivot(index="exp_name", columns="target", values=metric)
    energies = energies.rename_axis(index=None, columns=None).reset_index()
    energies = energies.rename({"index": "exp_name"}, axis=1)
    energies = energies[["exp_name", "vocals", "bass", "drums", "other"]]  # re-order

    # Compute the total over targets and exp
    energies.loc["Column_Total"] = energies.sum(numeric_only=True, axis=0)
    energies.loc[:, "Total"] = energies.sum(numeric_only=True, axis=1)

    return energies


def get_record_energy_summary(
    targets, out_dir="outputs/", pue=1.5, track_epochs=3, exp_list=None, disp=True
):

    energies = get_energies_models(
        targets, out_dir=out_dir, pue=pue, track_epochs=track_epochs, exp_list=exp_list
    )

    for metric in ["CC", "GA", "MLCO2"]:
        energies_summary = get_energy_summary(energies, metric=metric)
        if disp:
            print("----- " + metric + " -------")
            display(energies_summary)
        energies_summary.to_csv(
            join(out_dir, NAME_VAL_RESULTS_ENERGY + "_" + metric + ".csv"), index=False
        )

    return


# EOF
