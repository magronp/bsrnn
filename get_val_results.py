import pandas as pd
from os.path import join
import hydra
from omegaconf import DictConfig
from helpers.utils import load_val_log


def process_all_val_tblogs(out_dir="outputs/"):

    exp_info = pd.read_csv(join(out_dir, "exp_infos.csv"))
    exp_info = exp_info.reset_index()

    all_results = pd.DataFrame(
        columns=["exp_name", "target", "best_sdr", "total_epochs"]
    )

    list_exps = pd.unique(exp_info["exp_name"])

    for exp_name in list_exps:

        # Get info and all tb versions corresponding to the current exp
        df = exp_info.loc[exp_info["exp_name"] == exp_name]
        version_list = list(df["tb_version"])
        targets = df["target"].iloc[0]
        tblogdir_model = df["tblog_dir"].iloc[0]
        exp_name_str = exp_name.replace("-" + targets, "")
        targets = targets.split("-")

        for t in targets:

            # Get the best SDR for this exp
            best_sdr, _, _, _, total_epochs = load_val_log(
                version_list,
                tblogdir_model,
                monitor_val_loss="monitor_val=loss" in exp_name,
                monitor_target=t if len(targets) > 1 else "epoch",
            )

            # Store the results into a df
            curr_res = {
                "exp_name": exp_name_str,
                "target": t,
                "best_sdr": best_sdr,
                "total_epochs": total_epochs,
            }

            # Append to the main result df
            all_results.loc[len(all_results)] = curr_res

    # Save results (all)
    all_results.to_csv(join(out_dir, "val_results_all.csv"), index=False)

    return


def get_val_epochs(out_dir="outputs/"):

    df = pd.read_csv(join(out_dir, "val_results_all.csv"), index_col=None)
    df = df.drop(columns=["best_sdr"])
    df.to_csv(join(out_dir, "val_results_epochs.csv"), index=False)

    return


def get_val_sdr(targets, out_dir="outputs/"):

    # Load the df containing all results
    df = pd.read_csv(join(out_dir, "val_results_all.csv"), index_col=None)

    # Prepare the SDR results (including average)
    df = df.drop(columns=["total_epochs"])
    df = df.pivot(index="exp_name", columns="target", values="best_sdr")
    df = df.rename_axis(index=None, columns=None).reset_index()
    df = df.rename({"index": "exp_name"}, axis=1)
    df = df[["exp_name", "vocals", "bass", "drums", "other"]]  # reorder

    # Get the average over sources
    for i, row in df.iterrows():
        if not (any(list(pd.isnull(row)))):
            df.at[i, "song"] = row[targets].mean()

    # Take the mean of 3 runs for the base model
    indx_seeds = df["exp_name"].str.contains("bsrnn-seed")
    bsrnnseeds = df.loc[indx_seeds].mean(numeric_only=True)
    df = df[~indx_seeds]
    bsrnnseeds["exp_name"] = "bsrnn"

    df.reset_index(inplace=True, drop=True)  # avoid index issues
    df.loc[len(df)] = bsrnnseeds

    # Save results
    df.to_csv(join(out_dir, "val_results_sdr.csv"), index=False)

    return


@hydra.main(version_base=None, config_name="config", config_path="conf")
def get_val_results(args: DictConfig):

    out_dir = args.out_dir
    targets = args.targets

    # Scan the exp_infos file to get corresponding best uSDRs and total epochs per exp
    process_all_val_tblogs(out_dir=out_dir)

    # Filter out the total number of epochs separately (useful for energy reporting)
    get_val_epochs(out_dir=out_dir)

    # Assemble the val uSDR results to aggregate sources per exp
    get_val_sdr(targets, out_dir=out_dir)

    return


if __name__ == "__main__":
    get_val_results()

# EOF
