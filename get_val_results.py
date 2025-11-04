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
    
    df.reset_index(inplace=True, drop=True) # avoid index issues
    df.loc[len(df)] = bsrnnseeds

    # Save results
    df.to_csv(join(out_dir, "val_results_sdr.csv"), index=False)

    return


def get_val_energy(targets, out_dir="outputs/", track_epochs=None):

    # Load the validation results (to get epochs per exp/target) and energy
    df = pd.read_csv(join(out_dir, "val_results_all.csv"), index_col=None)
    try:
        em = pd.read_csv(join(out_dir, "emissions.csv"))
    except:
        print("No energy log found")
        return

    # List of experiments for the main Table + a few others (no preliminary exp / layers / BSCNN hyperparameters)
    exp_list = [
        "bsrnn-seed=1",
        "bsrnn-seed=2",
        "bsrnn-seed=3",
        "bsrnn-acc_grad=2",
        "bsrnn-monitor_val=loss",
        "bsrnn-loss_domain=t",
        "bsrnn-loss_domain=tf",
        "bsrnn-n_fft=4096-n_hop=1024",
        "bsrnn-fac_mask=2",
        "bsrnn-fac_mask=1",
        "bsrnn-large",
        "bsrnn-large-patience=30",
        "bsrnn-stereo=naive",
        "bsrnn-stereo=naive-fac_mask=8",
        "bsrnn-stereo=tac",
        "bsrnn-stereo=tac-act_tac=prelu",
        "bscnn-feature_dim=64-num_repeat=8",
        "bsrnn-n_att_head=1-attn_enc_dim=8",
        "bsrnn-n_att_head=2-attn_enc_dim=16",
        "bsrnn-n_heads=2",
        "bsrnn-dset.aug_list=[random_chunk,random_track_mix,rescale_db,silenttarget]",
        "bsrnn-dset=musdb18hq",
        "bsrnn-opt-notac-dset=musdb18hq-patience=30",
        "bsrnn-opt-dset=musdb18hq-patience=30",
    ]

    exp_name_base_model = [
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
        "bsrnn-dset=musdb18hq",
    ]
    exp_name_large = [
        "bsrnn-large-patience=30",
    ]

    all_energy = pd.DataFrame(columns=["exp_name", "target", "energy"])

    el = {}
    for exp_name in exp_list:

        exp_name_emission = exp_name
        if exp_name in exp_name_base_model:
            exp_name_emission = "bsrnn"
        if exp_name in exp_name_large:
            exp_name_emission = "bsrnn-large"

        # Number of epochs for the current exp
        df_exp = df.loc[df["exp_name"] == exp_name]

        ## Build the "emission exp name" including target
        for t in targets:
            # Number of epochs for the current exp and target
            epoch_t = df_exp.loc[df_exp["target"] == t]["total_epochs"].iloc[0]

            # Corresponding energy
            exp_t = exp_name_emission + "-" + t
            enrg_t = em[em["project_name"] == exp_t]["energy_consumed"].iloc[0]

            # if emissions were computed separately with a given number of epochs
            if track_epochs:
                enrg_t = enrg_t / track_epochs * epoch_t

            # Handle the case of the "large" BSRNN model, which was further trained with additional epochs
            if exp_name == "bsrnn-large":
                el[t] = enrg_t

            if exp_name == "bsrnn-large-patience=30":
                enrg_t += el[t]

            # Store the results into the df
            curr_res = {
                "exp_name": exp_name,
                "target": t,
                "energy": enrg_t,
            }

            all_energy.loc[len(all_energy)] = curr_res

    # Pivot to get a summary for exp/target
    all_energy = all_energy.pivot(index="exp_name", columns="target", values="energy")
    all_energy = all_energy.rename_axis(index=None, columns=None).reset_index()
    all_energy = all_energy.rename({"index": "exp_name"}, axis=1)
    all_energy = all_energy[["exp_name", "vocals", "bass", "drums", "other"]]  # reorder

    # Take the mean of 3 runs for the base model
    indx_seeds = all_energy["exp_name"].str.contains("bsrnn-seed")
    bsrnnseeds = all_energy.loc[indx_seeds].mean(numeric_only=True)
    all_energy = all_energy[~indx_seeds]
    bsrnnseeds["exp_name"] = "bsrnn"
    all_energy.loc[all_energy.tail(1).index[0] + 1] = bsrnnseeds

    # Compute the total over targets and exp
    all_energy.loc["Column_Total"] = all_energy.sum(numeric_only=True, axis=0)
    all_energy.loc[:, "Total"] = all_energy.sum(numeric_only=True, axis=1)

    # Save results
    all_energy.to_csv(join(out_dir, "val_results_energy.csv"), index=False)

    return


@hydra.main(version_base=None, config_name="config", config_path="conf")
def get_val_results(args: DictConfig):

    out_dir = args.out_dir
    targets = args.targets
    track_epochs = args.track_epochs

    # Scan the exp_infos file to get corresponding best uSDRs per exp
    #process_all_val_tblogs(out_dir=out_dir)

    # Assemble the val uSDR results to aggregate sources per exp
    get_val_sdr(targets, out_dir=out_dir)

    # Combine it with codecarbon outputs to get energy consumption
    get_val_energy(targets, out_dir=out_dir, track_epochs=track_epochs)

    return


if __name__ == "__main__":
    get_val_results()

# EOF
