from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from os.path import join
from helpers.utils import load_val_log, get_pareto_mask
import pandas as pd
import numpy as np


def get_energy_sdr(exp_list, out_dir="outputs/"):

    metric = "GA"
    df_sdr = pd.read_csv(join(out_dir, "val_results_sdr.csv"), index_col=None)
    df_energy = pd.read_csv(
        join(out_dir, "val_results_energy_" + metric + ".csv"), index_col=None
    )

    sdr, energy = [], []
    for exp_name in exp_list:
        sdr.append(df_sdr.loc[df_sdr["exp_name"] == exp_name]["song"].item())
        e = df_energy.loc[df_energy["exp_name"] == exp_name]["Total"].item()
        energy.append(e)

    x = np.column_stack((energy, sdr))

    return x


def plot_energy_sdr(
    x, annot_list=None, plt_legend=True, figsize=(6.4, 4.8), out_dir="outputs/"
):

    if annot_list is None:
        annot_list = [int(x + 1) for x in range(x.shape[0])]

    mask = get_pareto_mask(x * [1, -1])

    plt.figure(figsize=figsize)
    for i, a in enumerate(x):
        enrgi, sdri = a[0], a[1]
        lin_numb = annot_list[i]
        pos = (enrgi+1, sdri)
        marker = "ob" if mask[i] else "xr"
        plt.plot(enrgi, sdri, marker, markersize=6)
        plt.annotate(lin_numb, pos, fontsize=14)

    plt.xlabel("Energy (kWh)", fontsize=16)
    plt.ylabel("SDR (dB)", fontsize=16)
    plt.xticks(fontsize=14), plt.yticks(fontsize=14)

    if plt_legend:
        blue_dot = mlines.Line2D(
            [],
            [],
            color="blue",
            marker="o",
            linestyle="None",
            markersize=4,
            label="Pareto-optimal",
        )
        plt.legend(handles=[blue_dot], fontsize=14)

    plt.tight_layout()
    plt.savefig(join(out_dir, "pareto.pdf"))
    plt.savefig(join(out_dir, "pareto.png"))

    return


# ICASSP
exp_list = [
    "bsrnn",
    "bsrnn-loss_domain=t",
    "bsrnn-loss_domain=tf",
    "bsrnn-acc_grad=2",
    "bsrnn-monitor_val=loss",
    "bsrnn-n_fft=4096-n_hop=1024",
    "bsrnn-fac_mask=2",
    "bsrnn-fac_mask=1",
    "bsrnn-dset.aug_list=[random_chunk,random_track_mix,rescale_db,silenttarget]",
    "bsrnn-dset=musdb18hq-dset.aug_list=[random_chunk,shuffle_tracks,rescale_db,silentsource]",
    "bsrnn-dset=musdb18hq",
]

x = get_energy_sdr(exp_list)
plot_energy_sdr(x, figsize=(7, 4))
