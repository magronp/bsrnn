import hydra
from omegaconf import DictConfig
import torch
from os.path import join
import lightning.pytorch as pl
from helpers.transforms import myISTFT, mySTFT
from helpers.data import build_fulltrack_sampler
from models.separator import load_separator
from helpers.spec_inv import SpectrogramInversion
from models.instanciate_src import instanciate_src_models_targets
from helpers.eval import (
    append_sdr_to_main_file,
    process_all_tracks,
)

# TODO: coder les Modules pour gérer les oracles sur GPU


def apply_oracle_phase(y, args):
    # y: [1, n_targets, n_channels, n_samples]

    device_ini = y.device
    device_eval = args.eval_device
    targets = args.targets

    # Transforms
    stft = mySTFT(n_fft=args.n_fft, n_hop=args.n_hop)
    istft = myISTFT(n_fft=args.n_fft, n_hop=args.n_hop)

    # Compute STFT of references (to get the phase) and mixture
    ref_stft = stft(y)  # [1, n_targets, n_channels, F, T]
    ref_stft_phase = torch.angle(ref_stft)  # [1, n_targets, n_channels, F, T]
    mix_stft_mag = torch.abs(torch.sum(ref_stft, dim=1))  # [1, n_channels, F, T]

    # Load the models
    spectro_models = instanciate_src_models_targets(
        args.optim,
        args.scheduler,
        args.spectro_model,
        targets=targets,
        load_pretrained_sources=True,
    )
    spectro_models.eval()
    spectro_models.to(device_eval)

    # Spectrogram estimation
    with torch.no_grad():
        mix_stft_mag = mix_stft_mag.to(device_eval)
        V_est = [spectro_models[t](mix_stft_mag)["magnitudes"] for t in targets]
        V_est = torch.cat(V_est, dim=1).to(
            device_ini
        )  # [1, n_targets, n_channels, F, T]

    # Plug the oracle phases and iSTFT
    S_est = torch.mul(V_est, torch.exp(1j * ref_stft_phase))
    y_hat = istft(S_est, length=y.shape[-1])

    return y_hat


def apply_oracle_mag(y, args):
    # y: [1, n_targets, n_channels, n_samples]

    _, n_targets, _, n_samples = y.shape

    # Transforms
    stft = mySTFT(n_fft=args.n_fft, n_hop=args.n_hop)
    istft = myISTFT(n_fft=args.n_fft, n_hop=args.n_hop)

    # Compute STFT of references (to get the magnitude) and mixture
    Y = stft(y)  # [1, n_targets, n_channels, F, T]
    V = torch.abs(Y)  # [1, n_targets, n_channels, F, T]
    X = torch.sum(Y, dim=1)  # [1, n_channels, F, T]

    # Initial phase (mixture's)
    phase_ini = torch.angle(X)
    phase_ini = phase_ini.unsqueeze(1).repeat(1, n_targets, 1, 1, 1)

    # Mask / spectrogram inversion algorithm
    spinvalgo = SpectrogramInversion(
        stft, istft, algo=args.spec_inv.algo, max_iter=args.spec_inv.iter
    )
    Y_hat = spinvalgo(
        V,
        X,
        phase_ini,
        cons_weight=args.spec_inv.consistency_weight,
        audio_len=n_samples,
    )

    # Mask and iSTFT
    y_hat = istft(Y_hat, length=n_samples)

    return y_hat


@hydra.main(version_base=None, config_name="config_sep", config_path="conf")
def oracle(args: DictConfig):
    # Set random seed for reproducibility and pytorch audio backend
    pl.seed_everything(args.seed, workers=True)

    sdr_type = args.sdr_type
    only_append_res = args.only_append_res
    args.separator = 'oracle' + args.oracle_type

    # Dirs and files
    if args.oracle_type == "phase":
        method_name = "oracle-phase-" + args.spectro_model.name
        apply_oracle_fn = apply_oracle_phase

    elif args.oracle_type == "mag":
        method_name = "oracle-mag-" + args.spec_inv.algo
        apply_oracle_fn = apply_oracle_mag

    model_dir = join(args.out_dir, 'oracle', method_name)
    rec_dir = join(model_dir, "audio/")

    # Evaluation results file
    path_eval_file = join(model_dir, "test_results_" + sdr_type + ".csv")
    path_main_file = join(args.out_dir, "oracle_results_" + sdr_type + ".csv")

    # Display the method
    print(' Method:', method_name)

    # Evaluation
    if not (only_append_res):

        # If cuda, use the 'test' method for speed 
        if args.eval_device == "cuda":

            # Load the separator model
            model = load_separator(args)

            # Set the model evaluation-related attributes
            model.rec_dir = rec_dir
            model.verbose_per_track = args.verbose_per_track
            model.eval_segment_len = args.eval_segment_len
            model.eval_overlap = args.eval_overlap
            model.sdr_type = sdr_type
            model.win_dur = args.win_dur
            model.eval_device = "cuda"

            # Test dataloader
            test_sampler = build_fulltrack_sampler(args.targets, args.dset, subset='test')

            # Testing
            trainer = pl.Trainer(num_nodes=1, devices=1, logger=False)
            trainer.test(model, dataloaders=test_sampler)
            test_results = model.test_results

        else:

            test_results = process_all_tracks(
                apply_oracle_fn,
                subset="test",
                split=None,
                parallel_cpu=args.parallel_cpu,
                targets=args.targets,
                rec_dir=rec_dir,
                data_dir=args.data_dir,
                win_dur=args.win_dur,
                sample_rate=args.sample_rate,
                max_len=args.eval_max_len,
                verbose_per_track=args.verbose_per_track,
                eval_segment_len=args.eval_segment_len,
                eval_overlap=args.eval_overlap,
                sdr_type=sdr_type,
                # Now the arguments for the function to test
                args=args,
            )

        # Record the results
        test_results.to_csv(path_eval_file)

    # Append the results to the main results file
    curr_meth_results = append_sdr_to_main_file(
        path_main_file, method_name, path_eval_file, sdr_type=sdr_type
    )
    print(curr_meth_results)

    return


if __name__ == "__main__":
    oracle()

# EOF
