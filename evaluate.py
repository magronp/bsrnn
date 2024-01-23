import hydra
from omegaconf import DictConfig
import torch
from os.path import join
import lightning.pytorch as pl
import tqdm
from models.separator import load_separator
from helpers.data import build_fulltrack_sampler
from helpers.utils import get_model_info
from helpers.eval import (
    append_sdr_to_main_file,
    process_all_tracks,
)

tqdm.monitor_interval = 0


def load_model_and_inference(y, args, path_separator, device='cpu'):
    # Get the mixture from the references  [1, n_channels, n_samples]
    mix = torch.sum(y, dim=1)

    # Load the separator
    model = load_separator(
        args,
        ckpt_path=path_separator
    )
    
    model.eval()
    model.to(device)

    # Apply the model (inference)
    with torch.no_grad():
        mix = mix.to(device)
        outputs = model(mix)
        y_hat = outputs["waveforms"].detach()

    return y_hat


@hydra.main(version_base=None, config_name='config_sep', config_path='conf')
def evaluate(args: DictConfig):

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    sdr_type = args.sdr_type
    targets = args.targets

    # Get the model info (useful folders / paths)
    model_info = get_model_info(
        args,
        out_dir=args.out_dir
    )

    ## TODO à simplifier car pour l'eval, juste à aller chercher les parametres liés au spinv optimal (dans un yaml)

    # Get the useful information for evaluation
    model_dir = model_info["model_dir"]
    rec_dir = model_info["rec_dir"]
    path_separator = model_info["path_separator"]
    method_name = model_info["method_name"]

    # Evaluation results file
    path_eval_file = join(model_dir, "test_results_" + sdr_type + ".csv")
    path_main_file = join(args.out_dir, "separator_results_" + sdr_type + ".csv")

    # Display the method
    print(' Method:', method_name)

    # Evaluation
    if not (args.only_append_res):

        # If cuda, use the test method for speed 
        if args.eval_device == "cuda":

            # Load the separator model
            model = load_separator(
                args,
                ckpt_path=path_separator
            )

            # Set the model evaluation-related attributes
            model.rec_dir = rec_dir
            model.verbose_per_track = args.verbose_per_track
            model.eval_segment_len = args.eval_segment_len
            model.eval_overlap = args.eval_overlap
            model.sdr_type = sdr_type
            model.win_dur = args.win_dur
            model.eval_device = "cuda"

            # Test dataloader
            test_sampler = build_fulltrack_sampler(targets, args.dset, subset='test')

            # Testing
            trainer = pl.Trainer(num_nodes=1, devices=1, logger=False)
            trainer.test(model, dataloaders=test_sampler)
            test_results = model.test_results

        else:
            # Process test tracks
            test_results = process_all_tracks(
                load_model_and_inference,
                subset="test",
                split=None,
                parallel_cpu=args.parallel_cpu,
                targets=targets,
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
                args_sep=args,
                path_separator=path_separator,
            )

        # Record the results
        test_results.to_csv(path_eval_file)

    # Add to the overall results file
    curr_meth_results = append_sdr_to_main_file(
        path_main_file, method_name, path_eval_file, sdr_type=sdr_type
    )
    print(curr_meth_results)


if __name__ == "__main__":
    evaluate()

# EOF
