import hydra
from omegaconf import DictConfig
from os.path import join
import lightning.pytorch as pl
import tqdm
from models.separator import Separator
from helpers.data import build_fulltrack_sampler
from helpers.parallel import process_all_tracks_parallel
from helpers.eval import (
    aggregate_res_over_tracks,
    append_df_to_main_file,
)

tqdm.monitor_interval = 0


@hydra.main(version_base=None, config_name="config", config_path="conf")
def evaluate(args: DictConfig):

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    sdr_type = args.eval.sdr_type
    targets = args.targets
    src_mod = args.src_mod.name

    model_dir = join(args.out_dir, src_mod)
    args.eval.rec_dir = join(model_dir, "audio")

    # Evaluation results file
    path_eval_file = join(model_dir, "test_results_" + sdr_type + ".csv")
    path_main_file = join(args.out_dir, "test_results_" + sdr_type + ".csv")

    # Evaluation
    if not (args.only_append_res):

        # Special function to use parallel CPU
        if args.parallel_cpu:
            args.eval.device = "cpu"  # make sure the device is CPU
            test_results = process_all_tracks_parallel(
                args, subset="test", split=None, num_cpus=10
            )

        else:
            # Load the separator model
            model = Separator(args)

            # Test dataloader
            test_sampler = build_fulltrack_sampler(targets, args.dset, subset="test")

            # Testing
            trainer = pl.Trainer(num_nodes=1, devices=1, logger=False)
            trainer.test(model, dataloaders=test_sampler)
            test_results = model.test_results

        # Record the results
        test_results.to_csv(path_eval_file, index=False)

    # Aggregate the results over tracks, display it, and append the the main file
    df = aggregate_res_over_tracks(path_eval_file, src_mod, sdr_type)
    print(df)
    append_df_to_main_file(path_main_file, df)


if __name__ == "__main__":
    evaluate()

# EOF
