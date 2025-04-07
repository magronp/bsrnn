import os
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from models.instanciate_src import instanciate_src_model
from helpers.data import build_training_samplers
from helpers.trainer import create_trainer
from helpers.utils import get_exp_params_str, get_emission_tracker, store_exp_info


@hydra.main(version_base=None, config_name="config", config_path="conf")
def train(args: DictConfig):

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Get the target to train
    target = args.targets
    assert target in ["vocals", "bass", "drums", "other"], "Unknown instrument"

    # Define the exp name from the input prompt, and model/target
    exp_params = get_exp_params_str()
    exp_name = args.src_mod.name + exp_params + "-" + target

    # Instanciate and start an object to track emissions
    if args.track_emissions:
        tracker = get_emission_tracker(
            exp_name,
            country_iso_code=args.country_iso_code,
            pue=args.pue,
            out_dir=args.out_dir,
        )
        tracker.start()
        # Set the max number of epochs
        if args.track_epochs:
            args.optim.max_epochs = args.track_epochs

    # Data samplers
    tr_sampler, val_sampler = build_training_samplers(
        target, args.dset, fast_tr=args.fast_tr
    )

    # Dir to record the model and tblogs
    ckpt_dir = os.path.join(args.out_dir, args.src_mod.name_out_dir)
    if args.tblog_dir is None:
        tblog_dir = None
    else:
        tblog_dir = os.path.join(
            args.tblog_dir, args.src_mod.name_tblog_dir + "-" + target + "/"
        )

    # Trainer
    trainer, ngpus, vnum = create_trainer(
        args.optim,
        ckpt_name=target,
        ckpt_dir=ckpt_dir,
        log_dir=tblog_dir,
        fast_tr=args.fast_tr,
    )

    # Adjust learning rate using effective batch size (to match the effective lr from the BSRNN paper)
    args.optim.lr *= args.dset.batch_size * ngpus * args.optim.acc_grad / (2 * 8 * 1)

    # Instanciate model
    ckpt_path = args.ckpt_path
    model = instanciate_src_model(
        args.optim,
        args.scheduler,
        args.eval,
        args.src_mod,
        target=target,
        pretrained_src_path=ckpt_path,
    )
    nparams = model.count_params()

    # Display / store experiment only once (not on multiple GPUs)
    if "LOCAL_RANK" not in os.environ.keys() and "NODE_RANK" not in os.environ.keys():

        print("------------------")
        print(f"Experiment (model-parameters-target): {exp_name}")
        print(f"Number of parameters: {nparams}")
        print(f"--- Dir to record ckpts:  {ckpt_dir}")
        print(f"--- Dir to record tb log: {tblog_dir}")
        print("------------------")

        # Record exp info (name, target, tb version, and num params)
        if vnum:
            store_exp_info(
                exp_name, target, vnum, nparams, tblog_dir, out_dir=args.out_dir
            )

    # Fit
    trainer.fit(model, tr_sampler, val_sampler, ckpt_path)

    # Stop emission tracking and record the results
    if args.track_emissions:
        tracker.stop()

    return


if __name__ == "__main__":
    train()

# EOF
