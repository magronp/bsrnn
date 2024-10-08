import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from helpers.data import build_training_samplers
from helpers.trainer import create_trainer
from models.instanciate_src import instanciate_src_model
from os.path import join


@hydra.main(version_base=None, config_name="config", config_path="conf")
def train(args: DictConfig):

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Get the target to train
    target = args.src_mod.target

    # Data samplers
    tr_sampler, val_sampler = build_training_samplers(
        target, args.dset, fast_tr=args.fast_tr
    )

    # Method name
    src_mod_name = args.src_mod.name

    # Dir to record the model and tblogs
    ckpt_dir = join(args.out_dir, args.src_mod.name_out_dir)
    tblog_dir = join(args.tblog_dir, args.src_mod.name_tblog_dir + "-" + target)

    # Display some info
    print("------------------")
    print(f"Model type: {src_mod_name} -- Target: {target}")
    print(f"--- Dir to record ckpts:  {ckpt_dir}")
    print(f"--- Dir to record tb log: {tblog_dir}")

    # Trainer
    trainer, ngpus = create_trainer(
        args.optim,
        ckpt_name=target,
        ckpt_dir=ckpt_dir,
        log_dir=tblog_dir,
        fast_tr=args.fast_tr,
    )

    # Adjust learning rate using effective batch size (to match the base lr from BSRNN paper)
    args.optim.lr *= (
        args.dset.batch_size * ngpus * args.optim.accumulate_grad_batches / (2 * 8 * 1)
    )

    # Instanciate model
    model = instanciate_src_model(
        args.optim,
        args.scheduler,
        args.eval,
        args.src_mod,
        pretrained_src_path=args.ckpt_path,
    )
    print("Number of parameters: ", model.count_params())

    # Fit
    trainer.fit(model, tr_sampler, val_sampler, ckpt_path=args.ckpt_path)

    return


if __name__ == "__main__":
    train()

# EOF
