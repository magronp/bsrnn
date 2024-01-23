import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from helpers.data import build_training_samplers, build_fulltrack_sampler
from helpers.trainer import create_trainer
from models.instanciate_src import instanciate_src_model_onetarget
from os.path import join
from pathlib import Path


@hydra.main(version_base=None, config_name='config', config_path='conf')
def train(args: DictConfig):

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    target = args.src_mod.target
    
    # Data samplers
    tr_sampler, _ = build_training_samplers(target, args.dset, fast_tr=args.fast_tr)
    val_sampler = build_fulltrack_sampler(target, args.dset, subset="train", split="valid")

    # Method name
    src_mod_name = args.src_mod.name
    method_name = src_mod_name + "-" + target
    model_dir = args.out_dir + src_mod_name + "/"
    ckpt_path_log = join(model_dir, target + ".ckpt")
    if not (args.resume_tr and Path(ckpt_path_log).exists()):
        ckpt_path_log = None

    # Display some info
    print('Method: ', method_name)

    # Instanciate model
    model = instanciate_src_model_onetarget(
        args.optim,
        args.scheduler,
        args.src_mod,
        target=target,
        load_pretrained_sources=args.resume_tr
    )
    print('Number of parameters: ', model.count_params())

    # Trainer
    trainer = create_trainer(
        args.optim,
        method_name,
        ckpt_name=target,
        ckpt_dir=model_dir,
        fast_tr=args.fast_tr,
    )

    # Fit
    trainer.fit(model, tr_sampler, val_sampler, ckpt_path=ckpt_path_log)

    return


if __name__ == "__main__":
    train()

# EOF
