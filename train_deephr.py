import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from helpers.data import build_training_samplers
from helpers.utils import get_deephr_info_td
from helpers.trainer import create_trainer
from models.separator import load_separator
from pathlib import Path


@hydra.main(version_base=None, config_name='config_sep', config_path='conf')
def traindeephr(args: DictConfig):

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # Data samplers
    tr_sampler, val_sampler = build_training_samplers(args.targets, args.dset, fast_tr=args.fast_tr)

    # Get the useful folders / paths
    model_info = get_deephr_info_td(
        args.spec_inv.use_phase_prior,
        args.spec_inv.algo,
        args.spec_inv.iter,
        args.spectro_model.name,
        args.out_dir
    )

    path_separator = model_info["path_separator"]
    path_pretrained = model_info["path_pretrained_separator"]
    method_name = model_info["method_name"]
    model_dir = model_info["model_dir"]

    # If resume training, load ckpt from "path_separator", and use it for TB log
    if args.resume_tr:
        ckpt_path = path_separator
        if Path(ckpt_path).exists():
            args.ckpt_path_log = path_separator 
    else:
        ckpt_path = path_pretrained

    # Display some info
    print('Method: ', method_name, '  ---  Checkpoint to load', ckpt_path)

    # Load the separator model
    model = load_separator(
            args,
            ckpt_path=ckpt_path,
        )
    print('Parameters: ', model.count_params())
    
    # Trainer
    trainer = create_trainer(
        args.optim,
        method_name,
        ckpt_name='separator',
        ckpt_dir=model_dir,
        fast_tr=args.fast_tr,
    )

    # Fit
    trainer.fit(model, tr_sampler, val_sampler, ckpt_path=args.ckpt_path_log)


if __name__ == "__main__":
    traindeephr()

# EOF
