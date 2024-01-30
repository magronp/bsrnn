import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from helpers.data import build_training_sampler, build_fulltrack_sampler
from helpers.trainer import create_trainer
from helpers.instanciate_src import instanciate_src_model_onetarget


@hydra.main(version_base=None, config_name='config', config_path='conf')
def train(args: DictConfig):

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    target = args.src_mod.target
    ckpt_path = args.ckpt_path
    
    # Data samplers
    tr_sampler = build_training_sampler(target, args.dset, fast_tr=args.fast_tr)
    if args.fast_tr:
        val_sampler = tr_sampler  # for debugging (overfitting on one batch)
    else:
        val_sampler = build_fulltrack_sampler(target, args.dset, subset="train", split="valid")

    # Method name
    src_mod_name = args.src_mod.name
    method_name = src_mod_name + "-" + target
    model_dir = args.out_dir + src_mod_name + "/"

    # Display the method's name
    print('Method: ', method_name)

    # Instanciate model
    model = instanciate_src_model_onetarget(
        args.optim,
        args.scheduler,
        args.src_mod,
        pretrained_src_path=ckpt_path
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
    trainer.fit(model, tr_sampler, val_sampler, ckpt_path=ckpt_path)

    return


if __name__ == "__main__":
    train()

# EOF
