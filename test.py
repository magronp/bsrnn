import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from helpers.data import build_training_samplers
from helpers.trainer import create_trainer
from models.instanciate_src import instanciate_src_model
from os.path import join
from codecarbon import OfflineEmissionsTracker


@hydra.main(version_base=None, config_name="config", config_path="conf")
def train(args: DictConfig):

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Get the target to train
    target = args.targets
    assert target in ["vocals", "bass", "drums", "other"], "Unknown instrument"

    # Instanciate model
    model = instanciate_src_model(
        args.optim,
        args.scheduler,
        args.eval,
        args.src_mod,
        target=target,
        pretrained_src_path=args.ckpt_path,
    )
    print("Number of parameters: ", model.count_params())

    return


if __name__ == "__main__":
    train()

# EOF
