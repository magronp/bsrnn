import os
import torch
from models.bsrnn import BSRNN
from models.bscnn import BSCNN


# A general function that returns the model class: any other model can be added here
def get_class_from_str(model_name):
    if model_name == "bscnn":
        Model = BSCNN
    else:
        Model = BSRNN

    return Model


def instanciate_src_model(
    args,
    targets=["vocals"],
    ckpt_path=None,
):
    # Collect the appropriate cfg dicts
    cfg_optim = args.optim
    cfg_scheduler = args.scheduler
    cfg_eval = args.eval
    cfg_src_mod = args.src_mod

    # Targets
    if isinstance(targets, str):
        targets = [targets]

    # Model class
    Model = get_class_from_str(cfg_src_mod.name)

    # Load the pretrained sources only if path is provided and if they exist
    ckpt_exists = False
    if ckpt_path:
        if os.path.exists(ckpt_path):
            ckpt_exists = True

    if ckpt_exists:
        print(f"Loading checkpoint for the {targets} track(s): {ckpt_path}")
        model = Model.load_from_checkpoint(
            ckpt_path,
            cfg_optim=cfg_optim,
            cfg_scheduler=cfg_scheduler,
            cfg_eval=cfg_eval,
            map_location=torch.device("cpu"),
            strict=False,
        )

    # Otherwise, instanciate the model from scratch
    else:
        print(f"Initializing model for the {targets} track(s)")
        model = Model(cfg_optim, cfg_scheduler, cfg_eval, targets, **cfg_src_mod)

    return model


# EOF
