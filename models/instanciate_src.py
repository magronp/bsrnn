import os
import torch
from models.bsrnn import BSRNN


# A general function to return the model class: thus any other model can be added here
def get_class_from_str(model_name):
    if model_name == "bsrnn":
        Model = BSRNN
    else:
        raise NameError("Unknown model type")

    return Model


def instanciate_src_model(
    cfg_optim,
    cfg_scheduler,
    cfg_eval,
    cfg_src_mod,
    pretrained_src_path=None,
):
    device = torch.device("cpu")

    # Model class
    Model = get_class_from_str(cfg_src_mod.name)

    # Load the pretrained sources only if path is provided and if they exist
    ckpt_exists = False
    if pretrained_src_path:
        if os.path.exists(pretrained_src_path):
            ckpt_exists = True

    if ckpt_exists:
        model = Model.load_from_checkpoint(
            pretrained_src_path,
            cfg_optim=cfg_optim,
            cfg_scheduler=cfg_scheduler,
            cfg_eval=cfg_eval,
            map_location=device,
            strict=False,
        )

    # Otherwise, instanciate the model from scratch
    else:
        model = Model(cfg_optim, cfg_scheduler, cfg_eval, **cfg_src_mod)

    return model


# EOF
