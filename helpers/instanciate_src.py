import os
import torch
from models.bsrnn import BSRNN


def get_class_from_str(model_name):
    if model_name == "bsrnn":
        Model = BSRNN
    else:
        raise NameError("Unknown model type")

    return Model


def instanciate_src_models_targets(
    cfg_optim,
    cfg_scheduler,
    cfg_src_mod,
    targets,
    load_pretrained_sources=False,
):
    
    models = torch.nn.ModuleDict(
        {
            t: instanciate_src_model_onetarget(
                cfg_optim,
                cfg_scheduler,
                cfg_src_mod,
                target=t,
                load_pretrained_sources=load_pretrained_sources,
            )
            for t in targets
        }
    )

    return models


def instanciate_src_model_onetarget(
    cfg_optim,
    cfg_scheduler,
    cfg_src_mod,
    target="vocals",
    load_pretrained_sources=False,
):
    device = torch.device("cpu")

    # Model class
    Model = get_class_from_str(cfg_src_mod.name)

    # Load the pretrained sources only if they exist (in "outputs/src_mod_dir/target.ckpt")
    if load_pretrained_sources:
        ckpt_path = os.path.join(cfg_src_mod.out_dir, cfg_src_mod.name, target + ".ckpt")
        ckpt_exists = os.path.exists(ckpt_path)
    else:
        ckpt_exists = False

    if ckpt_exists:
        model = Model.load_from_checkpoint(
            ckpt_path,
            cfg_optim=cfg_optim,
            cfg_scheduler=cfg_scheduler,
            map_location=device,
            strict=False
        )

    # Otherwise, instanciate the model from scratch
    else:
        model = Model(cfg_optim, cfg_scheduler, **cfg_src_mod)

    return model

# EOF
