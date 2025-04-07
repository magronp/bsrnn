import torch
from omegaconf import OmegaConf


ckpt_path="outputs/bsrnn-opt/other.ckpt"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
conf = OmegaConf.create(checkpoint["hyper_parameters"])
print(OmegaConf.to_yaml(conf))
print(checkpoint["lr_schedulers"][0]["last_epoch"])
