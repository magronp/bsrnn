import torch
from models.pl_module import PLModule
from helpers.instanciate_src import instanciate_src_models_targets


class Separator(PLModule):
    def __init__(
        self,
        args,
    ):
        # Collect the appropriate cfg dicts
        cfg_optim = args.optim
        cfg_scheduler = args.scheduler
        cfg_src_mod = args.src_mod
        targets = args.targets

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            targets=targets,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_hop=args.n_hop,
            eps=args.eps
        )

        # Source models
        self.source_models = instanciate_src_models_targets(
            cfg_optim,
            cfg_scheduler,
            cfg_src_mod,
            targets=targets,
            load_pretrained_sources=args.load_pretrained_sources,
        )

    def forward(self, mix):
        # mix: [B, nch, n_samples]
        # s_est: [B, n_targets, nch, n_samples]
        s_est = [self.source_models[t](mix)["waveforms"] for t in self.targets]
        s_est = torch.cat(s_est, dim=1)

        return {"waveforms": s_est}

# EOF
