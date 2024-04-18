import torch
from models.pl_module import PLModule
from models.instanciate_src import instanciate_src_model
import os


class Separator(PLModule):
    def __init__(
        self,
        args,
    ):
        # Collect the appropriate cfg dicts
        cfg_optim = args.optim
        cfg_scheduler = args.scheduler
        cfg_eval = args.eval
        cfg_src_mod = args.src_mod
        targets = args.targets

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            cfg_eval,
            targets=targets,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_hop=args.n_hop,
            eps=args.eps,
        )

        self.source_models = torch.nn.ModuleDict(
            {
                t: instanciate_src_model(
                    cfg_optim,
                    cfg_scheduler,
                    cfg_eval,
                    cfg_src_mod,
                    pretrained_src_path=os.path.join(
                        cfg_src_mod.out_dir, cfg_src_mod.name, t + ".ckpt"
                    ),
                )
                for t in targets
            }
        )

    def forward(self, mix):
        # mix: [B, nch, n_samples]
        # s_est: [B, n_targets, nch, n_samples]
        s_est = [self.source_models[t](mix)["waveforms"] for t in self.targets]
        s_est = torch.cat(s_est, dim=1)

        return {"waveforms": s_est}

# EOF
