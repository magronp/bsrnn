import torch
from models.pl_module import PLModule
from models.instanciate_src import instanciate_src_model
from os.path import join


class Separator(PLModule):
    def __init__(
        self,
        args,
    ):

        # Targets
        targets = args.targets
        if isinstance(targets, str):
            targets = [targets]

        # Inherit from the PLModule
        super().__init__(
            args.optim,
            args.scheduler,
            args.eval,
            targets=targets,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_hop=args.n_hop,
            eps=args.eps,
        )

        # Dir where to load checkpoint(s)
        ckpt_dir = args.ckpt_dir

        # By default, that should be none -> use predifined path structure
        if ckpt_dir is None:
            ckpt_dir = join(args.out_dir, args.model.name_out_dir)
            self.simo = "simo" in args.model.name

        else:
            self.simo = "simo" in ckpt_dir

        # Either instanciate a SIMO model from a separator checkpoint
        if self.simo:
            ckpt_path = join(ckpt_dir, "separator.ckpt")
            self.model = instanciate_src_model(
                args,
                targets=targets,
                ckpt_path=ckpt_path,
            )
        else:
            # Or instanciate a set of target-specific checkpoints
            self.source_models = torch.nn.ModuleDict(
                {
                    t: instanciate_src_model(
                        args,
                        targets=t,
                        ckpt_path=join(ckpt_dir, t + ".ckpt"),
                    )
                    for t in targets
                }
            )

    def forward(self, mix):
        # mix: [B, nch, n_samples]
        # s_est: [B, n_targets, nch, n_samples]

        if self.simo:
            s_est = self.model(mix)["waveforms"]
        else:
            s_est = [self.source_models[t](mix)["waveforms"] for t in self.targets]
            s_est = torch.cat(s_est, dim=1)

        return {"waveforms": s_est}


# EOF
