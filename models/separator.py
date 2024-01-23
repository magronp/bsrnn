import torch
import os
from models.pl_module import PLModule
from models.instanciate_src import instanciate_src_models_targets
from helpers.spec_inv import SpectrogramInversion


class SepSTFT(PLModule):
    def __init__(
        self,
        args,
    ):
        # Collect the appropriate cfg dicts
        cfg_optim = args.optim
        cfg_scheduler = args.scheduler
        cfg_src_mod = args.stft_model
        targets = args.targets

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            targets=targets,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_hop=args.n_hop,
            eps=args.eps,
            module_type="time",
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


class SepPhase(PLModule):
    def __init__(
        self,
        args,
    ):
        # Collect the appropriate cfg dicts
        cfg_optim = args.optim
        cfg_scheduler = args.scheduler
        cfg_src_mod = args.phase_model
        targets = args.targets

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            targets=targets,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_hop=args.n_hop,
            eps=args.eps,
            module_type="phase",
        )

        # Source models
        self.phase_models = instanciate_src_models_targets(
            cfg_optim,
            cfg_scheduler,
            cfg_src_mod,
            targets=targets,
            load_pretrained_sources=args.load_pretrained_sources,
        )

    def forward(self, X, V):
        # X: [B, nch, F, T]
        # V: [B, n_targets, nch, F, T]
        # phi_est: [B, n_targets, nch, F, T]

        phi = [
            self.phase_models[t](X, V[:, i])["phases"]
            for i, t in enumerate(self.targets)
        ]
        phi = torch.cat(phi, dim=1)

        return {"phases": phi}


class DeePhR(PLModule):
    def __init__(
        self,
        args,
    ):
        # Collect the appropriate cfg dicts
        cfg_optim = args.optim
        cfg_scheduler = args.scheduler
        cfg_spectro_mod = args.spectro_model
        cfg_phase_mod = args.phase_model
        targets = args.targets

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            targets=targets,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_hop=args.n_hop,
            eps=args.eps,
            module_type="time",
        )
        
        # Spectrogram models
        self.spectro_models = instanciate_src_models_targets(
            cfg_optim,
            cfg_scheduler,
            cfg_spectro_mod,
            targets=targets,
            load_pretrained_sources=args.load_pretrained_sources,
        )

        # Phase prior models
        self.use_phase_prior = args.spec_inv.use_phase_prior
        if self.use_phase_prior:
            self.phase_prior_models = instanciate_src_models_targets(
                cfg_optim,
                cfg_scheduler,
                cfg_phase_mod,
                targets=targets,
                load_pretrained_sources=args.load_pretrained_sources,
            )

        # Spectrogram inversion
        self.spec_inv_algo = args.spec_inv.algo
        self.spec_inv_iter = args.spec_inv.iter
        self.spinv_algo = SpectrogramInversion(
            self.stft, self.istft, algo=self.spec_inv_algo, max_iter=self.spec_inv_iter, eps=self.eps
        )

        # For the algorithms that require it, define a (learnable) consistency weight
        if self.spec_inv_algo in [
            "MagIncons_hardMix",
            "MixIncons",
            "MixIncons_hardMag",
        ]:
            if args.spec_inv.time_domain_tr:
                self.cons_weight = torch.nn.Parameter(
                    torch.tensor(args.spec_inv.consistency_weight, dtype=float)
                )
            else:
                self.cons_weight = args.spec_inv.consistency_weight
        else:
            self.cons_weight = None

    def forward(self, mix):
        # First, compute the mixture's STFT
        mix_stft = self.stft(mix)

        # Spectrogram separation
        V_est = [
            self.spectro_models[t](torch.abs(mix_stft))["magnitudes"]
            for t in self.targets
        ]
        V_est = torch.cat(V_est, dim=1)

        # Phase prior
        if self.use_phase_prior:
            phase_ini = [
                self.phase_prior_models[t](torch.angle(mix_stft), V_est[:, i, ...])[
                    "phases"
                ]
                for i, t in enumerate(self.targets)
            ]
            phase_ini = torch.cat(phase_ini, dim=1)
        else:
            # otherwise, use the mixture's phase
            phase_ini = (
                torch.angle(mix_stft).unsqueeze(1).repeat(1, len(self.targets), 1, 1, 1)
            )

        # Spectrogram inversion
        n_samples = mix.shape[-1]
        Y_hat = self.spinv_algo(
            V_est,
            mix_stft,
            phase_ini,
            cons_weight=self.cons_weight,
            audio_len=n_samples,
        )

        y_hat = self.istft(Y_hat, length=n_samples)

        return {"waveforms": y_hat}


class OracleMag(PLModule):
    def __init__(
        self,
        args,
    ):
        # Collect the appropriate cfg dicts
        cfg_optim = args.optim
        cfg_scheduler = args.scheduler
        targets = args.targets

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            targets=targets,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_hop=args.n_hop,
            eps=args.eps,
            module_type="oracle",
        )

        # Spectrogram inversion algorithm
        self.spec_inv_algo = args.spec_inv.algo
        self.spec_inv_iter = args.spec_inv.iter
        self.spinv_algo = SpectrogramInversion(
            self.stft, self.istft, algo=self.spec_inv_algo, max_iter=self.spec_inv_iter, eps=self.eps
        )

        # For the algorithms that require it, define the consistency weight
        if self.spec_inv_algo in [
            "MagIncons_hardMix",
            "MixIncons",
            "MixIncons_hardMag",
        ]:
            self.cons_weight = args.spec_inv.consistency_weight
        else:
            self.cons_weight = None

    def forward(self, y):

        Y = self.stft(y)
        V_est = torch.abs(Y)
        mix_stft = torch.sum(Y, dim=1)

        # initial phase (=mixture's)
        phase_ini = (
            torch.angle(mix_stft).unsqueeze(1).repeat(1, len(self.targets), 1, 1, 1)
        )

        # Spectrogram inversion
        n_samples = y.shape[-1]
        Y_hat = self.spinv_algo(
            V_est,
            mix_stft,
            phase_ini,
            cons_weight=self.cons_weight,
            audio_len=n_samples,
        )

        y_hat = self.istft(Y_hat, length=n_samples)

        return {"waveforms": y_hat}


class OraclePhase(PLModule):
    def __init__(
        self,
        args,
    ):
        # Collect the appropriate cfg dicts
        cfg_optim = args.optim
        cfg_scheduler = args.scheduler
        cfg_spectro_mod = args.spectro_model
        targets = args.targets

        # Inherit from the PLModule
        super().__init__(
            cfg_optim,
            cfg_scheduler,
            targets=targets,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_hop=args.n_hop,
            eps=args.eps,
            module_type="oracle",
        )
        
        # Spectrogram models
        self.spectro_models = instanciate_src_models_targets(
            cfg_optim,
            cfg_scheduler,
            cfg_spectro_mod,
            targets=targets,
            load_pretrained_sources=args.load_pretrained_sources,
        )


    def forward(self, y):

        n_samples = y.shape[-1]

        # Compute STFT of references (to get the phase) and mixture
        ref_stft = self.stft(y)
        ref_stft_phase = torch.angle(ref_stft) 
        mix_stft = torch.sum(ref_stft, dim=1)

        # Spectrogram separation
        V_est = [
            self.spectro_models[t](torch.abs(mix_stft))["magnitudes"]
            for t in self.targets
        ]
        V_est = torch.cat(V_est, dim=1)

        # Plug the oracle phases and iSTFT
        Y_hat = torch.mul(V_est, torch.exp(1j * ref_stft_phase))
        y_hat = self.istft(Y_hat, length=n_samples)

        return {"waveforms": y_hat}
    

def load_separator(args, ckpt_path=None):
    device = torch.device("cpu")

    # TODO: à tenter de passer sous python3.10 pour remplacer par "match"

    # Model class
    sep_type = args.separator

    if sep_type == "deephr":
        Model = DeePhR
    elif sep_type == "sepstft":
        Model = SepSTFT
    elif sep_type == "sepphase":
        Model = SepPhase
    elif sep_type == "oraclemag":
        Model = OracleMag
    elif sep_type == "oraclephase":
        Model = OraclePhase
    else:
        raise NameError("Unknown model type")

    # Check if a (pre)trained separator checkpoint is provided, and if it exists
    if ckpt_path:
        ckpt_exists = os.path.exists(ckpt_path)
    else:
        ckpt_exists = False

    # Load pretrained separator
    if ckpt_exists:
        model = Model.load_from_checkpoint(
            ckpt_path, args=args, map_location=device, strict=False
        )

    # Otherwise, instanciate a model using the provided parameters
    else:
        model = Model(args)

    return model


# EOF
