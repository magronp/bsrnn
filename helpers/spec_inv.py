import torch
from helpers.transforms import mySTFT, myISTFT
from openunmix.filtering import wiener


def mwf(
    V_est,
    mix_stft,
    wiener_win_len=300,
    spec_inv_iter=1,
    is_softmask=False,
    audiotype=torch.float32,
):
    # Get some sizes
    nb_samples = V_est.shape[0]
    nb_frames = V_est.shape[-1]
    nb_sources = V_est.shape[1]

    if wiener_win_len is None:
        wiener_win_len = nb_frames

    # transposing target spectros as (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
    spectrograms = V_est.permute(0, 4, 3, 2, 1)

    # rearranging mix_stft into (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed  filtering methods
    mix_stft_r = torch.view_as_real(mix_stft).permute(0, 3, 2, 1, 4)

    # Initialize the target sources tensor
    targets_stft = torch.zeros(
        mix_stft_r.shape + (nb_sources,), dtype=audiotype, device=mix_stft_r.device
    )

    for sample in range(nb_samples):
        pos = 0
        while pos < nb_frames:
            cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
            pos = int(cur_frame[-1]) + 1

            targets_stft[sample, cur_frame] = wiener(
                spectrograms[sample, cur_frame],
                mix_stft_r[sample, cur_frame],
                spec_inv_iter,
                softmask=is_softmask,
                residual=False,
            )

    # getting to (nb_samples, nb_targets, channel, fft_size, n_frames)
    targets_stft = torch.view_as_complex(
        targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()
    )

    return targets_stft


class SpectrogramInversion(object):
    def __init__(self, stft, istft, algo="MISI", max_iter: int = 5, eps: float = 1e-7):
        self.stft = stft
        self.istft = istft
        self.eps = eps
        self.max_iter = max_iter
        self.algo = algo

        self.algo_fn = getattr(self, "_" + algo)


    def _pcons(self, Y: torch.Tensor, audio_len: int = None) -> torch.Tensor:
        """
        Projector on the consistent matrices' subspace, which consists of an inverse STFT followed by an STFT
        Y: [B, n_targets, F, T]
        output: [B, n_targets, F, T]
        """
        return self.stft(self.istft(Y, length=audio_len))

    def _pmix(
        self, Y: torch.Tensor, X: torch.Tensor, mixing_weights=None
    ) -> torch.Tensor:
        """
        Projector on the conservative subspace, which consists in calculating the mixing error and distributing
        it over the components
        Y: [B, n_targets, F, T]
        X: [B, F, T]
        """
        if mixing_weights is None:
            mixing_weights = 1 / Y.shape[1]
        mixing_error = X - torch.sum(Y, dim=1)
        return Y + mixing_error.unsqueeze(1) * mixing_weights

    def _pmag(self, Y: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Projector on the target magnitudes subspace: the magnitude of the sources is set at the target values
        Y: [B, n_targets, F, T]
        V: [B, n_targets, F, T]
        """
        return Y * V / (torch.abs(Y) + self.eps)

    def _AM(self, Y, V, X, cons_weight=None, audio_len=None):
        return Y

    def _MISI(self, Y, V, X, cons_weight=None, audio_len=None):
        return self._pmix(self._pmag(self._pcons(Y, audio_len), V), X)

    def _Incons_hardMix(self, Y, V, X, cons_weight=None, audio_len=None):
        return self._pmix(self._pcons(Y, audio_len), X)

    def _MagIncons_hardMix(self, Y, V, X, cons_weight=None, audio_len=None):
        return self._pmix(
            (self._pmag(Y, V) + cons_weight * self._pcons(Y, audio_len))
            / (1 + cons_weight),
            X,
        )

    def _MixIncons_hardMag(self, Y, V, X, cons_weight=None, audio_len=None):
        mix_weights = V / (torch.sum(V, keepdim=True, dim=1) + self.eps)
        return self._pmag(
            self._pmix(Y, X, mix_weights)
            + cons_weight * mix_weights * self._pcons(Y, audio_len),
            V,
        )

    def _MixIncons(self, Y, V, X, cons_weight=None, audio_len=None):
        mix_weights = V / (torch.sum(V, keepdim=True, dim=1) + self.eps)
        Ymix = self._pmix(Y, X, mix_weights)
        Ycons = self._pcons(Y, audio_len)
        Y = (Ymix + cons_weight * mix_weights * Ycons) / (1 + cons_weight * mix_weights)
        return Y

    def _SW1(self, V, X):
        return X.unsqueeze(1) * V / (torch.sum(V, keepdim=True, dim=1) + self.eps)

    def _SW2(self, V, X):
        return (
            X.unsqueeze(1)
            * V**2
            / (torch.sum(V**2, keepdim=True, dim=1) + self.eps)
        )

    def _MWF(self, V, X):
        return mwf(V, X, spec_inv_iter=self.max_iter)

    def __call__(
        self, V, X, phase_ini=None, cons_weight=None, audio_len=None
    ) -> torch.Tensor:
        # Process Wiener filters separately (no need to iterate explicitly)
        if self.algo in ["MWF", "SW1", "SW2"]:
            Y = self.algo_fn(V, X)
        else:
            # Initial estimate
            Y = V * torch.exp(1j * phase_ini)

            # Iterations
            for _ in range(self.max_iter):
                Y = self.algo_fn(Y, V, X, cons_weight=cons_weight, audio_len=audio_len)

        return Y


if __name__ == "__main__":

    stft = mySTFT(n_fft=2048, n_hop=512)
    istft = myISTFT(n_fft=2048, n_hop=512)

    torch.manual_seed(0)

    ytrue = torch.randn((2, 4, 2, 5000), dtype=torch.float64)
    x = torch.sum(ytrue, dim=1)
    Ytrue = stft(ytrue)
    X = torch.sum(Ytrue, dim=1)
    V = torch.abs(Ytrue)
    phase_ini = torch.angle(X).unsqueeze(1)

    spinv_algo = SpectrogramInversion(stft, istft, algo='SW1', max_iter=0)
    Xsw1 = spinv_algo(V, X, phase_ini, cons_weight=None, audio_len=x.shape[-1])

    spinv_algo = SpectrogramInversion(stft, istft, algo='MixIncons', max_iter=1)
    Xmixincons = spinv_algo(V, X, phase_ini, cons_weight=0, audio_len=x.shape[-1])

    print(torch.linalg.norm(Xsw1 - Xmixincons))

# EOF
