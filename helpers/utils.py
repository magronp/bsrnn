import torchaudio
from os.path import join
from pathlib import Path

# pb avec le "path separator" (commentaire donné), car on devrait pouvoir record tout le séparateur
# mieux gérer les dossiers / chemins


def get_model_info(
    args,
    out_dir="outputs/",
):
    sep_type = args.separator

    # Treat specifically deephr (more involved)
    if sep_type == "deephr":
        spectro_model = args.spectro_model.name
        model_info = get_deephr_info(
            args.spec_inv.use_phase_prior,
            args.spec_inv.algo,
            args.spec_inv.iter,
            spectro_model=spectro_model,
            time_domain_tr=args.spec_inv.time_domain_tr,
            out_dir=out_dir,
        )

    elif sep_type == "sepstft":
        src_mod = args.stft_model.name
        out_dir = join(out_dir, sep_type, src_mod)
    
        model_info = {
            "model_dir": out_dir,
            "rec_dir": join(out_dir, "audio"),
            "path_pretrained_separator": None,
            "path_separator": None,  # in that case the separator is not recorded since each src is trained separately (needed for eval)
            "method_name": sep_type + "-" + src_mod,
        }

    elif sep_type == "sepphase":
        src_mod = args.phase_model.name
        out_dir = join(out_dir, sep_type, src_mod)
        model_info = {
            "model_dir": out_dir,
            "rec_dir": join(out_dir, "audio"),
            "path_pretrained_separator": None,
            "path_separator": None,  # in that case the separator is not recorded since each src is trained separately (needed for eval)
            "method_name": sep_type + "-" + src_mod,
        }

    return model_info


def get_deephr_info(
    use_phase_prior,
    spec_inv_algo,
    spec_inv_iter,
    spectro_model="umx",
    time_domain_tr=False,
    out_dir="outputs/",
):
    if time_domain_tr:
        model_info = get_deephr_info_td(
            use_phase_prior, spec_inv_algo, spec_inv_iter, spectro_model, out_dir
        )
    else:
        model_info = get_deephr_info_notd(
            use_phase_prior, spec_inv_algo, spectro_model, out_dir
        )

    return model_info


def get_deephr_info_notd(
    use_phase_prior, spec_inv_algo, spectro_model="umx", out_dir="outputs/"
):
    
    # Prepare the model dir: account for the spectro model, and phase prior
    pp = "phprior/" if use_phase_prior else "noprior/"
    aux_dir = join(out_dir, "deephr", spectro_model, "notd", pp)

    # Model folder, along with the audio folder (for storing the estimated audio)
    model_dir = aux_dir + spec_inv_algo + "/"
    rec_dir = model_dir + "audio/"
    Path(rec_dir).mkdir(parents=True, exist_ok=True)

    # Define the method name (for storing results purpose)
    method_name = model_dir[:-1].replace(out_dir, "").replace("/", "-")

    # Store the info in a dict
    model_info = {
        "model_dir": model_dir,
        "rec_dir": rec_dir,
        "path_pretrained_separator": None,
        "path_separator": None,
        "method_name": method_name,
    }

    return model_info


def get_deephr_info_td(
    use_phase_prior, spec_inv_algo, spec_inv_iter, spectro_model="umx", out_dir="outputs/"
):
    # Prepare the model dir: account for the spectro model, td training, and phase prior
    pp = "phprior/" if use_phase_prior else "noprior/"
    aux_dir = join(out_dir, "deephr", spectro_model, "td", pp)

    # Model dir
    if spec_inv_algo in ["AM", "Incons_hardMix"]:
        model_dir = aux_dir + spec_inv_algo + "/"
    else:
        model_dir = aux_dir + spec_inv_algo + str(spec_inv_iter) + "/" # account for the number of iterations

    # Rec dir, path speparator, and method name
    rec_dir = model_dir + "audio/"
    Path(rec_dir).mkdir(parents=True, exist_ok=True)
    path_separator = model_dir + "separator.ckpt"
    method_name = model_dir[:-1].replace(out_dir, "").replace("/", "-")

    # Path pretraining
    path_pretrained_separator = None
    if spec_inv_iter > 1:
        path_pretrained_separator = path_separator.replace(spec_inv_algo + str(spec_inv_iter), spec_inv_algo + str(spec_inv_iter - 1))

    # Store the info in a dict
    model_info = {
        "model_dir": model_dir,
        "rec_dir": rec_dir,
        "path_pretrained_separator": path_pretrained_separator,
        "path_separator": path_separator,
        "method_name": method_name,
    }

    return model_info


def rec_estimates(estimates, track_rec_dir, targets, sample_rate):
    """
    estimates: [n_targets, n_channels, n_samples]
    """

    # Make sure the estimates tensor is detached and on cpu
    estimates = estimates.cpu().detach()

    # create the rec folder if needed
    Path(track_rec_dir).mkdir(parents=True, exist_ok=True)

    # Loop over targets
    for ind_trg, trg in enumerate(targets):
        torchaudio.save(
            join(track_rec_dir, trg + ".wav"),
            estimates[ind_trg],
            sample_rate,
        )

    return


# EOF
