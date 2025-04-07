import torch
import lightning.pytorch as pl


def create_trainer(
    cfg_optim,
    ckpt_dir="outputs/",
    ckpt_name="separator",
    log_dir="tb_logs/",
    fast_tr=False,
):

    # Number of GPUs
    ngpus = cfg_optim.ngpus
    ngpus_max = torch.cuda.device_count()
    if ngpus is None:
        ngpus = ngpus_max
    else:
        ngpus = min(ngpus_max, ngpus)

    # Validation criterion (loss/min or SDR/max)
    monitor_val = "val_" + cfg_optim.monitor_val
    mode = "min"
    if "sdr" in monitor_val:
        mode = "max"

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_val, patience=cfg_optim.patience, verbose=False, mode=mode
    )
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_val,
        mode=mode,
        save_top_k=1,
        dirpath=ckpt_dir,
        filename=ckpt_name,
    )

    last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last-" + ckpt_name,
    )

    # Tensorboard logger
    if log_dir is not None:
        my_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name="")
        vnum = my_logger.log_dir.replace(log_dir + 'version_', '')
    else:
        my_logger, vnum = False, None

    # For debugging, use the overfit_batches flag
    if fast_tr:
        overfit_batches = 1
    else:
        overfit_batches = 0.0

    # Instanciate the trainer
    trainer = pl.Trainer(
        max_epochs=cfg_optim.max_epochs,
        gradient_clip_val=cfg_optim.grad_clip_norm,
        callbacks=[
            early_stop_callback,
            best_checkpoint_callback,
            last_checkpoint_callback,
        ],
        logger=my_logger,
        accelerator="gpu",
        strategy="ddp",
        num_nodes=1,
        devices=ngpus,
        overfit_batches=overfit_batches,
        num_sanity_val_steps=1,
        sync_batchnorm=cfg_optim.sync_bn,
        accumulate_grad_batches=cfg_optim.acc_grad,
        deterministic=cfg_optim.deterministic,
    )

    return trainer, ngpus, vnum


# EOF
