import torch
import lightning.pytorch as pl


def create_trainer(
    cfg_optim,
    method_name="dummy",
    ckpt_dir="outputs/",
    ckpt_name="separator",
    log_dir="tb_logs/",
    fast_tr=False,
    ngpus=None,
    sync_bn=True,
    monitor_val ='loss'
):

    # Number of GPUs
    ngpus_max = torch.cuda.device_count()
    if ngpus is None:
        ngpus = ngpus_max
    else:
        ngpus = min(ngpus_max, ngpus)

    # Validation criterion (loss/min or SDR/max)
    monitor_val = "val_" + monitor_val
    mode = 'min'
    if 'sdr' in monitor_val:
        mode = 'max'

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_val, patience=cfg_optim.patience, verbose=False, mode=mode
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_val,
        mode=mode,
        save_top_k=1,
        dirpath=ckpt_dir,
        filename=ckpt_name,
    )

    # Tensorboard logger
    if log_dir is None:
        my_logger = False
    else:
        my_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=method_name)

    # For debugging, use the overfit_batches flag
    if fast_tr:
        overfit_batches = 1
    else:
        overfit_batches = 0.0

    # Instanciate the trainer
    trainer = pl.Trainer(
        max_epochs=cfg_optim.max_epochs,
        gradient_clip_val=cfg_optim.grad_clip_norm,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=my_logger,
        accelerator="gpu",
        strategy="ddp",
        num_nodes=1,
        devices=ngpus,
        overfit_batches=overfit_batches,
        num_sanity_val_steps=1,
        sync_batchnorm=sync_bn
    )

    return trainer

# EOF