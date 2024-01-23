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
):

    # Number of GPUs
    if ngpus is None:
        ngpus = torch.cuda.device_count()

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg_optim.patience, verbose=False, mode="min"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=ckpt_dir,
        filename=ckpt_name,
    )

    # Tensorboard logger
    if log_dir is not None:
        my_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=method_name, default_hp_metric=False)
    else:
        my_logger = False

    # For debugging, use the overfit_batches flag
    if fast_tr:
        overfit_batches=1
    else:
        overfit_batches=0.0

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
    )

    return trainer


# EOF
