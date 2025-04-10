import torch
from omegaconf import OmegaConf
from os.path import join
import argparse


def display_ckpt(model_dir="bsrnn", target="vocals", out_dir="outputs"):

    # Load checkpoint
    ckpt_path = join(out_dir, model_dir, target + ".ckpt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Get and display hyperparameters
    conf = OmegaConf.create(checkpoint["hyper_parameters"])
    print(OmegaConf.to_yaml(conf))

    # Get and display the last epoch
    last_ep = checkpoint["lr_schedulers"][0]["last_epoch"]
    print(f"Last epoch: {last_ep}")

    return


def update_ckpt(
    model_dir="bsrnn", target="vocals", out_dir="outputs", param="patience", value=30
):

    # Check if the update is implemented (you can add your own)
    if not (param in ["patience"]):
        print("Unkown parameter to update")
        return

    # Define paths
    ckpt_path = join(out_dir, model_dir, target + ".ckpt")
    ckpt_path_old = join(out_dir, model_dir, target + "-old.ckpt")

    # Load checkpoint, and save a duplicate
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.save(checkpoint, ckpt_path_old)

    # Update the checkpoint
    if param == "patience":
        checkpoint["hyper_parameters"]["cfg_optim"]["patience"] = value
        checkpoint["callbacks"]["EarlyStopping{'monitor': 'val_sdr', 'mode': 'max'}"][
            "patience"
        ] = value

    # Record updated checkpoint
    torch.save(checkpoint, ckpt_path)

    print(f"Checkpoint updated at: {ckpt_path}")
    print(f"Checkpoint backed-up at: {ckpt_path_old}")

    return


if __name__ == "__main__":

    # Get input arguments
    parser = argparse.ArgumentParser()

    # ckpt location
    parser.add_argument("-m", "--model_dir", default="bsrnn-opt")
    parser.add_argument("-t", "--target", default="vocals")
    parser.add_argument("-o", "--out_dir", default="outputs")

    # update parameter
    parser.add_argument("-u", "--updateckpt", default=False)
    parser.add_argument("-p", "--param", default="patience")
    parser.add_argument("-v", "--value", default=30)
    args = parser.parse_args()

    # Display config and last epoch
    display_ckpt(model_dir=args.model_dir, target=args.target, out_dir=args.out_dir)

    # Update checkpoint
    if args.updateckpt:
        update_ckpt(
            model_dir=args.model_dir,
            target=args.target,
            out_dir=args.out_dir,
            param=args.param,
            value=args.value,
        )

# EOF
