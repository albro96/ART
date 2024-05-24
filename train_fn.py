import numpy as np
import torch
import torch.nn.functional as F
import os, time
from utils import random_rotate_batch
import sys
import wandb
import glob
import os.path as op
import torch.nn as nn
import functorch
from easydict import EasyDict
import glob
from pytorch3d.loss import chamfer_distance

sys.path.append("/storage/share/code/01_scripts/modules/")
from general_tools.format import format_duration
from tools import builder
from models import art_model


def unfold_rotenc(data, rotenc, iters=5):
    R_cum = torch.eye(3).unsqueeze(0).repeat(data.size(0), 1, 1).to(data.device)
    for _ in range(iters):
        R = rotenc(data.transpose(1, 2).contiguous())
        data = torch.matmul(data, R).detach()
        R_cum = torch.matmul(R_cum, R)
    return R_cum


def get_angle(R1, R2):
    """
    Calculate the angle between two rotation matrices.
    http://www.boris-belousov.net/2016/12/01/quat-dist/

    Parameters:
    - R1 (torch.Tensor): The first rotation matrix.
    - R2 (torch.Tensor): The second rotation matrix.

    Returns:
    - torch.Tensor: The angle between the two rotation matrices.

    """
    product = torch.matmul(R1, R2.transpose(1, 2))

    batch_trace = functorch.vmap(torch.trace)(product)

    cos_angle = torch.clamp((batch_trace - 1) / 2, -0.999, 0.999)

    # angles of batched rotation matrices
    angle = torch.acos(cos_angle)

    # return mean angle
    return torch.mean(angle)


# ----------------------------- Train Single Model ----------------------------- #


def calc_loss_rotenc(model, data, config):
    loss_dict = EasyDict()

    # Define the rotation function
    random_rotate = random_rotate_batch

    # see chapter 3.2 EQ 4 and 5
    # X^T = data -- data.shape = [num_points, 3] --> (R * data^T)^T = data * R^T
    # Apply random rotations to the data (R_tilde^T)
    data_rot_1, rotmat_1 = random_rotate(data)
    data_rot_2, rotmat_2 = random_rotate(data)
    data_rot_3, rotmat_3 = random_rotate(data)

    # Unfold the rotation encodings (R2^T) of the rotated data (R_tilde) --
    R_1 = unfold_rotenc(data_rot_1, model, config.model.iters)
    R_2 = unfold_rotenc(data_rot_2, model, config.model.iters)
    R_3 = unfold_rotenc(data_rot_3, model, config.model.iters)

    # Unfold the rotation encoding of the original data (R1^T)
    R = unfold_rotenc(data, model, config.model.iters)

    # Compute the product of the rotation matrices R_tilde = R2^T * R1 -- R_tilde^T = R1^T * R2  eq4
    # code is correct as is bcs all rot matrices are transposed to match the data shape
    rotprod_1 = torch.matmul(R, R_1.transpose(1, 2))
    rotprod_2 = torch.matmul(R, R_2.transpose(1, 2))
    rotprod_3 = torch.matmul(R, R_3.transpose(1, 2))

    # Compute the mean squared error loss between the rotation matrices and their products

    loss_dict.rot_loss_mse = (
        F.mse_loss(rotmat_1, rotprod_1)
        + F.mse_loss(rotmat_2, rotprod_2)
        + F.mse_loss(rotmat_3, rotprod_3)
    ) / 3

    # Compute the angle between the rotation matrices and their products
    loss_dict.rot_loss_angle = (
        get_angle(rotmat_1, rotprod_1)
        + get_angle(rotmat_2, rotprod_2)
        + get_angle(rotmat_3, rotprod_3)
    ) / 3

    # Compute the chamfer loss between the rotated data and the product of the original data and the rotation matrices
    # with torch.cuda.device(data.device):
    loss_dict.rot_loss_chamfer = (
        chamfer_distance(
            torch.matmul(data, rotprod_1), data_rot_1, norm=config.model.cd_norm
        )[0]
        + chamfer_distance(
            torch.matmul(data, rotprod_2), data_rot_2, norm=config.model.cd_norm
        )[0]
        + chamfer_distance(
            torch.matmul(data, rotprod_3), data_rot_3, norm=config.model.cd_norm
        )[0]
    ) / 3

    return loss_dict, data_rot_1, torch.matmul(data_rot_1, rotprod_1.transpose(1, 2))


def run_net(args, config):

    # optimizer = optim.AdamW(list(model.parameters()), lr=opt.lr)

    _, train_dataloader = builder.dataset_builder(
        args, config.dataset, mode="train", bs=config.bs
    )
    _, val_dataloader = builder.dataset_builder(args, config.dataset, mode="val", bs=1)
    # _, test_dataloader = builder.dataset_builder(
    #     args, config.dataset, mode="test", bs=1
    # )

    model = art_model.PointNetTransformNet(config=config)

    if args.use_gpu:
        model = nn.DataParallel(model).to(args.device)

    optimizer = builder.build_optimizer(model, config)

    start_epoch = 1
    best_loss = 1e8

    if op.exists(args.ckpt_dir):
        ckpts = sorted(glob.glob(op.join(args.ckpt_dir, "*.pth")))

        if args.resume and len(ckpts) > 0:
            ckpt_file = ckpts[-1]
            ckpt = torch.load(ckpt_file, map_location=args.device)

            model.load_state_dict(ckpt)
            optimizer.load_state_dict(ckpt)

            # get epoch from state dict
            start_epoch = ckpt["epoch"] + 1

    print("Training started")
    epoch_times = []

    epoch_time_list = []

    for epoch in range(start_epoch, config.max_epoch + 1):
        epoch_allincl_start_time = time.time()

        print("\nEpoch {}/{}: Training".format(epoch, config.max_epoch))

        model.train()

        train_loss_dict = {
            "total_loss": 0,
            "rot_loss_mse": 0,
            "rot_loss_chamfer": 0,
            "rot_loss_angle": 0,
        }

        n_batches = len(train_dataloader)
        batch_start_time = time.time()

        for idx, data in enumerate(train_dataloader):
            print(
                "\rBatch {}/{}".format(idx + 1, len(train_dataloader)),
                end="",
                flush=True,
            )

            data = data.to(
                args.device
            )  # teethseg dataloader return corr, gt -- we only want gt here

            # Zero the gradients of the optimizer
            optimizer.zero_grad()

            loss_dict, _, _ = calc_loss_rotenc(model, data, config)

            # Compute the total loss
            # loss =  rot_loss_mse * 0.02 + rot_loss_chamfer * opt.lambda2
            # loss = (
            #     loss_dict.rot_loss_mse
            #     + loss_dict.rot_loss_chamfer * config.model.lambda_cd
            # )

            loss = (
                loss_dict[config.model.rot_loss_type]
                + loss_dict.rot_loss_chamfer * config.model.lambda_cd
            )

            # Backpropagate the loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.model.grad_norm_clip,
                norm_type=2,
            )

            # Update the model parameters
            optimizer.step()
            model.zero_grad()

            # print(loss.item(), rot_loss_chamfer.item(), rot_loss_mse.item())

            train_loss_dict["total_loss"] += loss * data.size(0)
            for key, val in loss_dict.items():
                if key != "total_loss":
                    train_loss_dict[key] += val.item() * data.size(0)

            # train_loss_dict["rot_loss_mse"] += rot_loss_mse.item() * data.size(0)
            # train_loss_dict["rot_loss_chamfer"] += rot_loss_chamfer.item() * data.size(
            #     0
            # )

        batch_time = time.time() - batch_start_time

        for key, val in train_loss_dict.items():
            train_loss_dict[key] = val / len(train_dataloader.dataset)

        if args.log_data:
            log_dict = {"epoch": epoch}
            for key, val in train_loss_dict.items():
                log_dict[f"train/{key}"] = val

            wandb.log(
                log_dict,
                step=epoch,
            )

        # epoch progress
        if not args.sweep:
            print(
                f'\r[Epoch {epoch}/{config.max_epoch}][Batch {idx}/{n_batches}] BatchTime = {format_duration(batch_time)} Losses = {[f"{key}: {val:.4f}"  for key, val in train_loss_dict.items()]} lr = {optimizer.param_groups[0]["lr"]:.6f}',
                end="\r",
            )

        if args.sweep and args.log_data:
            print(
                f'\rAgent: {wandb.run.id} Epoch [{epoch}/{config.max_epoch}] Losses = {[f"{key}: {val:.4f}"  for key, val in train_loss_dict.items()]} lr = {optimizer.param_groups[0]["lr"]:.6f}'
            )

        if epoch % args.val_freq == 0:

            val_loss_dict = validate(model, val_dataloader, epoch, args, config)

            if args.log_data:
                checkpoint = {
                    "m": model.state_dict() if model else None,
                    "o": optimizer.state_dict(),
                    "torch_rnd": torch.get_rng_state(),
                    "numpy_rnd": np.random.get_state(),
                }

                if val_loss_dict["total_loss"] < best_loss:
                    best_loss = val_loss_dict["total_loss"]
                    torch.save(
                        checkpoint,
                        os.path.join(args.ckpt_dir, f"ckpt-best-{wandb.run.name}.pth"),
                    )

        if args.save_checkpoints and not args.save_only_best and args.log_data:
            torch.save(
                checkpoint,
                os.path.join(
                    args.ckpt_dir, f"ckpt-epoch-{epoch:03d}-{wandb.run.name}.pth"
                ),
            )

        epoch_time_list.append(time.time() - epoch_allincl_start_time)

        mean_epoch_time = sum(epoch_time_list) / len(epoch_time_list)

        est_time = mean_epoch_time * (config.max_epoch - epoch + 1)

        print(
            f'[Training] EPOCH: {epoch}/{config.max_epoch} EpochTime = {format_duration(epoch_time_list[-1])} Remaining Time = {format_duration(est_time)} Losses = {[f"{key}: {val:.4f}"  for key, val in train_loss_dict.items()]} \n'
        )


def validate(model, val_dataloader, epoch, args, config):
    print("\nEpoch {}/{}: Validation".format(epoch, config.max_epoch))

    model.eval()
    val_loss_dict = {
        "total_loss": 0,
        "rot_loss_mse": 0,
        "rot_loss_chamfer": 0,
        "rot_loss_angle": 0,
    }

    # rdm_idx = np.random.randint(0, len(val_dataloader.dataset))

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            print(
                "\rBatch {}/{}".format(i + 1, len(val_dataloader)),
                end="",
                flush=True,
            )

            data = data.to(args.device)

            loss_dict, pcd_rot, pcd_backrot = calc_loss_rotenc(model, data, config)

            # loss = (
            #     loss_dict.rot_loss_mse
            #     + loss_dict.rot_loss_chamfer * config.model.lambda_cd
            # )

            loss = (
                loss_dict[config.model.rot_loss_type]
                + loss_dict.rot_loss_chamfer * config.model.lambda_cd
            )

            val_loss_dict["total_loss"] += loss * data.size(0)
            for key, val in loss_dict.items():
                if key != "total_loss":
                    val_loss_dict[key] += val.item() * data.size(0)

            if val_dataloader.dataset.patient == "0U1LI1CB" and args.log_data:
                suffix = (
                    f"-{val_dataloader.dataset.jaw}"
                    if config.dataset.tooth_range.jaw == "full-separate"
                    else ""
                )

                if epoch == args.val_freq:
                    # save gt only once
                    wandb.log(
                        {
                            f"val/pcd/gt{suffix}": wandb.Object3D(
                                {
                                    "type": "lidar/beta",
                                    "points": data[0].detach().cpu().numpy(),
                                }
                            ),
                        },
                        step=epoch,
                    )

                wandb.log(
                    {
                        f"val/pcd/rot{suffix}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": pcd_rot[0].detach().cpu().numpy(),
                            }
                        ),
                        f"val/pcd/backrot{suffix}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": pcd_backrot[0].detach().cpu().numpy(),
                            }
                        ),
                    },
                    step=epoch,
                )

    print(
        "Epoch {}/{}: Validation".format(epoch, config.max_epoch),
        flush=True,
    )

    print()
    for term in val_loss_dict:
        val_loss_dict[term] = val_loss_dict[term] / len(val_dataloader.dataset)
        print("\t{}: {:.5f}".format(term, val_loss_dict[term]))

    if args.log_data:
        log_dict = {}
        for key, val in val_loss_dict.items():
            log_dict[f"val/{key}"] = val
        wandb.log(
            log_dict,
            step=epoch,
        )

    return val_loss_dict
