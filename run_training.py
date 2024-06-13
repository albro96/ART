import torch
import numpy as np
import os
from torch import optim
import sys
import os.path as op
from pathlib import Path
from easydict import EasyDict
from collections import Counter
import wandb
import json
import glob
import os.path as op
import torch.multiprocessing as mp
import shutil
from pprint import pprint

BASE_DIR = Path(op.abspath(__file__)).parents[1].resolve()
sys.path.append(str(BASE_DIR))
torch.set_printoptions(precision=6)
sys.path.append("/storage/share/code/01_scripts/modules/")

from dnnlib.util import ask_yes_no
from os_tools.import_dir_path import import_dir_path
from train_fn import run_net


pada = import_dir_path()
model_dir = pada.models.art.model_dir

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def main(rank=0, world_size=1):

    data_config = EasyDict(
        {
            "num_points_gt": 8192,  # 2048, #2048,
            "num_points_corr": 0,  # 16384, #16384,  # 2048 4096 8192 16384
            "num_points_corr_type": "full",
            "num_points_gt_type": "full",
            "tooth_range": {
                "corr": "full",
                "gt": "full",  # "full",
                "jaw": "full-separate",
                "quadrants": "all",
            },
            "return_only_full_gt": True,
            "gt_type": "full",
            "data_type": "npy",
            "samplingmethod": "fps",
            "downsample_steps": 2,
            "use_fixed_split": False,
            "splits": {"train": 1, "val": 0, "test": 0},
            "enable_cache": True,
            "create_cache_file": True,
            "overwrite_cache_file": False,
            "cache_dir": op.join(
                pada.base_dir, "nobackup", "data", "3DTeethSeg22", "cache"
            ),
        }
    )

    args = EasyDict(
        {
            "num_gpus": world_size,
            "local_rank": rank,
            "num_workers": 16,  # only applies to mode='train', set to 0 for val and test
            "seed": 0,
            "experiment_dir": pada.model_base_dir,
            "start_ckpts": None,
            "ckpts": None,
            "val_freq": 10,
            "val_dataset": "train",
            "test_freq": None,
            "resume": False,
            "test": False,
            "mode": None,
            "save_checkpoints": True,
            "save_only_best": True,
            "consider_metric": "rot_loss_angle",
            "ckpt_dir": op.join(pada.model_base_dir, "ART", "ckpt"),
            "cfg_dir": op.join(pada.model_base_dir, "ART", "config"),
            "load_config_name": None,  # "expert-sweep-84",  # "bumbling-sweep-133",  # "azure-sweep-129",
            "run_name": None,  # "expert-sweep-84-1000-epochs-pyt3d-gramschmidt",  # "bumbling-sweep-133-fulldataset-unique-batch-rots",  # "azure-sweep-129-fulldataset"
            "log_data": True,  # if true: wandb logger on and save ckpts to local drive
        }
    )

    bs_dict = {
        2048: 56,
        4096: 28,
        8192: 14,
        16384: 6,
    }

    # azure-sweep-129
    config = EasyDict(
        {
            "optimizer": {
                "type": "AdamW",
                "kwargs": {
                    "lr": 0.0001,
                    "weight_decay": 0.0001,  # 0.0001
                },
            },
            "dataset": data_config,
            "model": {
                "NAME": "ART",
                "iters": 5,
                "lambda_cd": 0,
                "rot_loss_type": "rot_loss_angle",  # 'rot_loss_angle' or 'rot_loss_mse'
                "gt_type": data_config.gt_type,
                "grad_norm_clip": 8,
                "cd_norm": 2,
                "dropout_rate": 0,
                "unique_batch_rot_angles": True,
            },
            "max_epoch": 300,
            "bs": bs_dict[data_config.num_points_gt],
            "step_per_update": 1,
            "model_name": "ART",
            "train_losses": ["rot_loss_angle", "rot_loss_mse"],
            "val_losses": ["rot_loss_angle", "rot_loss_mse"],
        }
    )

    assert (
        args.consider_metric in config.val_losses
    ), f"{args.consider_metric} not in val_losses"

    assert (
        config.model.rot_loss_type in config.train_losses
    ), f"{config.model.rot_loss_type} not in train_losses"

    if config.model.lambda_cd != 0 and not "rot_loss_chamfer" in config.train_losses:
        config.train_losses.append("rot_loss_chamfer")

    if args.load_config_name is not None:
        cfg_path = os.path.join(
            args.cfg_dir, "config-" + args.load_config_name + ".json"
        )
        with open(cfg_path, "r") as json_file:
            config_load = EasyDict(json.load(json_file))

        # update model and optimizer in config
        config.model.update(config_load.model)
        config.optimizer.update(config_load.optimizer)

    if args.test and args.resume:
        raise ValueError("--test and --resume cannot be both activate")

    if args.resume and args.start_ckpts is not None:
        raise ValueError("--resume and --start_ckpts cannot be both activate")

    if args.test and args.ckpts is None:
        raise ValueError("ckpts shouldnt be None while test mode")

    if args.local_rank is not None:
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(args.local_rank)

    # CUDA
    args.use_gpu = torch.cuda.is_available()
    args.use_amp_autocast = False
    args.device = torch.device("cuda" if args.use_gpu else "cpu")

    # args.device = torch.device('cuda', args.local_rank) if args.use_gpu else torch.device('cpu')
    # print('\n\nHJKASFLDAS\n\n\n')
    # print(args.device)

    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    if config.model.iters > 5:
        config.bs = 2

    # ----------------------------- Prepare training -----------------------------

    if args.log_data:
        wandb.init(
            # set the wandb project where this run will be logged, dont set config here, else sweep will fail
            project="ART-Orientation",
            save_code=True,
        )

        # define custom x axis metric
        wandb.define_metric("epoch")

        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

        wandb.define_metric("val/pcd/gt/*", step_metric="epoch")
        wandb.define_metric("val/pcd/rot/*", step_metric="epoch")

        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        wandb_config = wandb.config
        args.sweep = True if "sweep" in wandb_config else False

        # update the model config with wandb config
        for key, value in wandb_config.items():
            if "." in key:
                keys = key.split(".")
                config_temp = config
                for sub_key in keys[:-1]:
                    config_temp = config_temp.setdefault(sub_key, {})
                config_temp[keys[-1]] = value
            else:
                config[key] = value

        wandb.config.update(config, allow_val_change=True)

        # set run name
        if not args.sweep and args.get("run_name", None) is not None:
            wandb.run.name = args.run_name
            print("Run name: ", wandb.run.name)

    else:
        args.sweep = False

    if not args.sweep and args.val_dataset != "val":
        # ask user if they want to use train dataset for validation
        if not ask_yes_no(
            f"\nUsing {args.val_dataset} dataset for validation: continue?"
        ):
            sys.exit()

    args.experiment_path = os.path.join(args.experiment_dir, config.model_name)
    args.cfg_dir = op.join(args.experiment_path, "config")
    args.ckpt_dir = op.join(args.experiment_path, "ckpt")

    if args.sweep:
        args.experiment_path = os.path.join(
            args.experiment_path, "sweep", wandb.run.sweep_id
        )
        args.save_only_best = True

    if args.log_data:
        if not os.path.exists(args.experiment_path):
            os.makedirs(args.experiment_path, exist_ok=True)
            print("Create experiment path successfully at %s" % args.experiment_path)

        shutil.copy(__file__, args.experiment_path)

        # set the wandb run dir
        # wandb.run.dir = args.experiment_path

        os.makedirs(args.cfg_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        cfg_name = f"config-{wandb.run.name}.json"

        with open(os.path.join(args.cfg_dir, cfg_name), "w") as json_file:
            json_file.write(json.dumps(config, indent=4))

    pprint(config)

    if not args.sweep and args.load_config_name is not None:
        print("Loaded config from: ", cfg_path)
        if not ask_yes_no("Continue?"):
            sys.exit()

    run_net(
        args=args,
        config=config,
    )


if __name__ == "__main__":

    # User Input
    num_gpus = 1  # number of gpus, dont use 3
    print("Number of GPUs: ", num_gpus)
    torch.autograd.set_detect_anomaly(True)

    if num_gpus > 1:
        sys.exit()
        # if num_gpus == 2:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        # elif num_gpus == 3:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        # elif num_gpus == 4:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "12345"  # Set any free port
        # os.environ["WORLD_SIZE"] = str(num_gpus)
        # # mp.spawn(main, args=(num_gpus, ), nprocs=num_gpus, join=True)
        # mp.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)
    else:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        main(rank=0, world_size=1)
