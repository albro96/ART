import torch
import numpy as np
import os, argparse
from torch import optim
import sys
import os.path as op
from pathlib import Path
from easydict import EasyDict 
from datetime import datetime
from collections import Counter
import wandb
import hashlib
import json
import math

BASE_DIR = Path(op.abspath(__file__)).parents[1].resolve()
sys.path.append(str(BASE_DIR))

from dataset import ShapeNet_PC
from loss import chamfer_distance
from models import pointnet, art_model
from train_fn import train_model, train_single_model
import glob
import os.path as op

torch.set_printoptions(precision=6)

sys.path.append('/storage/share/code/01_scripts/modules/')
from os_tools.import_dir_path import import_dir_path

pada = import_dir_path()
model_dir = pada.models.art.model_dir

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description='Training options')

    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--art', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--lambda_mse', type=int, default=2)
    parser.add_argument('--lambda_cd', type=int, default=50)
    parser.add_argument('--resamplemode', type=str, default='fps')
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--cache_data', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epoch_save', type=int, default=50)
    parser.add_argument('--val_step', type=int, default=1)
    parser.add_argument('--end_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='/storage/share/nobackup/data/ShapeNet55-34/shapenet_pc')
    parser.add_argument('--category', type=str, default='02691156')

    return EasyDict(parser.parse_args())

opt = parse_args()

if opt.iters > 5:
    opt.batch_size = 8

# opt = EasyDict()

# opt.size = 128
# opt.art = True
# opt.resume = False
# opt.iters = 5
# opt.lambda_mse = 2
# opt.lambda_cd = 50
# opt.resamplemode = 'fps'
# opt.num_points = 4096
# opt.cache_data = True
# opt.batch_size = 16
# opt.num_workers = 16
# opt.epoch_save = 50
# opt.val_step = 1
# opt.end_epoch = 500 
# opt.lr = 1e-4
# opt.data_dir = '/storage/share/nobackup/data/ShapeNet55-34/shapenet_pc'
# opt.category = '02691156' 

# get all categories from os.listdir(opt.data_dir) as path.split('-')[0] and make unique list
files = os.listdir(opt.data_dir)

category_counts = Counter(f.split('-')[0] for f in files)

categories = [category for category, _ in category_counts.most_common(5)]

# print number of files in each category
for category in categories:
    print(f'{category}: {category_counts[category]}')

data_paths = glob.glob(op.join(opt.data_dir, f'{categories[0]}*.npy'))
dataset = 'ShapeNet'

suffix = f''

# create unique hash from opt
opt_str = json.dumps(opt, sort_keys=True)
opt_hash = hashlib.sha256(opt_str.encode()).hexdigest()[:8]

print(f'opt_hash: {opt_hash}')

# runname with date in format YYMMDD from datetime function
runname = f'{datetime.now().strftime("%y%m%d")}_category-{opt.category}-{opt_hash}'

runname += suffix

save_dir = op.join(model_dir, dataset, runname)

os.makedirs(save_dir, exist_ok=True)

# save json
with open(op.join(save_dir, 'config.json'), 'w') as f:
    json.dump(opt, f, indent=4)

wandb.init(
    # set the wandb project where this run will be logged
    project="ART",
    # track hyperparameters and run metadata
    config=opt,
    name=f'{dataset}-{runname}',
    dir=save_dir,
    save_code=True,
)

with wandb.init(config=opt):
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    opt = wandb.config

    # ----------------------------- Prepare data -----------------------------
    train_set = ShapeNet_PC(
        data_dir=opt.data_dir, 
        category=opt.category, 
        mode='train', 
        num_points=opt.num_points, 
        resamplemode=opt.resamplemode,
        cache_data=opt.cache_data
        )

    val_set = ShapeNet_PC(
        data_dir=opt.data_dir, 
        category=opt.category, 
        mode='val', 
        num_points=opt.num_points, 
        resamplemode=opt.resamplemode,
        cache_data=opt.cache_data
        )

    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=opt.batch_size,
        shuffle=True, 
        pin_memory=False,
        num_workers=opt.num_workers
                )

    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=opt.batch_size,
        shuffle=False, 
        pin_memory=False,
        num_workers=opt.num_workers
        )

    # ----------------------------- Prepare training -----------------------------
    device = torch.device('cuda')

    rot_enc = art_model.PointNetTransformNet().to(device)
    optimizer = optim.AdamW(list(rot_enc.parameters()), lr=opt.lr)

    loss_dict = {}
    loss_dict['chamfer'] = chamfer_distance




    # define our custom x axis metric
    wandb.define_metric("train/epoch")
    wandb.define_metric("val/epoch")
    # set all other train/ metrics to use this step

    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")

    wandb.define_metric("val/total_loss", summary="min", step_metric="val/epoch")
    wandb.define_metric("val/rot_loss_mse", summary="min", step_metric="val/epoch")
    wandb.define_metric("val/rot_loss_chamfer", summary="min", step_metric="val/epoch")
    wandb.define_metric("train/total_loss", summary="min", step_metric="train/epoch")
    wandb.define_metric("train/rot_loss_mse", summary="min", step_metric="train/epoch")
    wandb.define_metric("train/rot_loss_chamfer", summary="min", step_metric="train/epoch")

    # import open3d

    # for i, data in enumerate(train_loader):
    #     pass
    # train_loader.dataset.set_use_cache(True)

    # for i, data in enumerate(train_loader):
    #     sample = data[0].numpy().astype(np.float64)

    #     print(sample)
    #     pcd = open3d.geometry.PointCloud()
    #     pcd.points = open3d.utility.Vector3dVector(sample)
    #     open3d.visualization.draw_geometries([pcd])

    # sys.exit()

    train_single_model(
        model=rot_enc, 
        losses=loss_dict, 
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        opt=opt,
        save_path=save_dir)