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


BASE_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(BASE_DIR))

from dataset import ShapeNet_PC
from loss import chamfer_distance
from models import pointnet, art_model
from train_fn import train_model
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

opt = EasyDict()
opt.size = 128
opt.art = True
opt.resume = False
opt.iters = 5
opt.lambda2 = 0.1
batch_size = 64

data_dir = '/storage/share/nobackup/data/ShapeNet55-34/shapenet_pc'
# opt.category = '02691156'

# get all categories from os.listdir(data_dir) as path.split('-')[0] and make unique list
files = os.listdir(data_dir)

category_counts = Counter(f.split('-')[0] for f in files)

categories = [category for category, _ in category_counts.most_common(5)]

# print number of files in each category
for category in categories:
    print(f'{category}: {category_counts[category]}')

for category in categories:
    opt.category = category

    dataset = 'ShapeNet'
    # runname with date in format YYMMDD from datetime function

    runname = f'{datetime.now().strftime("%y%m%d")}_category-{opt.category}'

    save_dir = op.join(model_dir, dataset, runname)

    # ----------------------------- Prepare data -----------------------------
    train_set = ShapeNet_PC(data_dir=data_dir,category=opt.category, mode=0, num_points=2048, resamplemode='fps')

    vald_set = ShapeNet_PC(data_dir=data_dir,category=opt.category, mode=1, num_points=2048, resamplemode='fps')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, pin_memory=False,
                                            num_workers=0)

    vald_loader = torch.utils.data.DataLoader(vald_set, batch_size=batch_size,
                                            shuffle=False, pin_memory=False,
                                            num_workers=0)

    # ----------------------------- Prepare training -----------------------------
    device = torch.device('cuda')

    rot_enc = art_model.PointNetTransformNet().to(device)
    optimizer = optim.AdamW(list(rot_enc.parameters()), lr=1e-4)


    model_dict = {
        'rot_enc': rot_enc if opt.art else None,
    }

    opt_dict = {
        'opt': optimizer,
    }

    loss_dict = {}
    loss_dict['chamfer'] = chamfer_distance


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_model(model_dict, loss_dict, opt_dict,
                train_loader=train_loader,
                vald_loader=vald_loader,
                device=device,
                opt=opt,
                save_path=save_dir)