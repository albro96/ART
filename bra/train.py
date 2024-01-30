import torch
import numpy as np
import os, argparse
from torch import optim
import sys
import os.path as op
from pathlib import Path
from easydict import EasyDict 
BASE_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(BASE_DIR))

from dataset import ShapeNet_PC
from loss import chamfer_distance
from models import pointnet, art_model
from train_fn import train_model
import glob
import os.path as op
torch.set_printoptions(precision=6)

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# ----------------------------- Parse arguments --------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('category', default='plane', choices=['plane', 'car', 'chair', 'table', 'sofa', 'multi'],
#                     help='choose training set')
# parser.add_argument('-s', '--size', type=int, default=128, action='store', help='choose latent size')
# parser.add_argument('-a', '--art', action='store_true', help='whether to train with ART')
# parser.add_argument('-r', '--resume', default=False, action='store_true', help='whether to resume training')

opt = EasyDict()
opt.category = 'plane'
opt.size = 128
opt.art = True
opt.resume = False
opt.iters = 5
opt.lambda2 = 0.1

data_dir = '/storage/share/nobackup/data/ShapeNet55-34/shapenet_pc'

category = '02691156'
data_paths = glob.glob(op.join(data_dir, f'{category}*.npy'))

# data = torch.from_numpy(np.load(data_paths[0])).float()
# # expand dim
# data = data.unsqueeze(0)
# print(data.shape)

# opt = parser.parse_args()

# ----------------------------- Prepare data -----------------------------
train_set = ShapeNet_PC(data_dir=data_dir, mode=0, num_points=2048)

vald_set = ShapeNet_PC(data_dir=data_dir, mode=1, num_points=2048)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                           shuffle=True, pin_memory=False,
                                           num_workers=0)

vald_loader = torch.utils.data.DataLoader(vald_set, batch_size=64,
                                           shuffle=False, pin_memory=False,
                                           num_workers=0)

# ----------------------------- Prepare training -----------------------------
device = torch.device('cuda')

if opt.art:
    nlat = opt.size
else:
    nlat = opt.size
enc_conv_size = [3, 64, 128, 128, 256]
dec_fc_size = [256, 256, 2048*3]
enc = pointnet.PointNet(nlat, enc_conv_size).to(device)
dec = pointnet.FCDecoder(nlat, dec_fc_size).to(device)
if opt.art:
    rot_enc = art_model.PointNetTransformNet().to(device)
    optimizer = optim.AdamW(list(enc.parameters())+list(dec.parameters())+list(rot_enc.parameters()), 
                            lr=1e-4)
else:
    optimizer = optim.AdamW(list(enc.parameters())+list(dec.parameters()), lr=1e-4)

model_dict = {
    'rot_enc': rot_enc if opt.art else None,
    'enc': enc,
    'dec': dec,
}

opt_dict = {
    'opt': optimizer,
}

loss_dict = {}
loss_dict['chamfer'] = chamfer_distance

save_path='experiments_art/{}_{}_{}/'.format(opt.category, opt.size, 'art' if opt.art else 'baseline')
if not os.path.isdir(save_path):
    os.makedirs(save_path)

train_model(model_dict, loss_dict, opt_dict,
            train_loader=train_loader,
            vald_loader=vald_loader,
            device=device,
            opt=opt,
            save_path=save_path)