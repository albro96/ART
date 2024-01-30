# from psbody.mesh import Mesh
import os
import numpy as np
import torch
import glob
import os.path as op
import sys
from pathlib import Path
from easydict import EasyDict

BASE_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(BASE_DIR))

from dataset import ShapeNet_PC
from models import pointnet, art_model
from loss import chamfer_distance

sys.path.append('/storage/share/code/01_scripts/modules/')
from os_tools.import_dir_path import import_dir_path

pada = import_dir_path()

model_dir = pada.models.art.model_dir
runname = '240130_1000-items-test'
dataset = 'ShapeNet'
checkpoint_path = op.join(model_dir, dataset, runname, '451.pth')

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# gpu_id = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

device = torch.device('cuda:0')

data_dir = '/storage/share/nobackup/data/ShapeNet55-34/shapenet_pc'
category = '02691156'

opt = EasyDict()
opt.category = 'plane'
opt.size = 128
opt.art = True
opt.resume = False
opt.iters = 5
opt.lambda2 = 0.1


test_set = ShapeNet_PC(data_dir=data_dir, category=category, mode=2, num_points=2048)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                          shuffle=False, pin_memory=False,
                                          num_workers=0)


nlat = opt.size

enc_conv_size = [3, 64, 128, 128, 256]
dec_fc_size = [256, 256, 2048*3]
enc = pointnet.PointNet(nlat, enc_conv_size).to(device)
dec = pointnet.FCDecoder(nlat, dec_fc_size).to(device)

rot_enc = art_model.PointNetTransformNet().to(device)

models = {
    'rot_enc': None,
    'enc': enc,
    'dec': dec,
}

checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
for t in models:
    if models[t]:
        models[t].load_state_dict(checkpoint['m_'+t])
        models[t] = models[t].to(device)

if models['rot_enc']:
    models['rot_enc'].eval()

models['enc'].eval()
models['dec'].eval()

vald_loss = 0

for x in test_loader:
    x = x.to(device)
    with torch.no_grad():
        R = models['rot_enc'](x)

        z = models['enc'](x)
        y = models['dec'](z)

        with torch.cuda.device(device):
            recon_loss = chamfer_distance(x, y)

        vald_loss += recon_loss.item() * x.size(0)

print(vald_loss / len(test_loader.dataset))