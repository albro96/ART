# from psbody.mesh import Mesh
import os
import numpy as np
import torch
import glob
import os.path as op
import sys
from pathlib import Path
from easydict import EasyDict
import timeit
import open3d as o3d

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

test_set = ShapeNet_PC(data_dir=data_dir, category=category, mode=2, num_points=2048, resamplemode='fps')


# test_set = ShapeNet_PC(data_dir=data_dir, category=category, mode=2, num_points=8192, resamplemode='none')
# obj0 = test_set.__getitem__(10)

# # Initialize lists to store the results
# random_cd_values = []
# fps_cd_values = []

# for resamplemode in ['random', 'fps']:
#     num_points_values = []
#     for num_points_exp in range(9, 14):

#         num_points = 2**num_points_exp
#         test_set = ShapeNet_PC(data_dir=data_dir, category=category, mode=2, num_points=num_points, resamplemode=resamplemode)
#         obj1 = test_set.__getitem__(10)
#         cd = chamfer_distance(obj1, obj0)

#         # convert cd to numpy float
#         cd = cd.cpu().numpy().astype(np.float64)

#         # Store the results
#         if resamplemode == 'random':
#             random_cd_values.append(cd)
#         else:
#             fps_cd_values.append(cd)

#         num_points_values.append(num_points)

# # save values as npz file
# np.savez('/storage/share/nobackup/temp/cd_values.npz', random_cd_values=random_cd_values, fps_cd_values=fps_cd_values, num_points_values=num_points_values)

# # Wrap the code you want to time in a function
# def get_item():
#     return test_set.__getitem__(3)

# # Time the function using timeit
# execution_time = timeit.timeit(get_item, number=1)

# print(f"Execution time: {execution_time} seconds")

# print(
#     test_set.__getitem__(3)[0].shape
# )


test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False, pin_memory=False,
                                          num_workers=0)


nlat = opt.size

enc_conv_size = [3, 64, 128, 128, 256]
dec_fc_size = [256, 256, 2048*3]
enc = pointnet.PointNet(nlat, enc_conv_size).to(device)
dec = pointnet.FCDecoder(nlat, dec_fc_size).to(device)

rot_enc = art_model.PointNetTransformNet().to(device)

models = {
    'rot_enc': rot_enc,
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

# for x in test_loader:
#     print(x.shape)
#     x = x.to(device)
#     with torch.no_grad():
#         R = models['rot_enc'](x.transpose(1, 2))
#         print(R)

#         # z = models['enc'](x)
#         # y = models['dec'](z)

#         # with torch.cuda.device(device):
#         #     recon_loss = chamfer_distance(x, y)

#         # vald_loss += recon_loss.item() * x.size(0)
#     break

for x in test_loader:
    x = x.to(device)
    # rotate x by 30deg around z-axis
    R_30deg = torch.tensor([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]]).to(device)
    x_rot = torch.matmul(x[0], R_30deg).unsqueeze(0)

    with torch.no_grad():
        R = models['rot_enc'](x.transpose(1, 2))
        R2 = models['rot_enc'](x_rot.transpose(1, 2))

        print(R.shape)
        z = models['enc'](x)
        y = models['dec'](z)

    break

    # z = models['enc'](x)
    # y = models['dec'](z)

encoder_pcd = o3d.geometry.PointCloud()
encoder_pcd.points = o3d.utility.Vector3dVector(y[0].cpu().numpy())
# color red
encoder_pcd.paint_uniform_color([1, 0, 0])

pcd_rand_rot = o3d.geometry.PointCloud()
pcd_rand_rot.points = o3d.utility.Vector3dVector(x_rot[0].cpu().numpy())
# color green
pcd_rand_rot.paint_uniform_color([0, 1, 0])

pcd_norot = o3d.geometry.PointCloud()
pcd_norot.points = o3d.utility.Vector3dVector(x[0].cpu().numpy())
# color blue
pcd_norot.paint_uniform_color([0, 0, 1])


pcd_rot = o3d.geometry.PointCloud()
pcd_rot.points = o3d.utility.Vector3dVector(x[0].cpu().numpy())
# rotate pcd_rot by R
pcd_rot.rotate(R[0].cpu().numpy())
# color red
pcd_rot.paint_uniform_color([1, 0, 0])

pcd_rot2 = o3d.geometry.PointCloud()
pcd_rot2.points = o3d.utility.Vector3dVector(x_rot[0].cpu().numpy())
# rotate pcd_rot by R
pcd_rot2.rotate(R2[0].cpu().numpy())
# color red
pcd_rot2.paint_uniform_color([1, 0, 0])


o3d.visualization.draw_geometries([encoder_pcd])


# print(vald_loss / len(test_loader.dataset))