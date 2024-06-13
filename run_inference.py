import os
import numpy as np
import torch
import sys
from easydict import EasyDict
import json
import os.path as op
import os
import torch.nn as nn
from tqdm import tqdm
from pprint import pprint
import copy

sys.path.append("/storage/share/code/01_scripts/modules/")
from train_fn import get_angle
from os_tools.import_dir_path import import_dir_path
from models import art_model
from tools import builder

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pada = import_dir_path()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = EasyDict()
args = EasyDict(
    {
        "ckpt_dir": op.join(pada.model_base_dir, "ART", "ckpt"),
        "cfg_dir": op.join(pada.model_base_dir, "ART", "config"),
        "save_dir": op.join(
            pada.base_dir, "nobackup", "data", "3DTeethSeg22", "testrot"
        ),
        "val_dataset": "train",
        "load_config_name": "expert-sweep-84",  # -unique-batch-rots",
        "device": torch.device("cuda:0"),
    }
)
args.save_dir = op.join(args.save_dir, args.load_config_name)
os.makedirs(args.save_dir, exist_ok=True)

cfg_path = os.path.join(args.cfg_dir, "config-" + args.load_config_name + ".json")
ckpt_path = os.path.join(args.ckpt_dir, "ckpt-" + args.load_config_name + "-best.pth")

with open(cfg_path, "r") as json_file:
    config = EasyDict(json.load(json_file))

_, data_loader = builder.dataset_builder(
    args, config.dataset, mode=args.val_dataset, bs=1
)

ckpt = torch.load(
    ckpt_path, map_location=args.device
)  # map_location=lambda storage, loc: storage)

model = art_model.PointNetTransformNet(config=config)
model = nn.DataParallel(model).to(args.device)
model.load_state_dict(ckpt["m"])

print(ckpt["torch_rnd"])

model.eval()

angle_dict = EasyDict()
for data in tqdm(data_loader):
    patient = data_loader.dataset.patient
    jaw = data_loader.dataset.jaw
    data = data.to(args.device)
    angle_dict[patient] = {}
    angle_dict[patient][jaw] = {}

    with torch.no_grad():
        R_cum = torch.eye(3).unsqueeze(0).repeat(data.size(0), 1, 1).to(data.device)
        for iter in range(config.model.iters):
            R = model(torch.matmul(data, R_cum).transpose(1, 2).contiguous())
            R_cum = torch.matmul(R_cum, R)
            if iter == 0:
                R_single = copy.deepcopy(R)

        angle_dict[patient][jaw]["R"] = R_single.cpu()
        angle_dict[patient][jaw]["R_angle"] = (
            get_angle(
                R_single,
                torch.eye(3).unsqueeze(0).repeat(data.size(0), 1, 1).to(data.device),
            )
            .cpu()
            .item()
        )

        angle_dict[patient][jaw]["R_cum"] = R_cum.cpu()
        angle_dict[patient][jaw]["R_cum_angle"] = (
            get_angle(
                R_cum,
                torch.eye(3).unsqueeze(0).repeat(data.size(0), 1, 1).to(data.device),
            )
            .cpu()
            .item()
        )

    data_rot_R = torch.matmul(data, R_single)[0].cpu().numpy()
    data_rot_R_cum = torch.matmul(data, R_cum)[0].cpu().numpy()

    np.save(
        op.join(args.save_dir, f"{patient}_{jaw}_rot_R.npy"),
        data_rot_R,
    )
    np.save(
        op.join(args.save_dir, f"{patient}_{jaw}_rot_R_cum.npy"),
        data_rot_R_cum,
    )

    # save GT
    np.save(op.join(args.save_dir, f"{patient}_{jaw}_GT.npy"), data[0].cpu().numpy())

# calc mean, median, std of angles
angles = []
angles_cum = []
for patient in angle_dict.keys():
    for jaw in angle_dict[patient].keys():
        angles.append(angle_dict[patient][jaw]["R_angle"])
        angles_cum.append(angle_dict[patient][jaw]["R_cum_angle"])

print("R", np.mean(angles), np.median(angles), np.std(angles))
print("R_cum", np.mean(angles_cum), np.median(angles_cum), np.std(angles_cum))
