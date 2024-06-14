from scipy.spatial import KDTree
import trimesh
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_gather, knn_points, ball_query
import torch
import os.path as op
import time

sys.path.append(
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"
        )
    )
)
from os_tools.import_dir_path import import_dir_path, convert_path

pada = import_dir_path()
lower_path = convert_path(r"O:\data\scan\raw\studentenkurs\230109\baulea_lowerjaw.stl")
upper_path = convert_path(r"O:\data\scan\raw\studentenkurs\230109\baulea_upperjaw.stl")


# save lower_pcd and upper_pcd to disk
save_dir = op.join(
    pada.base_dir, "nobackup", "temp", op.basename(__file__).split(".")[0]
)
os.makedirs(save_dir, exist_ok=True)

if os.path.exists(op.join(save_dir, "lower_pcd.pth")) and os.path.exists(
    op.join(save_dir, "upper_pcd.pth")
):
    lower_pcd = torch.load(op.join(save_dir, "lower_pcd.pth")).to("cuda")
    upper_pcd = torch.load(op.join(save_dir, "upper_pcd.pth")).to("cuda")
    print("Loaded pointclouds from disk")
else:
    print("Creating pointclouds from meshes")
    # load meshes with trimesh
    lower_mesh = trimesh.load(lower_path)
    upper_mesh = trimesh.load(upper_path)

    # Convert trimesh faces to PyTorch tensors
    lower_faces_tensor = torch.tensor(lower_mesh.faces, dtype=torch.int64)
    upper_faces_tensor = torch.tensor(upper_mesh.faces, dtype=torch.int64)

    # Now use these tensors in the Meshes constructor
    lower_mesh = Meshes(
        verts=[torch.tensor(lower_mesh.vertices, dtype=torch.float32)],
        faces=[lower_faces_tensor],
        verts_normals=[torch.tensor(lower_mesh.vertex_normals, dtype=torch.float32)],
    )
    upper_mesh = Meshes(
        verts=[torch.tensor(upper_mesh.vertices, dtype=torch.float32)],
        faces=[upper_faces_tensor],
        verts_normals=[torch.tensor(upper_mesh.vertex_normals, dtype=torch.float32)],
    )

    def create_pointcloud_from_mesh(mesh, num_samples=32768, return_normals=True):

        samples, normals = sample_points_from_meshes(
            mesh, num_samples=num_samples, return_normals=return_normals
        )
        return Pointclouds(points=samples, normals=normals)

    lower_pcd = create_pointcloud_from_mesh(lower_mesh, num_samples=30000).to("cuda")
    upper_pcd = create_pointcloud_from_mesh(upper_mesh, num_samples=30000).to("cuda")

    torch.save(lower_pcd, op.join(save_dir, "lower_pcd.pth"))
    torch.save(upper_pcd, op.join(save_dir, "upper_pcd.pth"))

    # save as numpy arrays with normals
    lower = lower_pcd.points_packed().cpu().numpy()
    lower_normals = lower_pcd.normals_packed().cpu().numpy()
    upper = upper_pcd.points_packed().cpu().numpy()
    upper_normals = upper_pcd.normals_packed().cpu().numpy()

    np.savez(
        op.join(save_dir, "both_pcd.npz"),
        lower=lower,
        lower_normals=lower_normals,
        upper=upper,
        upper_normals=upper_normals,
    )

t0 = time.time()
tree = KDTree(upper_pcd.points_packed().cpu().numpy())
d_kdtree, idx = tree.query(
    lower_pcd.points_packed().cpu().numpy(), p=2, k=1, workers=-1, eps=0
)
print(time.time() - t0)


t0 = time.time()
dists, upper_idx, upper_nn = knn_points(
    lower_pcd.points_packed().unsqueeze(0),
    upper_pcd.points_packed().unsqueeze(0),
    K=1,
    return_nn=True,
    return_sorted=False,
    norm=2,
)
dists = dists.sqrt()[0, :, 0]

vectors = lower_pcd.points_packed() - upper_nn[0, :, 0, :]

dot_product = torch.einsum(
    "ij,ij->i", vectors, upper_pcd.normals_packed()[upper_idx[0, :, 0]]
)

dists = torch.where(dot_product < 0, -dists, dists)
print(time.time() - t0)

print(dists.shape)
print(torch.sum(dists < 0.1))
print(torch.sort(dists.cpu())[0])
