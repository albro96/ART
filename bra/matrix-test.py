import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

mat1 = torch.tensor([[1,2],[3,4]]).unsqueeze(0)
mat2 = torch.tensor([[2,3],[4,5]]).unsqueeze(0)
mat3 = torch.tensor([[6,7],[8,9]]).unsqueeze(0)
mat4 = torch.tensor([[10,11],[12,13]]).unsqueeze(0)

# concat
mat12 = torch.cat((mat1, mat2), dim=0)
mat34 = torch.cat((mat3, mat4), dim=0)

matmul = torch.matmul(mat12, mat34)

print(torch.matmul(mat1, mat3))
print(torch.matmul(mat2, mat4))
print(matmul)





