import torch
# import chamfer
# from kaolin.metrics.point import SidedDistance
from pytorch3d.loss import chamfer_distance as chamfer_distance_pytorch3d

def batch_gather(arr, ind):
    """
    :param arr: B x N x D
    :param ind: B x M
    :return: B x M x D
    """
    dummy = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), arr.size(2))
    out = torch.gather(arr, 1, dummy)
    return out

def chamfer_distance(s1, s2, w1=None, w2=None, norm=2):
    # compute chamfer distance
    loss, _ = chamfer_distance_pytorch3d(s1, s2, norm=norm, batch_reduction='mean', point_reduction='mean')
    return loss


# def chamfer_distance(s1, s2, w1=1., w2=1.):
#     """
#     :param s1: B x N x 3
#     :param s2: B x M x 3
#     :param w1: weight for distance from s1 to s2
#     :param w2: weight for distance from s2 to s1
#     """
#     assert s1.is_cuda and s2.is_cuda
#     sided_minimum_dist = SidedDistance()
#     closest_index_in_s2 = sided_minimum_dist(s1, s2)
#     closest_index_in_s1 = sided_minimum_dist(s2, s1)
#     closest_s2 = batch_gather(s2, closest_index_in_s2)
#     closest_s1 = batch_gather(s1, closest_index_in_s1)
#     dist_to_s2 = (((s1 - closest_s2) ** 2).sum(dim=-1)).mean() * w1
#     dist_to_s1 = (((s2 - closest_s1) ** 2).sum(dim=-1)).mean() * w2
#     return dist_to_s2 + dist_to_s1

# def chamfer_distance(s1, s2, w1=1., w2=1.):
#     """
#     :param s1: B x N x 3
#     :param s2: B x M x 3
#     :param w1: weight for distance from s1 to s2
#     :param w2: weight for distance from s2 to s1
#     """
#     batch_size = s1.size(0)
    
#     if batch_size == 1:
#         non_zeros1 = torch.sum(s1, dim=2).ne(0)
#         non_zeros2 = torch.sum(s2, dim=2).ne(0)
#         s1 = s1[non_zeros1].unsqueeze(dim=0)
#         s2 = s2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(s1, s2)
#         return torch.mean(dist1)*w1 + torch.mean(dist2)*w2

#     dist1, dist2 = ChamferFunction.apply(s1, s2)

#     return torch.mean(dist1)*w1 + torch.mean(dist2)*w2
    

# class ChamferFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):

#         # Ensure the inputs are float32 tensors (if in torch.cuda.amp.autocast context)
#         xyz1 = xyz1.float()
#         xyz2 = xyz2.float()

#         dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
#         ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

#         return dist1, dist2

#     @staticmethod
#     def backward(ctx, grad_dist1, grad_dist2):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
#         grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
#         return grad_xyz1, grad_xyz2
    

# class ChamferDistanceL1(torch.nn.Module):
#     f''' Chamder Distance L1
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         # import pdb
#         # pdb.set_trace()
#         dist1 = torch.sqrt(dist1)
#         dist2 = torch.sqrt(dist2)
#         return (torch.mean(dist1) + torch.mean(dist2))/2

# class ChamferDistanceL2(torch.nn.Module):
#     f''' Chamder Distance L2
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         return torch.mean(dist1) + torch.mean(dist2)


