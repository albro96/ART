import torch
import numpy as np
import torch.utils.data as data
import glob
import os.path as op
import pytorch3d.ops.sample_farthest_points as fps 


class ShapeNet_PC(data.Dataset):
    def __init__(self, data_dir, category, num_points=2048, mode=0, splits=(0.85, 0.05, 0.1), resamplemode='fps'):

        data_paths = glob.glob(op.join(data_dir, f'{category}*.npy'))

        num_samples_train = int(len(data_paths) * splits[0])
        num_samples_vald =int(len(data_paths) * (splits[0]+splits[1]))
        
        self.num_points = num_points
        self.resamplemode = resamplemode
        
        if mode == 0:
            self.data = data_paths[:num_samples_train]
        elif mode == 1:
            self.data = data_paths[num_samples_train: num_samples_vald]
        elif mode == 2:
            self.data = data_paths[num_samples_vald:]

    def resample(self, obj, resamplemode=None):
        if resamplemode is None:
            resamplemode = self.resamplemode

        if resamplemode == 'random':
            if self.num_points < obj.shape[1]:
                ind = torch.from_numpy(np.random.choice(obj.shape[1], self.num_points)).long()
                return obj[:, ind]
            else:
                return obj
            
        elif resamplemode == 'fps':
            return fps(obj, K=self.num_points)[0] 
        
        elif resamplemode == 'none':
            return obj

    def __getitem__(self, index):
        return self.resample(torch.from_numpy(np.load(self.data[index])).float().unsqueeze(0))[0]

    def __len__(self):
        return len(self.data)
    

# class ShapeNet_PC(data.Dataset):
#     def __init__(self, path=None, paths=None, num_points=2048, mode=0, splits=(0.85, 0.05, 0.1)):
#         if paths is not None:
#             data = []
#             for path in paths:
#                 data.append(np.load(path))
#             data = np.concatenate(data, axis=0)
#         else:
#             data = np.load(path)

#         num_samples_train = int(data.shape[0] * splits[0])
#         num_samples_vald = int(data.shape[0] * (splits[0]+splits[1]))

#         self.num_points = num_points
#         if mode == 0:
#             self.data = data[:num_samples_train]
#         elif mode == 1:
#             self.data = data[num_samples_train: num_samples_vald]
#         elif mode == 2:
#             self.data = data[num_samples_vald:]
#         self.data = torch.from_numpy(self.data).float()
#         self.resample()

#     def resample(self):
#         if self.num_points < self.data.shape[1]:
#             ind = torch.from_numpy(np.random.choice(self.data.shape[1], self.num_points)).long()
#             self.pc = self.data[:, ind]
#         else:
#             self.pc = self.data

#     def __getitem__(self, index):
#         return self.pc[index]

#     def __len__(self):
#         return self.data.size(0)
