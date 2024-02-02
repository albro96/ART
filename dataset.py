import torch
import numpy as np
import torch.utils.data as data
import glob
import os.path as op
import pytorch3d.ops.sample_farthest_points as fps 
import sys

import ctypes
import multiprocessing as mp

class ShapeNet_PC(data.Dataset):
    def __init__(
            self, 
            data_dir, 
            category, 
            num_points=2048, 
            mode='train', 
            splits=  {
                    'train': 0.85,
                    'val': 0.1,
                    'test': 0.05
                    }, 
            resamplemode='fps',
            cache_data=False):
        
        """
        mode: 0 for training, 1 for validation, 2 for testing
        """

        data_paths = glob.glob(op.join(data_dir, f'{category}*.npy')) #[:500]

        self.cache_data = cache_data
        self.mode = mode
        self.num_samples= int(len(data_paths) * splits[mode])

        self.num_points = num_points

        if self.cache_data:
            shared_array_base = mp.Array(ctypes.c_float, self.num_samples*self.num_points*3)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            shared_array = shared_array.reshape(self.num_samples, self.num_points, 3)
            self.shared_array = torch.from_numpy(shared_array)
        
        self.use_cache = False

        self.resamplemode = resamplemode
        
        if mode == 'train':
            self.data_paths = data_paths[:int(len(data_paths)*splits['train'])]
        elif mode == 'val':
            lower_bound = int(len(data_paths)*splits['train'])
            upper_bound = int(len(data_paths)*splits['train']) + int(len(data_paths)*splits['val'])
            self.data_paths = data_paths[lower_bound:upper_bound]
        elif mode == 'test':
            lower_bound = int(len(data_paths)*splits['train']) + int(len(data_paths)*splits['val'])
            self.data_paths = data_paths[lower_bound:]

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

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

        if self.cache_data:
            if not self.use_cache:
                # print(f'\r[{self.mode}] Filling cache...', end='')

                # Add your loading logic here
                sample = self.resample(torch.from_numpy(np.load(self.data_paths[index])).float().unsqueeze(0))
                self.shared_array[index] = sample
            return self.shared_array[index]
        
        else:
            return self.resample(torch.from_numpy(np.load(self.data_paths[index])).float().unsqueeze(0))[0]
        
    def __len__(self):
        return self.num_samples
    


# class ShapeNet_PC(data.Dataset):
#     def __init__(self, data_dir, category, num_points=2048, mode=0, splits=(0.85, 0.05, 0.1), resamplemode='fps'):

#         data_paths = glob.glob(op.join(data_dir, f'{category}*.npy'))

#         num_samples_train = int(len(data_paths) * splits[0])
#         num_samples_vald =int(len(data_paths) * (splits[0]+splits[1]))
        
#         self.num_points = num_points
#         self.resamplemode = resamplemode
        
#         if mode == 0:
#             self.data = data_paths[:num_samples_train]
#         elif mode == 1:
#             self.data = data_paths[num_samples_train: num_samples_vald]
#         elif mode == 2:
#             self.data = data_paths[num_samples_vald:]

#     def resample(self, obj, resamplemode=None):
#         if resamplemode is None:
#             resamplemode = self.resamplemode

#         if resamplemode == 'random':
#             if self.num_points < obj.shape[1]:
#                 ind = torch.from_numpy(np.random.choice(obj.shape[1], self.num_points)).long()
#                 return obj[:, ind]
#             else:
#                 return obj
            
#         elif resamplemode == 'fps':
#             return fps(obj, K=self.num_points)[0] 
        
#         elif resamplemode == 'none':
#             return obj

#     def __getitem__(self, index):
#         return self.resample(torch.from_numpy(np.load(self.data[index])).float().unsqueeze(0))[0]

#     def __len__(self):
#         return len(self.data)
    

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
