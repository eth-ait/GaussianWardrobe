import torch
import numpy as np
import config
import torch.nn as nn
import cv2 as cv
class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist
    
def to_cuda(items: dict, add_batch = False, precision = torch.float32):
    items_cuda = dict()
    for key, data in items.items():
        if isinstance(data, torch.Tensor):
            items_cuda[key] = data.to(config.device)
        elif isinstance(data, np.ndarray):
            items_cuda[key] = torch.from_numpy(data).to(config.device)
        elif isinstance(data, dict):  # usually some float tensors
            for key2, data2 in data.items():
                if isinstance(data2, np.ndarray):
                    data[key2] = torch.from_numpy(data2).to(config.device)
                elif isinstance(data2, torch.Tensor):
                    data[key2] = data2.to(config.device)
                else:
                    raise TypeError('Do not support other data types.')
                if data[key2].dtype == torch.float32 or data[key2].dtype == torch.float64:
                    data[key2] = data[key2].to(precision)
            items_cuda[key] = data
        else:
            items_cuda[key] = data
        if isinstance(items_cuda[key], torch.Tensor) and\
                (items_cuda[key].dtype == torch.float32 or items_cuda[key].dtype == torch.float64):
            items_cuda[key] = items_cuda[key].to(precision)
        if add_batch:
            if isinstance(items_cuda[key], torch.Tensor):
                items_cuda[key] = items_cuda[key].unsqueeze(0)
            elif isinstance(items_cuda[key], dict):
                for k in items_cuda[key].keys():
                    items_cuda[key][k] = items_cuda[key][k].unsqueeze(0)
            else:
                items_cuda[key] = [items_cuda[key]]
    return items_cuda


def delete_batch_idx(items: dict):
    for k, v in items.items():
        if isinstance(v, torch.Tensor):
            assert v.shape[0] == 1
            items[k] = v[0]
    return items


def generate_volume_points(bounds, testing_res = (256, 256, 256)):
    x_coords = torch.linspace(0, 1, steps = testing_res[0], dtype = torch.float32, device = config.device).detach()
    y_coords = torch.linspace(0, 1, steps = testing_res[1], dtype = torch.float32, device = config.device).detach()
    z_coords = torch.linspace(0, 1, steps = testing_res[2], dtype = torch.float32, device = config.device).detach()
    xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)  # print(xv.shape) # (256, 256, 256)
    xv = torch.reshape(xv, (-1, 1))  # print(xv.shape) # (256*256*256, 1)
    yv = torch.reshape(yv, (-1, 1))
    zv = torch.reshape(zv, (-1, 1))
    pts = torch.cat([xv, yv, zv], dim = -1)

    # transform to canonical space
    if isinstance(bounds, np.ndarray):
        bounds = torch.from_numpy(bounds).to(pts)
    pts = pts * (bounds[1] - bounds[0]) + bounds[0]

    return pts

def read_map_mask(map_path):
    cano_map = cv.imread(map_path, cv.IMREAD_UNCHANGED)
    cano_map = torch.from_numpy(cano_map).to(torch.float32).to(config.device)
    cano_mask = torch.linalg.norm(cano_map, dim = -1) > 0.
    return cano_map, cano_mask

def process_pos_map(map_path):
    smpl_pos_map = cv.imread(map_path, cv.IMREAD_UNCHANGED)
    if smpl_pos_map.shape == (1024, 2048, 3):
        smpl_pos_map = cv.resize(smpl_pos_map, (1024, 512))
    pos_map_size = smpl_pos_map.shape[1] // 2
    smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
    smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
    return smpl_pos_map
