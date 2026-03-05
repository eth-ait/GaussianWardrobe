import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import cv2 as cv
from sympy import rotations

import config
from network.avatar import AvatarNet
from network.styleunet.dual_styleunet import DualStyleUNet
from gaussians.gaussian_model import GaussianModel
from gaussians.gaussian_renderer import render3
from utils.net_util import read_map_mask
from utils.general_utils import rectify_depth, merge_gaussian_val, masked_mean_filter, filter_gaussian, render_gaussian
from pytorch3d.ops import knn_points

class CombinedAvatarNet(nn.Module):
    def __init__(self, opt, body_net, upper_net, lower_net):
        super(CombinedAvatarNet, self).__init__()
        self.body_layer = body_net
        self.upper_layer = upper_net
        self.lower_layer = lower_net

        self.layers = ["body", "cloth"]
        self.data_dir = opt["data_dir"]
        # lower mask
        # _, self.cloth_cloth_mask = read_map_mask(self.data_dir + '/cloth_cloth.exr')
        # upper mask

        # full cloth mask
        # self.cloth_mask = self.upper_layer.layers_nn["cloth"].cano_smpl_mask & (~self.cloth_cloth_mask)

        # separate cloth mask
        self.upper_cloth_mask = self.upper_layer.layers_nn["cloth"].cano_smpl_mask 
        self.lower_cloth_mask = self.lower_layer.layers_nn["cloth"].cano_smpl_mask 

        _, self.smplx_mask = read_map_mask(self.data_dir + '/cano_smpl_smplx_lower_map.exr')
        _, self.smplx_lower_mask = read_map_mask(self.data_dir + '/cano_smpl_smplx_lower_map.exr')
        _, self.smplx_upper_mask = read_map_mask(self.data_dir + '/cano_smpl_smplx_upper_map.exr')
        _, self.half_cloth_mask = read_map_mask(self.data_dir + '/cano_smpl_smplx_map.exr')
        self.upper_cloth_mask = self.smplx_upper_mask 
        self.lower_cloth_mask = self.smplx_lower_mask 
        # self.cloth_mask = self.half_cloth_mask 
        self.cloth_mask = self.smplx_upper_mask 


        _, self.face_mask = read_map_mask(self.data_dir + '/smplx_segment.exr')
        # self.body_layer.layers_nn["body"].selected_gaussian = ~(self.combined_cloth_mask[self.body_layer.layers_nn["body"].cano_smpl_mask])

    def smooth_boundary(self, update_mask, gaussian_body_vals, cano_mask, ori_gaussian_body_pos, kernel_size = 20, K=100):
        update_mask_np = update_mask.detach().cpu().numpy().astype('uint8')
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_erode = cv.erode(update_mask_np.copy(), kernel)
        mask_dilate = cv.dilate(update_mask_np.copy(), kernel)
        boundary_mask = (mask_dilate - mask_erode) == 1
        
        boundary_mask = torch.tensor(boundary_mask).to(config.device) 
        cv.imwrite('boundary_mask.png', boundary_mask.cpu().numpy().astype("uint8")*255)
        boundary_mask = boundary_mask[self.body_layer.layers_nn["body"].cano_smpl_mask]
        bounday_pos = gaussian_body_vals["positions"][boundary_mask]
        query_pos =  torch.concat([gaussian_body_vals["positions"]]) 
        knn_result = knn_points(bounday_pos.unsqueeze(0), query_pos.unsqueeze(0), K=K, return_nn=True)
        dists = torch.sqrt(knn_result.dists.squeeze(0))
        weights = -torch.log(dists + 1e-5)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).unsqueeze(-1).expand(-1, -1, 3)
        
        new_pos = weights * knn_result.knn.squeeze(0)
        new_pos = torch.sum(new_pos, dim=1)
        gaussian_body_vals["positions"][boundary_mask] = new_pos
        # new_offset_map = masked_mean_filter(gaussian_body_vals["pos_map"], cano_mask, kernel_size=kernel_size)
        # new_offset_map = torch.tensor(new_offset_map).to(config.device)
        # gaussian_body_vals["pos_map"] = new_offset_map

    def render(self, items_body, items_upper, items_lower, bg_color = (0., 0., 0.), layers=None, cross_change=True):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items_body["smpl_pos_map"][layer] = items_body["smpl_pos_map"][layer].squeeze(0)
            items_upper["smpl_pos_map"][layer] = items_upper["smpl_pos_map"][layer].squeeze(0)
            items_lower["smpl_pos_map"][layer] = items_lower["smpl_pos_map"][layer].squeeze(0)
        
        gaussian_body_vals = self.body_layer.render(items_body, layers=["body"], only_gaussian=True)
        
        gaussian_upper_cloth_vals = self.upper_layer.render(items_upper, layers=["upper"], only_gaussian=True)
        gaussian_lower_cloth_vals = self.lower_layer.render(items_lower, layers=["lower"], only_gaussian=True)
        gaussian_upper_cloth_body_vals = self.upper_layer.render(items_upper, layers=["body"], only_gaussian=True)
        gaussian_lower_cloth_body_vals = self.lower_layer.render(items_lower, layers=["body"], only_gaussian=True)

        # add label color to cloth gaussians
        gaussian_lower_cloth_vals["label_colors"]  = torch.zeros_like(gaussian_lower_cloth_vals["colors"]).to(config.device)
        gaussian_lower_cloth_vals["label_colors"][:, 0] = 1
        gaussian_upper_cloth_vals["label_colors"] = torch.zeros_like(gaussian_upper_cloth_vals["colors"]).to(config.device)
        gaussian_upper_cloth_vals["label_colors"][:, 1] = 1

        # update offset for upper cloth
        update_offset_mask_upper = self.upper_cloth_mask[self.upper_layer.layers_nn["body"].cano_smpl_mask] 

        update_offset_mask_body = self.upper_cloth_mask[self.body_layer.layers_nn["body"].cano_smpl_mask]
        gaussian_body_vals = self.body_layer.layers_nn["body"].update_selected_pos(gaussian_body_vals, gaussian_upper_cloth_body_vals, items_body, update_offset_mask_upper, update_offset_mask_body)

        # update offset for lower cloth
        update_offset_mask_lower = self.lower_cloth_mask[self.lower_layer.layers_nn["body"].cano_smpl_mask] 

        update_offset_mask_body = self.lower_cloth_mask[self.body_layer.layers_nn["body"].cano_smpl_mask]
        gaussian_body_vals = self.body_layer.layers_nn["body"].update_selected_pos(gaussian_body_vals, gaussian_lower_cloth_body_vals, items_body, update_offset_mask_lower, update_offset_mask_body)


        # self.smooth_boundary(self.cloth_cloth_mask, gaussian_body_vals, ori_gaussian_body_pos)
        gaussian_vals = {}

        gaussian_combined_cloth_vals = merge_gaussian_val(gaussian_upper_cloth_vals, gaussian_lower_cloth_vals)

        gaussian_combined_cloth_vals["label_colors"]  = torch.zeros_like(gaussian_combined_cloth_vals["colors"]).to(config.device)
        gaussian_combined_cloth_vals["label_colors"][:, 1] = 1

        gaussian_body_vals["label_colors"]  = torch.zeros_like(gaussian_body_vals["colors"]).to(config.device)
        gaussian_body_vals["label_colors"][:, 2] = 1

        gaussian_vals = merge_gaussian_val(gaussian_body_vals, gaussian_combined_cloth_vals)


        render_ret = render_gaussian(gaussian_vals, items_body, bg_color)
        label_rgb_map = render_ret['render'][:3, :, :]
        rgb_map = render_ret['render'][3:, :, :]

        # render body only

        render_body_ret = render_gaussian(gaussian_body_vals, items_body, bg_color)
        label_body_rgb_map = render_body_ret['render'][:3, :, :]
        body_rgb_map = render_body_ret['render'][3:, :, :]

        # render cloth only
        
        gaussian_combined_cloth_vals["colors"] = torch.concat([gaussian_combined_cloth_vals["label_colors"], gaussian_combined_cloth_vals["colors"]], dim=1)
        render_cloth_ret = render3(
            gaussian_combined_cloth_vals,
            bg_color.repeat(2),
            items_body['extr'],
            items_body['intr'],
            items_body['img_w'],
            items_body['img_h']
        )
        label_cloth_rgb_map = render_cloth_ret['render'][:3, :, :]
        cloth_rgb_map = render_cloth_ret['render'][3:, :, :]
        # filter upper and lower
        gaussian_upper_cloth_vals["colors"] = torch.concat([gaussian_upper_cloth_vals["label_colors"], gaussian_upper_cloth_vals["colors"]], dim=1)
        render_upper_cloth_ret = render3(
        gaussian_upper_cloth_vals,
        bg_color.repeat(2),
        items_body['extr'],
        items_body['intr'],
        items_body['img_w'],
        items_body['img_h']
        )
        label_upper_rgb_map = render_upper_cloth_ret['render'][:3, :, :]
        upper_rgb_map = render_upper_cloth_ret['render'][3:, :, :]
        gaussian_lower_cloth_vals["colors"] = torch.concat([gaussian_lower_cloth_vals["label_colors"], gaussian_lower_cloth_vals["colors"]], dim=1)
        render_lower_cloth_ret = render3(
        gaussian_lower_cloth_vals,
        bg_color.repeat(2),
        items_body['extr'],
        items_body['intr'],
        items_body['img_w'],
        items_body['img_h']
        )
        label_lower_rgb_map = render_lower_cloth_ret['render'][:3, :, :]
        lower_rgb_map = render_lower_cloth_ret['render'][3:, :, :]
        if layers == ["body"]:
            ret = {
                'rgb_map':  body_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_body_rgb_map.permute(1, 2, 0),
            }
            return ret
        elif layers == ["both"]:
            ret = {
                'rgb_map': rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_rgb_map.permute(1, 2, 0),
            }
            return ret
        elif layers == ["cloth"]:
            ret = {
                'rgb_map': cloth_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_cloth_rgb_map.permute(1, 2, 0),
            }
            return ret
        elif layers == ["upper"]:
            ret = {
                'rgb_map': upper_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_upper_rgb_map.permute(1, 2, 0),
            }
            return ret
        elif layers == ["lower"]:
            ret = {
                'rgb_map': lower_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_lower_rgb_map.permute(1, 2, 0),
            }
            return ret
        
    def render_filtered(self, items_body, items_upper, items_lower, bg_color = (0., 0., 0.), layers=None):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items_body["smpl_pos_map"][layer] = items_body["smpl_pos_map"][layer].squeeze(0)
            items_upper["smpl_pos_map"][layer] = items_upper["smpl_pos_map"][layer].squeeze(0)
            items_lower["smpl_pos_map"][layer] = items_lower["smpl_pos_map"][layer].squeeze(0)
        
        gaussian_body_vals = self.body_layer.render(items_body, layers=["body"], only_gaussian=True)
        
        gaussian_upper_cloth_vals = self.upper_layer.render(items_upper, layers=["upper"], only_gaussian=True)
        gaussian_lower_cloth_vals = self.lower_layer.render(items_lower, layers=["lower"], only_gaussian=True)
        gaussian_upper_cloth_body_vals = self.upper_layer.render(items_upper, layers=["body"], only_gaussian=True)
        gaussian_lower_cloth_body_vals = self.lower_layer.render(items_lower, layers=["body"], only_gaussian=True)

        # add label color to cloth gaussians
        gaussian_lower_cloth_vals["label_colors"]  = torch.zeros_like(gaussian_lower_cloth_vals["colors"]).to(config.device)
        gaussian_lower_cloth_vals["label_colors"][:, 0] = 1
        gaussian_upper_cloth_vals["label_colors"] = torch.zeros_like(gaussian_upper_cloth_vals["colors"]).to(config.device)
        gaussian_upper_cloth_vals["label_colors"][:, 1] = 1

        # update offset for upper cloth
        update_offset_mask_upper = self.cloth_mask[self.upper_layer.layers_nn["body"].cano_smpl_mask] 

        update_offset_mask_body = self.cloth_mask[self.body_layer.layers_nn["body"].cano_smpl_mask]
        gaussian_body_vals = self.body_layer.layers_nn["body"].update_selected_pos(gaussian_body_vals, gaussian_upper_cloth_body_vals, items_body, update_offset_mask_upper, update_offset_mask_body)

        # update offset for lower cloth
        update_offset_mask_lower = self.lower_cloth_mask[self.lower_layer.layers_nn["body"].cano_smpl_mask] 

        update_offset_mask_body = self.lower_cloth_mask[self.body_layer.layers_nn["body"].cano_smpl_mask]
        gaussian_body_vals = self.body_layer.layers_nn["body"].update_selected_pos(gaussian_body_vals, gaussian_lower_cloth_body_vals, items_body, update_offset_mask_lower, update_offset_mask_body)


        # self.smooth_boundary(self.cloth_cloth_mask, gaussian_body_vals, ori_gaussian_body_pos)
        gaussian_vals = {}

        gaussian_combined_cloth_vals = merge_gaussian_val(gaussian_upper_cloth_vals, gaussian_lower_cloth_vals)


        gaussian_body_vals["label_colors"]  = torch.zeros_like(gaussian_body_vals["colors"]).to(config.device)
        gaussian_body_vals["label_colors"][:, 2] = 1

        gaussian_vals = merge_gaussian_val(gaussian_body_vals, gaussian_combined_cloth_vals)


        render_ret = render_gaussian(gaussian_vals, items_body, bg_color)
        label_rgb_map = render_ret['render'][:3, :, :]
        rgb_map = render_ret['render'][3:, :, :]

        # render body only

        render_body_ret = render_gaussian(gaussian_body_vals, items_body, bg_color)
        label_body_rgb_map = render_body_ret['render'][:3, :, :]
        body_rgb_map = render_body_ret['render'][3:, :, :]

        # render cloth only
        
        gaussian_combined_cloth_vals["colors"] = torch.concat([gaussian_combined_cloth_vals["label_colors"], gaussian_combined_cloth_vals["colors"]], dim=1)
        render_cloth_ret = render3(
            gaussian_combined_cloth_vals,
            bg_color.repeat(2),
            items_body['extr'],
            items_body['intr'],
            items_body['img_w'],
            items_body['img_h']
        )
        label_cloth_rgb_map = render_cloth_ret['render'][:3, :, :]
        cloth_rgb_map = render_cloth_ret['render'][3:, :, :]
        # filter upper and lower
        gaussian_upper_cloth_vals["colors"] = torch.concat([gaussian_upper_cloth_vals["label_colors"], gaussian_upper_cloth_vals["colors"]], dim=1)
        render_upper_cloth_ret = render3(
        gaussian_upper_cloth_vals,
        bg_color.repeat(2),
        items_body['extr'],
        items_body['intr'],
        items_body['img_w'],
        items_body['img_h']
        )
        label_upper_rgb_map = render_upper_cloth_ret['render'][:3, :, :]
        gaussian_lower_cloth_vals["colors"] = torch.concat([gaussian_lower_cloth_vals["label_colors"], gaussian_lower_cloth_vals["colors"]], dim=1)
        render_lower_cloth_ret = render3(
        gaussian_lower_cloth_vals,
        bg_color.repeat(2),
        items_body['extr'],
        items_body['intr'],
        items_body['img_w'],
        items_body['img_h']
        )
        cloth_rgb_map, label_cloth_rgb_map, _ = filter_gaussian(render_upper_cloth_ret, render_lower_cloth_ret, label_upper_rgb_map, cloth_rgb_map, render_cloth_ret["mask"], 1, buffer=0.1)
        render_cloth_ret['render'] = torch.concat([label_cloth_rgb_map, cloth_rgb_map], dim=0)
        
        if layers == ["body"]:
            ret = {
                'rgb_map':  body_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_body_rgb_map.permute(1, 2, 0),
            }
            return ret
        elif layers == ["both"]:
            label_rgb_map[1, :, :] += label_rgb_map[0, :, :]
            filtered_rgb_map, visibility_mask, depth_diff = filter_gaussian(render_cloth_ret, render_body_ret, label_rgb_map, rgb_map, render_ret["mask"], 1)
            # filtered_rgb_map, visibility_mask, depth_diff = filter_gaussian(render_body_ret, render_ret, visibility_mask, filtered_rgb_map, render_ret["mask"], 2)
            # filtered_rgb_map = rgb_map
            # visibility_mask = label_rgb_map
            ret = {
                'rgb_map': filtered_rgb_map.permute(1, 2, 0),
                'body_visible_mask': visibility_mask.permute(1, 2, 0),
                "label_rgb_map": label_rgb_map.permute(1, 2, 0),
            }
            return ret
        elif layers == ["cloth"]:
            ret = {
                'rgb_map': cloth_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_cloth_rgb_map.permute(1, 2, 0),
            }
            return ret
        elif layers == ["upper"]:
            ret = {
                'rgb_map': render_upper_cloth_ret["render"][3:, :, :].permute(1, 2, 0),
                "label_rgb_map": render_upper_cloth_ret["render"][:3, :, :].permute(1, 2, 0),
            }
            return ret


    def render_full_filter(self, items_body, items_cloth, bg_color = (0., 0., 0.), layers=None):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items_body["smpl_pos_map"][layer] = items_body["smpl_pos_map"][layer].squeeze(0)
            items_cloth["smpl_pos_map"][layer] = items_cloth["smpl_pos_map"][layer].squeeze(0)
    
        gaussian_body_vals = self.body_layer.render(items_body, layers=["body"], only_gaussian=True)
        gaussian_cloth_vals = self.upper_layer.render(items_cloth, layers=["cloth"], only_gaussian=True)
        gaussian_cloth_body_vals = self.upper_layer.render(items_cloth, layers=["body"], only_gaussian=True)


        update_offset_mask_cloth = self.cloth_mask[self.upper_layer.layers_nn["body"].cano_smpl_mask]
        # offset_to_update =  gaussian_cloth_body_vals["offset"][update_offset_mask_cloth]

        update_offset_mask_body = self.cloth_mask[self.body_layer.layers_nn["body"].cano_smpl_mask] 
        ori_gaussian_body_pos = gaussian_body_vals["positions"]
        gaussian_body_vals = self.body_layer.layers_nn["body"].update_selected_pos(gaussian_body_vals, gaussian_cloth_body_vals, items_body, update_offset_mask_cloth, update_offset_mask_body)
        # self.smooth_boundary(self.cloth_mask, gaussian_body_vals, self.cloth_mask, ori_gaussian_body_pos)
        # gaussian_body_vals = self.body_layer.layers_nn["body"].update_pos(gaussian_body_vals, items_body)


        gaussian_body_vals["label_colors"]  = torch.zeros_like(gaussian_body_vals["colors"]).to(config.device)
        gaussian_cloth_vals["label_colors"] = torch.full_like(gaussian_cloth_vals["colors"], 1.0).to(config.device)
        gaussian_vals = merge_gaussian_val(gaussian_cloth_vals, gaussian_body_vals)

        gaussian_vals["colors"] = torch.concat([gaussian_vals["label_colors"], gaussian_vals["colors"]], dim=1)
        render_ret = render3(
            gaussian_vals,
            bg_color.repeat(2),
            items_body['extr'],
            items_body['intr'],
            items_body['img_w'],
            items_body['img_h']
        )
        rgb_map = render_ret['render'][3:, :, :]
        label_rgb_map = render_ret['render'][:3, :, :]

        gaussian_body_vals["colors"] = torch.concat([gaussian_body_vals["label_colors"], gaussian_body_vals["colors"]], dim=1)
        render_body_ret = render3(
            gaussian_body_vals,
            bg_color.repeat(2),
            items_body['extr'],
            items_body['intr'],
            items_body['img_w'],
            items_body['img_h']
        )
        body_rgb_map = render_body_ret['render'][3:, :, :]
        label_body_rgb_map = render_body_ret['render'][:3, :, :]

        if layers == ["body"]:
            ret = {
                'rgb_map':  body_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_body_rgb_map.permute(1, 2, 0),
            }
            return ret
        

        # render cloth only
        
        gaussian_cloth_vals["colors"] = torch.concat([gaussian_cloth_vals["label_colors"], gaussian_cloth_vals["colors"]], dim=1)
        render_cloth_ret = render3(
            gaussian_cloth_vals,
            bg_color.repeat(2),
            items_body['extr'],
            items_body['intr'],
            items_body['img_w'],
            items_body['img_h']
        )
        cloth_rgb_map = render_cloth_ret['render'][3:, :, :]
        label_cloth_rgb_map = render_cloth_ret['render'][:3, :, :]
        if layers == ["cloth"]:
            ret = {
                'rgb_map': cloth_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_cloth_rgb_map.permute(1, 2, 0),
            }
            return ret


        filtered_rgb_map, visibility_mask, depth_diff = filter_gaussian(render_cloth_ret, render_body_ret, label_rgb_map, rgb_map, render_ret["mask"], 1, buffer=0.05)
        ret = {
            'rgb_map': filtered_rgb_map.permute(1, 2, 0),
            'body_visible_mask': visibility_mask.permute(1, 2, 0),
            "label_rgb_map": label_rgb_map.permute(1, 2, 0),
            "depth_diff_map": depth_diff.permute(1, 2, 0)
        }
        # ret = {
        #     'rgb_map': rgb_map.permute(1, 2, 0),
        #     "label_rgb_map": label_rgb_map.permute(1, 2, 0),
        # }
        return ret
    
    def render_outer_filter(self, items_body, items_cloth, bg_color = (0., 0., 0.), layers=None):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items_body["smpl_pos_map"][layer] = items_body["smpl_pos_map"][layer].squeeze(0)
            items_cloth["smpl_pos_map"][layer] = items_cloth["smpl_pos_map"][layer].squeeze(0)
    
        gaussian_body_vals = self.body_layer.render(items_body, layers=["body"], only_gaussian=True)
        gaussian_cloth_vals = self.upper_layer.render(items_cloth, layers=["cloth"], only_gaussian=True)
        gaussian_outer_vals = self.upper_layer.render(items_cloth, layers=["outer"], only_gaussian=True)
        gaussian_cloth_body_vals = self.upper_layer.render(items_cloth, layers=["body"], only_gaussian=True)


        update_offset_mask_cloth = self.cloth_mask[self.upper_layer.layers_nn["body"].cano_smpl_mask]
        # offset_to_update =  gaussian_cloth_body_vals["offset"][update_offset_mask_cloth]

        update_offset_mask_body = self.cloth_mask[self.body_layer.layers_nn["body"].cano_smpl_mask] 
        ori_gaussian_body_pos = gaussian_body_vals["positions"]
        gaussian_body_vals = self.body_layer.layers_nn["body"].update_selected_pos(gaussian_body_vals, gaussian_cloth_body_vals, items_body, update_offset_mask_cloth, update_offset_mask_body)
        # self.smooth_boundary(self.cloth_mask, gaussian_body_vals, self.cloth_mask, ori_gaussian_body_pos, kernel_size=10, K=100)
        # gaussian_body_vals = self.body_layer.layers_nn["body"].update_pos(gaussian_body_vals, items_body)


        gaussian_body_vals["label_colors"]  = torch.zeros_like(gaussian_body_vals["colors"]).to(config.device)
        gaussian_body_vals["label_colors"][:, 2] = 1
        gaussian_cloth_vals["label_colors"] = torch.full_like(gaussian_cloth_vals["colors"], 0.0).to(config.device)
        gaussian_cloth_vals["label_colors"][:, 1] = 1
        gaussian_outer_vals["label_colors"] = torch.full_like(gaussian_outer_vals["colors"], 0.0).to(config.device)
        gaussian_outer_vals["label_colors"][:, 0] = 1

        gaussian_full_cloth_vals = merge_gaussian_val(gaussian_cloth_vals, gaussian_outer_vals)
        gaussian_vals = merge_gaussian_val(gaussian_full_cloth_vals, gaussian_body_vals)

   
        render_ret = render_gaussian(gaussian_vals, items_body, bg_color)
        rgb_map = render_ret['render'][3:, :, :]
        label_rgb_map = render_ret['render'][:3, :, :]


        render_body_ret = render_gaussian(gaussian_body_vals, items_body, bg_color)
        body_rgb_map = render_body_ret['render'][3:, :, :]
        label_body_rgb_map = render_body_ret['render'][:3, :, :]

        if layers == ["body"]:
            ret = {
                'rgb_map':  body_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_body_rgb_map.permute(1, 2, 0),
            }
            return ret

        # render cloth only
        render_cloth_ret = render_gaussian(gaussian_full_cloth_vals, items_body, bg_color)
        cloth_rgb_map = render_cloth_ret['render'][3:, :, :]
        label_cloth_rgb_map = render_cloth_ret['render'][:3, :, :]
        if layers == ["cloth"]:
            ret = {
                'rgb_map': cloth_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_cloth_rgb_map.permute(1, 2, 0),
            }
            return ret
        
        render_inner_cloth_ret = render_gaussian(gaussian_cloth_vals, items_body, bg_color)
        render_outer_ret = render_gaussian(gaussian_outer_vals, items_body, bg_color)
        filtered_cloth_rgb_map, visibility_cloth_mask, depth_diff = filter_gaussian(render_outer_ret, render_inner_cloth_ret, label_cloth_rgb_map, cloth_rgb_map, render_cloth_ret["mask"], 0, buffer=0.05)
        render_cloth_ret["render"][3:, :, :] = filtered_cloth_rgb_map
        render_cloth_ret["render"][:3, :, :] = visibility_cloth_mask

        if layers == ["outer"]:
            ret = {
                'rgb_map':  render_outer_ret["render"][3:, :, :].permute(1, 2, 0),
                "label_rgb_map": label_body_rgb_map.permute(1, 2, 0),
            }
            return ret
        label_rgb_map[1, :, :] += label_rgb_map[0, :, :]
        filtered_rgb_map, visibility_mask, depth_diff = filter_gaussian(render_cloth_ret, render_body_ret, label_rgb_map, rgb_map, render_ret["mask"], 1)
        filtered_rgb_map, visibility_mask, depth_diff = filter_gaussian(render_body_ret, render_ret, visibility_mask, filtered_rgb_map, render_ret["mask"], 2)
        ret = {
            'rgb_map': filtered_rgb_map.permute(1, 2, 0),
            'body_visible_mask': visibility_mask.permute(1, 2, 0),
            "label_rgb_map": label_rgb_map.permute(1, 2, 0),
            "depth_diff_map": depth_diff.permute(1, 2, 0)
        }

        # ret = {
        #     'rgb_map': rgb_map.permute(1, 2, 0),
        #     "label_rgb_map": label_rgb_map.permute(1, 2, 0),
        # }
        return ret
