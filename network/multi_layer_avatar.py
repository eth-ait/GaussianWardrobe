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
from utils.net_util import read_map_mask, process_pos_map
from utils.general_utils import filter_gaussian, render_gaussian
class MultiGaussianModel:
    def __init__(self, model_dict, layers):
        self.cano_gaussian_dict = {}
        self.layers = layers
        for layer in layers:
            self.cano_gaussian_dict[layer] = model_dict[layer].cano_gaussian_model
        

    @property
    def get_scaling(self):
        scaling_l = []
        for layer in self.layers:
            scaling_l.append(self.cano_gaussian_dict[layer].get_scaling)
        return torch.concat(scaling_l, dim=0)

    @property
    def get_rotation(self):
        rotation_l = []
        for layer in self.layers:
            rotation_l.append(self.cano_gaussian_dict[layer].get_rotation)
        return torch.concat(rotation_l, dim=0)

    @property
    def get_xyz(self):
        xyz_l = []
        for layer in self.layers:
            xyz_l.append(self.cano_gaussian_dict[layer].get_xyz)
        return torch.concat(xyz_l, dim=0)

    @property
    def get_opacity(self):
        opacity_l = []
        for layer in self.layers:
            opacity_l.append(self.cano_gaussian_dict[layer].get_opacity)
        return torch.concat(opacity_l, dim=0)


class MultiLAvatarNet(nn.Module):
    def __init__(self, opt, layers, data_dir=None):
        super(MultiLAvatarNet, self).__init__()
        self.layers = layers
        self.layers_nn = nn.ModuleDict()
        self.init_points = []
        self.lbs = []
        for layer in layers:
            self.layers_nn[layer] = AvatarNet(opt, layer, data_dir=data_dir)
            self.init_points.append(self.layers_nn[layer].cano_smpl_map[self.layers_nn[layer].cano_smpl_mask])
            self.lbs.append(self.layers_nn[layer].lbs)
        self.mode = config.opt["mode"]
        self.init_points = torch.concat(self.init_points, dim=0)
        self.lbs = torch.concat(self.lbs, dim=0)
        self.max_sh_degree = 0
        self.cano_gaussian_model = MultiGaussianModel(self.layers_nn, self.layers)
        self.data_dir = config.opt[self.mode]['data']['data_dir'] if data_dir is None else data_dir
        self.upper_body_mask = self.layers_nn["body"].cano_smpl_mask & (~self.layers_nn["cloth"].cano_smpl_mask)
        self.selected_body_gaussian = self.upper_body_mask[self.layers_nn["body"].cano_smpl_mask]
        self.upper_cloth_mask = self.layers_nn["cloth"].cano_smpl_mask & self.layers_nn["body"].cano_smpl_mask
        self.selected_cloth_gaussian = self.upper_cloth_mask[self.layers_nn["cloth"].cano_smpl_mask]
        _, self.cano_smpl_fix_offset_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["body"].smpl_pos_map}/cano_smpl_smplx_fix_offset_map.exr')
        self.cano_smpl_fix_offset_input_mask = process_pos_map(self.data_dir + f'/{self.layers_nn["body"].smpl_pos_map}/cano_smpl_smplx_fix_offset_map.exr')
        self.cano_smpl_fix_offset_input_mask = torch.linalg.norm(torch.tensor(self.cano_smpl_fix_offset_input_mask[:3, :, :]).to(config.device) , dim = 0) > 0.
        self.smplx_fix_offset_idx = self.cano_smpl_fix_offset_mask[self.layers_nn["body"].cano_smpl_mask]

        _, cano_smpl_hand_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["body"].smpl_pos_map}/cano_smpl_hand_map.exr')
        self.body_hand_idx = cano_smpl_hand_mask[self.layers_nn["body"].cano_smpl_mask]

        # set up select gaussian for body
        self.layers_nn["body"].selected_gaussian = self.selected_body_gaussian
    def transform_cano2live(self, gaussian_vals, lbs, items):
        pt_mats = torch.einsum('nj,jxy->nxy', lbs, items['cano2live_jnt_mats'])
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)
        return gaussian_vals

    def get_lbs(self, layers=None):
        if layers is None:
            return self.lbs
        layer_lbs = []
        for layer in layers:
            layer_lbs.append(self.layers_nn[layer].lbs)
        return torch.concat(layer_lbs, dim=0)
    
    def get_init_points(self, layers=None):
        if layers is None:
            return self.init_points
        layer_init_points = []
        for layer in layers:
            layer_init_points.append(self.layers_nn[layer].cano_smpl_map[self.layers_nn[layer].cano_smpl_mask])
        return torch.concat(layer_init_points, dim=0)
    def upscale_opacity(self, gaussian_vals):
        gaussian_vals["opacity"][gaussian_vals["opacity"] >= 0.8] = 1

    def filter_gaussian(self, gaussian_vals, mask):
        gaussian_vals_filtered = {}
        for key in gaussian_vals.keys():
            if key == "max_sh_degree" or key == "pos_map":
                gaussian_vals_filtered[key] = gaussian_vals[key]
            # elif key == "opacity":
            #     gaussian_vals_filtered[key] = torch.ones_like(gaussian_vals[key]).to(config.device)
            else:
                gaussian_vals_filtered[key] = gaussian_vals[key][mask]
        return gaussian_vals_filtered
    
    def render_filtered(self, items, bg_color = (0., 0., 0.), use_pca = -1, layers=None, only_gaussian=False):
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items["smpl_pos_map"][layer] = items["smpl_pos_map"][layer].squeeze(0)
        gaussian_body_vals = self.layers_nn["body"].render(items, only_gaussian=True)
        gaussian_inner_cloth_vals = self.layers_nn["cloth"].render(items, only_gaussian=True)

        # use constant color for label image
        body_color = torch.zeros_like(gaussian_body_vals["colors"]).to(config.device)
        body_color[:, 2] = 1
        gaussian_body_vals["label_colors"] = body_color
        cloth_color = torch.full_like(gaussian_inner_cloth_vals["colors"], 0.0).to(config.device)
        cloth_color[:, 1] = 1
        gaussian_inner_cloth_vals["label_colors"] = cloth_color
        gaussian_vals = self.concat_gaussian(gaussian_body_vals, gaussian_inner_cloth_vals)
        if "outer" in self.layers:
            gaussian_outer_vals = self.layers_nn["outer"].render(items, only_gaussian=True)
            outer_color = torch.full_like(gaussian_outer_vals["colors"], 0.).to(config.device)
            outer_color[:, 0] = 1
            gaussian_outer_vals["label_colors"] = outer_color
            gaussian_cloth_vals = self.concat_gaussian(gaussian_inner_cloth_vals, gaussian_outer_vals)
            gaussian_vals = self.concat_gaussian(gaussian_cloth_vals, gaussian_body_vals)
            render_outer_ret = render_gaussian(gaussian_outer_vals, items, bg_color)
            label_outer_rgb_map = render_outer_ret['render'][:3, :, :]
            outer_rgb_map = render_outer_ret['render'][3:, :, :]
        else:
            gaussian_cloth_vals = gaussian_inner_cloth_vals


        render_ret = render_gaussian(gaussian_vals, items, bg_color)
        label_rgb_map = render_ret['render'][:3, :, :]
        rgb_map = render_ret['render'][3:, :, :]
        mask_map = render_ret['mask']
        
        # render body only
        render_body_ret = render_gaussian(gaussian_body_vals, items, bg_color)
        if layers == ["body"]:
            ret = {
                'rgb_map': render_body_ret['render'][3:, :, :].permute(1, 2, 0),
                "label_rgb_map": render_body_ret['render'][:3, :, :].permute(1, 2, 0),
            }
            return ret
        # render cloth only
        render_cloth_ret = render_gaussian(gaussian_cloth_vals, items, bg_color)
        label_cloth_rgb_map = render_cloth_ret['render'][:3, :, :]
        cloth_rgb_map = render_cloth_ret['render'][3:, :, :]
        cloth_mask_map = render_cloth_ret['mask']
        if layers == ["cloth"]:
            ret = {
                'rgb_map': cloth_rgb_map.permute(1, 2, 0),
                "label_rgb_map": label_cloth_rgb_map.permute(1, 2, 0),
            }
            return ret


        if layers == ["outer"]:
            render_inner_cloth_ret = render_gaussian(gaussian_inner_cloth_vals, items, bg_color)
            filtered_cloth_rgb_map, visibility_cloth_mask, depth_diff = filter_gaussian(render_outer_ret, render_inner_cloth_ret, label_outer_rgb_map, cloth_rgb_map, cloth_mask_map, 0, buffer=0.05)
            render_cloth_ret["render"][3:, :, :] = filtered_cloth_rgb_map
            render_cloth_ret["render"][:3, :, :] = visibility_cloth_mask
            ret = {
                'rgb_map': filtered_cloth_rgb_map.permute(1, 2, 0),
                'body_visible_mask': visibility_cloth_mask.permute(1, 2, 0),
                "label_rgb_map": label_outer_rgb_map.permute(1, 2, 0),
            }
            return ret
        if "outer" in self.layers:
            render_inner_cloth_ret = render_gaussian(gaussian_inner_cloth_vals, items, bg_color)
            filtered_cloth_rgb_map, visibility_cloth_mask, depth_diff = filter_gaussian(render_outer_ret, render_inner_cloth_ret, label_outer_rgb_map, cloth_rgb_map, cloth_mask_map, 0, buffer=0.05)
            render_cloth_ret["render"][3:, :, :] = filtered_cloth_rgb_map
            render_cloth_ret["render"][:3, :, :] = visibility_cloth_mask
            label_rgb_map[1, :, :] += label_rgb_map[0, :, :]
            label_cloth_rgb_map[1, :, :] += label_cloth_rgb_map[0, :, :]
        filtered_rgb_map, visibility_mask, depth_diff = filter_gaussian(render_cloth_ret, render_body_ret, label_cloth_rgb_map, rgb_map, mask_map, 1, buffer=0.03)

        ret = {
            'rgb_map': filtered_rgb_map.permute(1, 2, 0),
            'body_visible_mask': visibility_mask,
            "label_rgb_map": label_rgb_map.permute(1, 2, 0),
        }
        return ret

    def concat_gaussian(self, gaussian_val_1, gaussian_val_2):
        gaussian_val = {}
        for key in gaussian_val_1.keys():
            if key == "max_sh_degree":
                gaussian_val[key] = gaussian_val_1[key]
            elif key != "gaussian_nml":
                gaussian_val[key] = torch.concat([gaussian_val_1[key], gaussian_val_2[key]], dim=0)
        return gaussian_val
    
    def render(self, items, bg_color = (0., 0., 0.), use_pca = -1, layers=None, only_gaussian=False):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        for layer in self.layers:
            items["smpl_pos_map"][layer] = items["smpl_pos_map"][layer].squeeze(0)
            items["smpl_cano_pos_map"][layer] = items["smpl_cano_pos_map"][layer].squeeze(0)

        if len(self.layers) == 2:
            gaussian_body_vals = self.layers_nn["body"].render(items, only_gaussian=True)
            gaussian_cloth_vals = self.layers_nn["cloth"].render(items, only_gaussian=True)
            gaussian_vals = {}

            gaussian_vals = self.concat_gaussian(gaussian_body_vals, gaussian_cloth_vals)

            if layers == ["body"]:
                gaussian_vals = gaussian_body_vals
            elif layers == ["cloth"]:
                gaussian_vals = gaussian_cloth_vals
            elif layers == ["lower"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                selected_cloth = upper_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            elif layers == ["upper"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                lower_mask = ~upper_mask
                selected_cloth = lower_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            if only_gaussian:
                return gaussian_vals


            gaussian_vals["colors"] = torch.concat([gaussian_vals["label_colors"], gaussian_vals["colors"]], dim=1)
            render_ret = render3(
                gaussian_vals,
                bg_color.repeat(2),
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            label_rgb_map = render_ret['render'][:3, :, :].permute(1, 2, 0)
            rgb_map = render_ret['render'][3:, :, :].permute(1, 2, 0)
            label_rgb_map_pixel = self.pixelwise_label(label_rgb_map, render_ret["mask"])

            # mask_map = render_ret['mask'][].permute(1, 2, 0)
            hand_color = gaussian_body_vals["colors"][self.body_hand_idx]
            covered_body_color = gaussian_body_vals["colors"][~self.selected_body_gaussian]
            gaussian_body_vals["colors"] = torch.full_like(gaussian_body_vals["colors"], 1.0).to(config.device)
            gaussian_body_vals["colors"] = gaussian_body_vals["colors"].repeat(1, 2)
            render_ret_body = render3(
                gaussian_body_vals,
                torch.tensor([0., 0., 0., 0., 0., 0.]).to(config.device),
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            label_body_rgb_map = render_ret_body['render'][:3, :, :].permute(1, 2, 0)
            
            
            ret = {
                'rgb_map': rgb_map,
                'offset': gaussian_vals["offset"],
                "label_body_rgb_map": label_body_rgb_map,
                "label_rgb_map": label_rgb_map,
                "label_pixel_map": label_rgb_map_pixel,
                "body_offset": gaussian_body_vals["offset"], 
                # "normal_map": normal_map,
                "gaussian_cloth_pos": gaussian_cloth_vals["positions"],
                "gaussian_body_pos": gaussian_body_vals["positions"],
                "gaussian_body_nml": gaussian_body_vals["gaussian_nml"],
                "gaussian_body_opacity": gaussian_body_vals["opacity"],
                "posed_gaussians": gaussian_vals,
                "gaussian_body_hand_color": hand_color,
                "gaussian_body_covered_color": covered_body_color
            }
            
        else:
            with torch.no_grad():
                gaussian_body_vals = self.layers_nn["body"].render(items, only_gaussian=True)
                gaussian_body_vals["label_colors"] = torch.full_like(gaussian_body_vals["colors"], 0.5).to(config.device)
            gaussian_cloth_vals = self.layers_nn["cloth"].render(items, only_gaussian=True)
            gaussian_outer_vals = self.layers_nn["outer"].render(items, only_gaussian=True)
            gaussian_vals = {}

            for key in gaussian_outer_vals.keys():
                if key == "max_sh_degree":
                    gaussian_vals[key] = gaussian_cloth_vals[key]
                else:
                    gaussian_vals[key] = torch.concat([gaussian_body_vals[key], gaussian_cloth_vals[key], gaussian_outer_vals[key]], dim=0)

            if layers == ["body"]:
                gaussian_vals = gaussian_body_vals
            elif layers == ["cloth"]:
                gaussian_vals = gaussian_cloth_vals
            elif layers == ["lower"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                selected_cloth = upper_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            elif layers == ["upper"]:
                _, upper_mask = read_map_mask(self.data_dir + f'/{self.layers_nn["cloth"].smpl_pos_map}/cano_smpl_segment_map.exr')
                lower_mask = ~upper_mask
                selected_cloth = lower_mask[self.layers_nn["cloth"].cano_smpl_mask]
                gaussian_vals = self.filter_gaussian(gaussian_cloth_vals, selected_cloth)
            elif layers == ["outer"]:
                gaussian_vals = gaussian_outer_vals
            if only_gaussian:
                return gaussian_vals
                
            
            gaussian_vals["colors"] = torch.concat([gaussian_vals["label_colors"], gaussian_vals["colors"]], dim=1)
            render_ret = render3(
                gaussian_vals,
                bg_color.repeat(2),
                items['extr'],
                items['intr'],
                items['img_w'],
                items['img_h']
            )
            label_rgb_map = render_ret['render'][:3, :, :].permute(1, 2, 0)
            rgb_map = render_ret['render'][3:, :, :].permute(1, 2, 0)


            ret = {
                "gaussian_cloth_opacity": gaussian_cloth_vals["opacity"],
                'rgb_map': rgb_map,
                'offset': gaussian_vals["offset"],
                'cloth_offset': gaussian_cloth_vals["offset"],
                "label_rgb_map": label_rgb_map,
                "gaussian_cloth_pos": gaussian_cloth_vals["positions"],
                "gaussian_body_pos": gaussian_body_vals["positions"],
                "gaussian_body_nml": gaussian_body_vals["gaussian_nml"],
                "gaussian_outer_pos": gaussian_outer_vals["positions"],
                "gaussian_cloth_nml": gaussian_cloth_vals["gaussian_nml"],
            }
        return ret



    def get_positions(self, pose_map, return_map = False, layers=None, position_only=False):
        if layers is None:
            layers = self.layers
        if return_map:
            positions_l = []
            position_map_l = []
            for layer in layers:
                pose_map_layer = pose_map[layer]
                positions, position_map = self.layers_nn[layer].get_positions(pose_map_layer, return_map)
                positions_l.append(positions)
                position_map_l.append(position_map)
            return torch.concat(positions_l, dim=0), torch.concat(position_map_l, dim=0)
        else:
            positions_l = []
            for layer in layers:
                pose_map_layer = pose_map[layer]
                positions = self.layers_nn[layer].get_positions(pose_map_layer, return_map)
                positions_l.append(positions[0])

            return torch.concat(positions_l, dim=0)

    def get_others(self, pose_map, layers=None):
        opacity_l = []
        scales_l = []
        rotations_l = []
        if layers is None:
            layers = self.layers
        for layer in layers:
            opacity, scales, rotations = self.layers_nn[layer].get_others(pose_map[layer])
            opacity_l.append(opacity)
            scales_l.append(scales)
            rotations_l.append(rotations)
        return torch.concat(opacity_l, dim=0), torch.concat(scales_l, dim=0), torch.concat(rotations_l, dim=0)

    def get_colors(self, pose_map, items, layers=None):
        colors_l = []
        color_map_l = []
        if layers is None:
            layers = self.layers
        for layer in layers:
            if self.layers_nn[layer].with_viewdirs:
                front_viewdirs, back_viewdirs = self.layers_nn[layer].get_viewdir_feat(items)
            else:
                front_viewdirs, back_viewdirs = None, None
            colors, color_map = self.layers_nn[layer].get_colors(pose_map[layer],
                                                                 front_viewdirs, back_viewdirs)
            colors_l.append(colors)
            color_map_l.append(color_map)
        return torch.concat(colors_l, dim=0), torch.concat(color_map_l, dim=0)
    

    def pixelwise_dominant_map_custom(self, tensor: torch.Tensor, mask, color_map=None) -> torch.Tensor:
        """
        Assign custom colors to pixels based on which channel is dominant.
        
        Args:
            tensor (torch.Tensor): Input image in [3, H, W] format, values [0,1].
            colors (list of lists): RGB colors in [0,1] to assign per channel dominance.
                                    Default = pure red, green, blue.
                                    Example: [[1,1,0], [1,0,1], [0,1,1]]  # yellow, magenta, cyan
        
        Returns:
            torch.Tensor: Output tensor [3, H, W] with mapped colors.
        """
        
        
        # Get dominant channel index per pixel
        dominant_idx = torch.argmax(tensor, dim=2)  # [H, W]

        # Build output image by indexing into color_map
        H, W = dominant_idx.shape
        output = torch.zeros((H, W, 3), dtype=tensor.dtype, device=tensor.device)
        low_mask = (mask[0] < 0.1) 
        for c in range(3):  # loop over RGB
            output[:,:, c] = color_map[dominant_idx, c].reshape(H, W)
        output[low_mask, :] = 1.0  
        return output

    def pixelwise_label(self, tensor, mask):
        color1 = torch.tensor([0.5, 0.5, 0.5])
        color2 = torch.tensor([0.0, 0.5, 1])
        color3 = torch.tensor([1, 0.0, 0.5])

        colors = torch.stack([color1, color2, color3]).to(config.device)  # Shape: (3, 3)

        # Flatten image to (H*W, 3)
        h, w, c = tensor.shape
        pixels = tensor.view(-1, 3)

        # Compute distances to each reference color
        # distances shape: (H*W, 3)
        distances = torch.cdist(pixels.unsqueeze(0), colors.unsqueeze(0)).squeeze(0)

        # Closest color index + 1 -> labels 1, 2, 3
        low_mask = (mask[0] < 0.1) 
        labels = torch.argmin(distances, dim=1)
        label_image = labels.view(h, w)/3
        label_image[low_mask] = 1.0  
        return label_image
