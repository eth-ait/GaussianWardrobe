import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
from pytorch3d.io import save_ply
import cv2 as cv

import config
from network.styleunet.dual_styleunet import DualStyleUNet
from gaussians.gaussian_model import GaussianModel
from gaussians.gaussian_renderer import render3
from pytorch3d.structures import Meshes 
from utils.net_util import read_map_mask
from utils.general_utils import render_gaussian
class AvatarNet(nn.Module):
    def __init__(self, opt, layer=None, data_dir=None):
        super(AvatarNet, self).__init__()
        self.opt = opt
        self.layer = layer
        self.mode = config.opt["mode"]
        self.data_dir = config.opt[self.mode]['data']['data_dir'] if data_dir is None else data_dir
        if layer is None:
            self.smpl_pos_map = config.opt.get("smpl_pos_map", "smpl_pos_map")
            valid_faces = np.load(self.data_dir + '/{}/valid_faces.npy'
                                    .format(self.smpl_pos_map))
            self.valid_faces = torch.from_numpy(valid_faces).to(torch.int64).to(config.device)
        else:
            self.smpl_pos_map = config.opt.get("smpl_pos_map", "smpl_pos_map") + f"_{layer}"
     
            valid_faces = np.load(self.data_dir + '/{}/valid_faces.npy'
                                    .format(self.smpl_pos_map))
            self.valid_faces = torch.from_numpy(valid_faces).to(torch.int64).to(config.device)
            with_faces = np.load(self.data_dir + '/{}/with_face.npy'
                                    .format(self.smpl_pos_map))
            self.with_faces = torch.from_numpy(with_faces).to(torch.bool).to(config.device)
     
            self.cano_offset_map, _ = read_map_mask(self.data_dir + '/{}/cano_smpl_offset_map.exr'
                                     .format(self.smpl_pos_map))
        self.max_sh_degree = 0
        self.cano_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)
        self.cano_smpl_map, self.cano_smpl_mask = read_map_mask(self.data_dir + '/{}/cano_smpl_pos_map.exr'
                                     .format(self.smpl_pos_map))

        
        if layer == "cloth":
            _, self.cloth_segment_mask = read_map_mask(self.data_dir + '/{}/cano_smpl_segment_map.exr'
                                     .format(self.smpl_pos_map))
        
        if layer =="body":
            _, self.smpl_body_mask = read_map_mask(self.data_dir + '/{}/cano_smpl_smplx_fix_offset_map.exr'.format(self.smpl_pos_map))
            _, cano_smpl_hand_mask = read_map_mask(self.data_dir + '/{}/cano_smpl_hand_map.exr'.format(self.smpl_pos_map))
            self.body_hand_idx = cano_smpl_hand_mask[self.cano_smpl_mask]

        self.random_style = opt.get('random_style', False)
        self.with_viewdirs = opt.get('with_viewdirs', True)

        # init canonical gausssian model

        self.init_points = self.cano_smpl_map[self.cano_smpl_mask]
        self.lbs = torch.from_numpy(np.load(self.data_dir + '/{}/init_pts_lbs.npy'
                                            .format(self.smpl_pos_map))).to(torch.float32).to(config.device)
        self.cano_gaussian_model.create_from_pcd(self.init_points, torch.rand_like(self.init_points), spatial_lr_scale = 2.5)

        self.color_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 3, out_size = 1024, style_dim = 512, n_mlp = 2)
        self.position_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 3, out_size = 1024, style_dim = 512, n_mlp = 2)
        self.other_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 8, out_size = 1024, style_dim = 512, n_mlp = 2)

        self.color_style = torch.ones([1, self.color_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.color_net.style_dim)
        self.position_style = torch.ones([1, self.position_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.position_net.style_dim)
        self.other_style = torch.ones([1, self.other_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.other_net.style_dim)

        if self.with_viewdirs:
            self.cano_nml_map, self.cano_smpl_mask = read_map_mask(self.data_dir + '/{}/cano_smpl_nml_map.exr'
                                     .format(self.smpl_pos_map))
            self.cano_nmls = self.cano_nml_map[self.cano_smpl_mask]
            self.viewdir_net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1)
            )
        self.selected_gaussian = None

    def get_normal(self, gaussian_pos):
        gaussian_mesh = Meshes(gaussian_pos.unsqueeze(0), self.valid_faces.unsqueeze(0))
        # save_ply(f"/local/home/zhiychen/AnimatableGaussain/gaussian_{layer}_mesh.ply", gaussian_pos, self.valid_faces)
        vertex_normal = gaussian_mesh.verts_normals_packed()
        return vertex_normal
    def generate_mean_hands(self):
        # print('# Generating mean hands ...')
        import glob
        # get hand mask
        lbs_argmax = self.lbs.argmax(1)
        self.hand_mask = lbs_argmax == 20
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax == 21)
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax >= 25)

        pose_map_paths = sorted(glob.glob(self.data_dir + '/{}/%08d.exr'
                                          .format(self.smpl_pos_map) % config.opt['test']['fix_hand_id']))
        smpl_pos_map = cv.imread(pose_map_paths[0], cv.IMREAD_UNCHANGED)
        pos_map_size = smpl_pos_map.shape[1] // 2
        smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
        smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
        pose_map = torch.from_numpy(smpl_pos_map).to(torch.float32).to(config.device)
        pose_map = pose_map[:3]

        cano_pts, _ = self.get_positions(pose_map)
        opacity, scales, rotations = self.get_others(pose_map)
        colors, color_map = self.get_colors(pose_map)

        self.hand_positions = cano_pts#[self.hand_mask]
        self.hand_opacity = opacity#[self.hand_mask]
        self.hand_scales = scales#[self.hand_mask]
        self.hand_rotations = rotations#[self.hand_mask]
        self.hand_colors = colors#[self.hand_mask]

        # # debug
        # hand_pts = trimesh.PointCloud(self.hand_positions.detach().cpu().numpy())
        # hand_pts.export('./debug/hand_template.obj')
        # exit(1)

    def transform_cano2live(self, gaussian_vals, items):
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return gaussian_vals

    def get_positions(self, pose_map, return_map = False, with_offset=False, position_only=False):
        position_map, _ = self.position_net([self.position_style], pose_map[None], randomize_noise = False)
        front_position_map, back_position_map = torch.split(position_map, [3, 3], 1)
        position_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)
        if (self.opt.get("offset_mode")) == "tightness":
            delta_position = self.cano_tightness_map[self.cano_smpl_mask] * position_map[self.cano_smpl_mask]
        elif (self.opt.get("offset_mode")) == "tightness_scaled":
            self.cano_tightness_map[self.cano_smpl_mask] *= 0.05/self.cano_tightness_map[self.cano_smpl_mask].mean()
            delta_position = self.cano_tightness_map[self.cano_smpl_mask] * position_map[self.cano_smpl_mask]
        elif self.opt.get("offset_mode") == "no_scale":
            delta_position = position_map[self.cano_smpl_mask]
        else:   
            delta_position = 0.05 * position_map[self.cano_smpl_mask]
        positions = delta_position + self.cano_gaussian_model.get_xyz
        if self.layer and with_offset:
            positions = self.cano_offset_map[self.cano_smpl_mask] + positions
        if position_only:
            return positions
        if return_map:
            return positions, position_map, delta_position
        else:
            return positions, delta_position

    def get_others(self, pose_map):
        other_map, _ = self.other_net([self.other_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(other_map, [8, 8], 1)
        other_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0)
        others = other_map[self.cano_smpl_mask]  # (N, 8)
        opacity, scales, rotations = torch.split(others, [1, 3, 4], 1)
        opacity = self.cano_gaussian_model.opacity_activation(opacity + self.cano_gaussian_model.get_opacity_raw)
        scales = self.cano_gaussian_model.scaling_activation(scales + self.cano_gaussian_model.get_scaling_raw)
        rotations = self.cano_gaussian_model.rotation_activation(rotations + self.cano_gaussian_model.get_rotation_raw)

        return opacity, scales, rotations

    def get_colors(self, pose_map, front_viewdirs = None, back_viewdirs = None):
        color_style = torch.rand_like(self.color_style) if self.random_style and self.training else self.color_style
        color_map, _ = self.color_net([color_style], pose_map[None], randomize_noise = False, view_feature1 = front_viewdirs, view_feature2 = back_viewdirs)
        front_color_map, back_color_map = torch.split(color_map, [3, 3], 1)
        color_map = torch.cat([front_color_map, back_color_map], 3)[0].permute(1, 2, 0)
        colors = color_map[self.cano_smpl_mask]
        return colors, color_map

    def get_viewdir_feat(self, items):
        with torch.no_grad():
            pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
            live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
            live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_nmls)
            cam_pos = -torch.matmul(torch.linalg.inv(items['extr'][:3, :3]), items['extr'][:3, 3])
            viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3)
            if self.training:
                viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
            viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3)
            viewdirs = (live_nmls * viewdirs).sum(-1)

            viewdirs_map = torch.zeros(*self.cano_nml_map.shape[:2]).to(viewdirs)
            viewdirs_map[self.cano_smpl_mask] = viewdirs

            viewdirs_map = viewdirs_map[None, None]
            viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest')
            front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [512, 512], -1)

        front_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(front_viewdirs)
        back_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(back_viewdirs)
        return front_viewdirs, back_viewdirs

    def get_pose_map(self, items):
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats_woRoot'])
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros_like(self.cano_smpl_map)
        live_pos_map[self.cano_smpl_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = torch.cat(torch.split(live_pos_map, [512, 512], 2), 0)
        items.update({
            'smpl_pos_map': live_pos_map
        })
        return live_pos_map
    
    
    def update_selected_pos(self, gaussian_body_vals, gaussian_cloth_body_vals, items_body, update_offset_mask_cloth, update_offset_mask_body):
        for key in ["opacity", "offset", "scales", "cano_rotations"]:
            new_value = torch.zeros_like(gaussian_body_vals[key]).to(config.device)
            new_value[~update_offset_mask_body] = gaussian_body_vals[key][~update_offset_mask_body]
            new_value[update_offset_mask_body] =  gaussian_cloth_body_vals[key][update_offset_mask_cloth]
            gaussian_body_vals[key] = new_value
        
        gaussian_body_vals["positions"] = gaussian_body_vals["offset"]  + self.cano_gaussian_model.get_xyz + self.cano_offset_map[self.cano_smpl_mask] 
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items_body['cano2live_jnt_mats'])
        gaussian_body_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_body_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_body_vals['cano_rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_body_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return gaussian_body_vals
    
    def update_pos(self, gaussian_body_vals, items_body):

        gaussian_body_vals["positions"] = gaussian_body_vals["offset"]  + self.cano_gaussian_model.get_xyz + self.cano_offset_map[self.cano_smpl_mask] 
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items_body['cano2live_jnt_mats'])
        gaussian_body_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_body_vals['positions']) + pt_mats[..., :3, 3]


        return gaussian_body_vals


    def select_gaussian(self, gaussian_vals):
        # use surface opacity of covered body
        surface_opacity = gaussian_vals["opacity"][self.selected_gaussian].detach().mean()
        new_opacity = torch.zeros_like(gaussian_vals["opacity"]).to(config.device)
        new_opacity[~self.selected_gaussian] = 1
        new_opacity[self.selected_gaussian] = gaussian_vals["opacity"][self.selected_gaussian]
        gaussian_vals["opacity"] = new_opacity
        
    def transform_normal(self, A, B):
        B_rotate = B.clone()[:3, :3]
        B_rotate[1:,:] = -  B_rotate[1:,:] 
        transformed_norm = torch.einsum('ij,jk->ik', A, B_rotate.T)
        normalized_norm = (transformed_norm + 1) / 2
        return torch.flip(normalized_norm, dims=(1,))
    
    def render(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False, only_gaussian=False, layers=None):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        if self.layer is None:
            pose_map = items['smpl_pos_map'][:3]
        else:
            pose_map = items['smpl_pos_map'][self.layer][:3]
        assert not (use_pca and use_vae), "Cannot use both PCA and VAE!"
        if use_pca:
            pose_map = items['smpl_pos_map_pca'][:3]
        if use_vae:
            pose_map = items['smpl_pos_map_vae'][:3]


       
        cano_pts, pos_map, offset = self.get_positions(pose_map, return_map = True, with_offset = True)
        opacity, scales, rotations = self.get_others(pose_map)
        # if not self.training:
        # scales = torch.clip(scales, 0., 0.03)
        if self.with_viewdirs:
            front_viewdirs, back_viewdirs = self.get_viewdir_feat(items)
        else:
            front_viewdirs, back_viewdirs = None, None
        colors, color_map = self.get_colors(pose_map, front_viewdirs, back_viewdirs)

        if not self.training and config.opt['test'].get('fix_hand', False) and config.opt['mode'] != 'train' and self.layer != "cloth":
            # print('# fuse hands ...')
            import utils.geo_util as geo_util
            cano_xyz = self.init_points
            wl = torch.sigmoid(2.5 * (geo_util.normalize_vert_bbox(items['left_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] + 2.0))
            wr = torch.sigmoid(-2.5 * (geo_util.normalize_vert_bbox(items['right_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] - 2.0))
            wl[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.
            wr[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.

            s = torch.maximum(wl + wr, torch.ones_like(wl))
            wl, wr = wl / s, wr / s

            w = wl + wr
            cano_pts = w * self.hand_positions + (1.0 - w) * cano_pts
            opacity = w * self.hand_opacity + (1.0 - w) * opacity
            scales = w * self.hand_scales + (1.0 - w) * scales
            rotations = w * self.hand_rotations + (1.0 - w) * rotations
            # cano_pts[self.body_hand_idx] = self.hand_positions[self.body_hand_idx]
            # opacity[self.body_hand_idx] = self.hand_opacity[self.body_hand_idx]
            # scales[self.body_hand_idx] = self.hand_scales[self.body_hand_idx]
            # rotations[self.body_hand_idx] = self.hand_rotations[self.body_hand_idx]
            colors = w * self.hand_colors + (1.0 - w) * colors
        # preparing gaussian values
        gaussian_vals = {
            'positions': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'max_sh_degree': self.max_sh_degree,
            'offset': offset,
            'pos_map': pos_map,
        }


        cano_gaussian_pos = gaussian_vals["positions"]
        gaussian_vals["cano_positions"] = cano_gaussian_pos

        # select gaussian
        # if not (self.selected_gaussian is None):
        #     self.select_gaussian(gaussian_vals)
        # transform to deformed space
        gaussian_vals["cano_rotations"] = gaussian_vals["rotations"].clone()
        gaussian_vals = self.transform_cano2live(gaussian_vals, items)
        if self.layer == "body":
            gaussian_vals["gaussian_nml"] = self.get_normal(gaussian_vals["positions"])
            gaussian_vals["label_colors"] = torch.full_like(gaussian_vals["colors"], 0.5).to(config.device)
            # gaussian_vals["label_colors"] = torch.full_like(gaussian_vals["colors"], 0).to(config.device)
            # gaussian_vals["label_colors"][:, 0] = 1
        elif self.layer == "cloth":
            gaussian_vals["gaussian_nml"] = self.get_normal(gaussian_vals["positions"])
            gaussian_vals["label_colors"] = torch.full_like(gaussian_vals["colors"], 0.0).to(config.device)
            gaussian_vals["label_colors"][self.cloth_segment_mask[self.cano_smpl_mask], :] = config.RGB_LABEL['lower']
            gaussian_vals["label_colors"][(~self.cloth_segment_mask)[self.cano_smpl_mask], :] = config.RGB_LABEL['upper']
            # gaussian_vals["label_colors"][self.cloth_segment_mask[self.cano_smpl_mask], 2] = 1
            # gaussian_vals["label_colors"][(~self.cloth_segment_mask)[self.cano_smpl_mask], 1] = 1
        elif self.layer == "outer":
            gaussian_vals["label_colors"] = torch.zeros_like(gaussian_vals["colors"]).to(config.device)
            gaussian_vals["label_colors"][:, :] = config.RGB_LABEL['outer']
        # gaussian_vals["gaussian_norm"] = self.transform_normal(gaussian_vals["gaussian_norm"], items['extr'])
        # In multilayer case we only use the gaussian_vals and render later
        if only_gaussian:
            return gaussian_vals
        gaussian_vals["label_colors"] = torch.zeros_like(gaussian_vals["colors"])
        render_ret = render_gaussian(gaussian_vals, items, bg_color)

        rgb_map = render_ret['render'][3:, :, :].permute(1, 2, 0)
        mask_map = render_ret['mask'].permute(1, 2, 0)

        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
            'pos_map': pos_map,
            "offset": offset
        }

        if not self.training:
            ret.update({
                'cano_tex_map': color_map,
                'posed_gaussians': gaussian_vals
            })

        return ret