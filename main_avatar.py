import os

from utils.geo_util import FaceNormals

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from sklearn import neighbors
import yaml
import shutil
import collections
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import glob
import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import importlib
import wandb
import config
from PIL import Image
from network.lpips import LPIPS
from dataset.dataset_pose import PoseDataset
import utils.net_util as net_util
import utils.visualize_util as visualize_util
from utils.renderer import Renderer
from utils.net_util import to_cuda, GMoF
from utils.visualize_util import post_process_img
from utils.obj_io import save_mesh_as_ply
from utils.general_utils import create_bounding_box
from gaussians.obj_io import save_gaussians_as_ply
from network.multi_layer_avatar import MultiLAvatarNet
from network.combined_network import CombinedAvatarNet
from pytorch3d.ops import knn_points
import pytorch_ssim

def safe_exists(path):
    if path is None:
        return False
    return os.path.exists(path)


class AvatarTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.patch_size = 512
        self.iter_idx = 0
        self.iter_num = 400000
        self.lr_init = float(self.opt['train'].get('lr_init', 5e-4))
        self.robustifier = GMoF(rho=100)
        avatar_module = self.opt['model'].get('module', 'network.avatar')
        avatar_network = self.opt['model'].get('network', 'AvatarNet')
        print('Import AvatarNet from %s' % avatar_module)
        AvatarNet = importlib.import_module(avatar_module).__getattribute__(avatar_network)
        if not config.opt["mode"] == "exchange_cloth":
            self.avatar_net = AvatarNet(self.opt['model'],
                                    self.opt[config.opt["mode"] ]['data'].get('layers', None)).to(config.device)
        else:
            self.avatar_net = AvatarNet(self.opt['model'],
                                    self.opt[config.opt["mode"] ]['data_body'].get('layers', None), data_dir=self.opt['exchange_cloth']['data_upper']["data_dir"]).to(config.device)
        self.optm = torch.optim.Adam(
            self.avatar_net.parameters(), lr = self.lr_init
        )

        self.random_bg_color = self.opt['train'].get('random_bg_color', True)
        self.bg_color = (0., 0., 0.)
        self.bg_color_cuda = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)
        self.loss_weight = self.opt['train']['loss_weight']
        self.finetune_color = self.opt['train']['finetune_color']
        if config.opt["mode"] == "train":
            self.logger = wandb.init(project='AG-avatar', config=config.opt)
        self.smpl_pos_map = config.opt.get("smpl_pos_map", "smpl_pos_map")
        print('# Parameter number of AvatarNet is %d' % (sum([p.numel() for p in self.avatar_net.parameters()])))

    def update_lr(self):
        alpha = 0.05
        progress = self.iter_idx / self.iter_num
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        lr = self.lr_init * learning_factor
        for param_group in self.optm.param_groups:
            param_group['lr'] = lr
        return lr

    @staticmethod
    def requires_net_grad(net: torch.nn.Module, flag = True):
        for p in net.parameters():
            p.requires_grad = flag

    def crop_face(self, patch_size, bbox, *args):
        """
        :param gt_mask: (H, W)
        :param patch_size: resize the cropped patch to the given patch_size
        :param randomly: whether to randomly sample the patch
        :param args: input images with shape of (C, H, W)
        """

        min_v, min_u = bbox[0], bbox[1]
        max_v, max_u = bbox[2], bbox[3]
        len_v = max_v - min_v
        len_u = max_u - min_u
        max_size = max(len_v, len_u)

        cropped_images = []
        for image in args:
            cropped_image = self.bg_color_cuda[:, None, None] * torch.ones((3, max_size, max_size), dtype = image.dtype, device = image.device)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]


            cropped_image = F.interpolate(cropped_image[None], size = (patch_size, patch_size), mode = 'bilinear')[0]
            cropped_images.append(cropped_image)

        # cv.imwrite('cropped_image', cropped_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.imwrite('cropped_gt_image', cropped_gt_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.waitKey(0)

        if len(cropped_images) > 1:
            return cropped_images
        else:
            return cropped_images[0]
        
    def crop_image(self, gt_mask, patch_size, randomly, *args):
        """
        :param gt_mask: (H, W)
        :param patch_size: resize the cropped patch to the given patch_size
        :param randomly: whether to randomly sample the patch
        :param args: input images with shape of (C, H, W)
        """
        mask_uv = torch.argwhere(gt_mask > 0.)
        min_v, min_u = mask_uv.min(0)[0]
        max_v, max_u = mask_uv.max(0)[0]
        len_v = max_v - min_v
        len_u = max_u - min_u
        max_size = max(len_v, len_u)

        cropped_images = []
        if randomly and max_size > patch_size:
            random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
            random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        for image in args:
            cropped_image = self.bg_color_cuda[:, None, None] * torch.ones((3, max_size, max_size), dtype = image.dtype, device = image.device)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

            if randomly and max_size > patch_size:
                cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
            else:
                cropped_image = F.interpolate(cropped_image[None], size = (patch_size, patch_size), mode = 'bilinear')[0]
            cropped_images.append(cropped_image)

        # cv.imwrite('cropped_image', cropped_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.imwrite('cropped_gt_image', cropped_gt_image.detach().cpu().numpy().transpose(1, 2, 0))


        if len(cropped_images) > 1:
            return cropped_images
        else:
            return cropped_images[0]

    def compute_lpips_loss(self, image, gt_image):
        assert image.shape[1] == image.shape[2] and gt_image.shape[1] == gt_image.shape[2]
        lpips_loss = self.lpips.forward(
            image[None, [2, 1, 0]],
            gt_image[None, [2, 1, 0]],
            normalize = True
        ).mean()
        return lpips_loss

    def forward_one_pass_pretrain(self, items):
        total_loss = 0
        batch_losses = {}
        l1_loss = torch.nn.L1Loss()

        items = net_util.delete_batch_idx(items)
        if isinstance(items["smpl_pos_map"], dict):
            pose_map = {}
            for k, v in items["smpl_pos_map"].items():
                pose_map[k] = v[0, :3]
        else:
            pose_map = items["smpl_pos_map"][:3]

        position_loss = l1_loss(self.avatar_net.get_positions(pose_map, position_only=True), self.avatar_net.cano_gaussian_model.get_xyz)
        total_loss += position_loss
        batch_losses.update({
            'position': position_loss.item()
        })

        opacity, scales, rotations = self.avatar_net.get_others(pose_map)
        opacity_loss = l1_loss(opacity, self.avatar_net.cano_gaussian_model.get_opacity)
        total_loss += opacity_loss
        batch_losses.update({
            'opacity': opacity_loss.item()
        })

        scale_loss = l1_loss(scales, self.avatar_net.cano_gaussian_model.get_scaling)
        total_loss += scale_loss
        batch_losses.update({
            'scale': scale_loss.item()
        })

        rotation_loss = l1_loss(rotations, self.avatar_net.cano_gaussian_model.get_rotation)
        total_loss += rotation_loss
        batch_losses.update({
            'rotation': rotation_loss.item()
        })


        total_loss.backward()

        self.optm.step()
        self.optm.zero_grad()

        return total_loss, batch_losses

    def prepare_img(self, img, boundary_mask_img):
        img = img.permute(2, 0, 1)
        # if self.iter_idx > 100000:
        #     return img
        # else:
        return img * boundary_mask_img[None] + (1. - boundary_mask_img[None]) * torch.from_numpy(np.asarray([0.75, 0.75, 0.75])).to(torch.float32).to(config.device)[:, None, None]
    def img_loss(self, gt_img, pred_img, boundary_mask_img=None, mask_img=None):
        if (not (mask_img is None)) and (not (boundary_mask_img is None)):
            pred_img = self.prepare_img(pred_img, boundary_mask_img)
            gt_img[~mask_img] = self.bg_color_cuda
            gt_img = self.prepare_img(gt_img, boundary_mask_img)
        # loss =  ((pred_img - gt_img)**2).mean()
        loss =  torch.abs(pred_img - gt_img).mean()
        return loss
    def seg_loss(self, gt_img, pred_img, boundary_mask_img, mask_img):
        pred_img = self.prepare_img(pred_img, boundary_mask_img)
        gt_img[~mask_img] = self.bg_color_cuda
        gt_img = self.prepare_img(gt_img, boundary_mask_img)
        pred_seg = pred_img[0, mask_img]
        gt_seg = gt_img[0, mask_img]
        eps = 1e-6
        loss =  - (gt_seg * torch.log(pred_seg + eps) + (1 - gt_seg) * torch.log(1 - pred_seg + eps))
        return loss.mean()

    def forward_one_pass(self, items):
        # forward_start = torch.cuda.Event(enable_timing = True)
        # forward_end = torch.cuda.Event(enable_timing = True)
        # backward_start = torch.cuda.Event(enable_timing = True)
        # backward_end = torch.cuda.Event(enable_timing = True)
        # step_start = torch.cuda.Event(enable_timing = True)
        # step_end = torch.cuda.Event(enable_timing = True)

        if self.random_bg_color:
            self.bg_color = np.random.rand(3)
            self.bg_color_cuda = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)

        total_loss = 0
        batch_losses = {}

        items = net_util.delete_batch_idx(items)

        """ Optimize generator """
        if self.finetune_color:
            self.requires_net_grad(self.avatar_net.color_net, True)
            self.requires_net_grad(self.avatar_net.position_net, False)
            self.requires_net_grad(self.avatar_net.other_net, True)
        else:
            self.requires_net_grad(self.avatar_net, True)

        # forward_start.record()
        render_output = self.avatar_net.render(items, self.bg_color)
        

        # mask image & set bg color
        items['color_img'][~items['mask_img']] = self.bg_color_cuda
        mask_img = items['mask_img'].to(torch.float32)
        boundary_mask_img = 1. - items['boundary_mask_img'].to(torch.float32)
        image = self.prepare_img(render_output['rgb_map'], boundary_mask_img)
        gt_image = self.prepare_img(items['color_img'], boundary_mask_img)

        if self.loss_weight["ssim"] > 0.:
            ssim_loss = pytorch_ssim.SSIM()
            ssim_out = 1 - ssim_loss(image.unsqueeze(0), gt_image.unsqueeze(0))
            total_loss += self.loss_weight['ssim'] * ssim_out.item()
            batch_losses.update({
                'ssim': 1 - ssim_out.item()
            })


        if self.loss_weight['consistent_label_body'] > 0.:
            consistent_label_body_loss = self.img_loss(gt_img=items['label_body_img'], pred_img=render_output['label_body_rgb_map'])
            total_loss += self.loss_weight['consistent_label_body'] * consistent_label_body_loss
            batch_losses.update({
                'consistent_label_body_loss': consistent_label_body_loss.item()
            })
            self.logger.log({'consistent_label_body_loss': consistent_label_body_loss.item()})

        if self.loss_weight['consistent_label'] > 0.:
            consistent_label_loss = self.img_loss(gt_img=items['label_img'], pred_img=render_output['label_rgb_map'], 
                                                boundary_mask_img=boundary_mask_img, mask_img=items['mask_img'])
            total_loss += self.loss_weight['consistent_label'] * consistent_label_loss
            batch_losses.update({
                'consistent_label_loss': consistent_label_loss.item()
            })
            self.logger.log({'consistent_label_loss': consistent_label_loss.item()})

        if self.loss_weight['l1'] > 0.:
            l1_loss = self.img_loss(gt_img=gt_image, pred_img=image)
            total_loss += self.loss_weight['l1'] * l1_loss
            batch_losses.update({
                'l1_loss': l1_loss.item()
            })
            self.logger.log({'l1_loss': l1_loss.item()})

        if self.loss_weight.get('mask', 0.) and 'mask_map' in render_output:
            rendered_mask = render_output['mask_map'].squeeze(-1) * boundary_mask_img
            gt_mask = mask_img * boundary_mask_img
            mask_loss = torch.abs(rendered_mask - gt_mask).mean()
            # mask_loss = torch.nn.BCELoss()(rendered_mask, gt_mask)
            total_loss += self.loss_weight.get('mask', 0.) * mask_loss
            batch_losses.update({
                'mask_loss': mask_loss.item()
            })
            self.logger.log({'mask_loss': mask_loss.item()})

        if self.loss_weight['lpips'] > 0.:
            # crop images
            random_patch_flag = False if self.iter_idx < 300000 else True
            image_lpips, gt_image_lpips = self.crop_image(mask_img, self.patch_size, random_patch_flag, image, gt_image)
            lpips_loss = self.compute_lpips_loss(image_lpips, gt_image_lpips)
            total_loss += self.loss_weight['lpips'] * lpips_loss
            batch_losses.update({
                'lpips_loss': lpips_loss.item()
            })
            self.logger.log({'lpips_loss': lpips_loss.item()})
        if self.loss_weight['face_lpips'] > 0.:
            # crop images
            bbox = create_bounding_box(items["head_pixel"], (125, 125), image.shape[1:])
            image_face_lpips, gt_image_face_lpips = self.crop_face(self.patch_size, bbox, image, gt_image)
            face_lpips_loss = self.compute_lpips_loss(image_face_lpips, gt_image_face_lpips)
            total_loss += self.loss_weight['face_lpips'] * face_lpips_loss
            batch_losses.update({
                'face_lpips_loss': face_lpips_loss.item()
            })
            self.logger.log({'face_lpips_loss': face_lpips_loss.item()})


        if self.loss_weight['offset'] > 0.:
            if "offset" in render_output:
                offset = render_output["offset"]
                offset_loss = torch.linalg.norm(offset, dim = -1).mean()
                total_loss += self.loss_weight['offset'] * offset_loss.item()
                batch_losses.update({
                    'offset_loss': offset_loss.item()
                })
                self.logger.log({'offset_loss': offset_loss.item()})
            
        if self.loss_weight["body_loss"] > 0:
            body_loss = self.body_loss(render_output["gaussian_cloth_pos"],
                            render_output["gaussian_body_pos"], render_output["gaussian_body_nml"], 0.025)
            total_loss += self.loss_weight['body_loss'] * body_loss
            batch_losses.update({
                'body_loss': body_loss.item()
            })
            self.logger.log({'body_loss': body_loss.item()})
        if self.loss_weight["inner_cloth_loss"] > 0:
            inner_cloth_loss = self.body_loss(render_output["gaussian_outer_pos"],
                            render_output["gaussian_cloth_pos"], render_output["gaussian_cloth_nml"])
            total_loss += self.loss_weight['inner_cloth_loss'] * inner_cloth_loss
            batch_losses.update({
                'inner_cloth_loss': inner_cloth_loss.item()
            })
            self.logger.log({'inner_cloth_loss': inner_cloth_loss.item()})
        if self.loss_weight["outer_to_body_loss"] > 0:
            outer_to_body_loss = self.body_loss(render_output["gaussian_outer_pos"],
                            render_output["gaussian_body_pos"], render_output["gaussian_body_nml"], eps=5e-2)
            total_loss += self.loss_weight['outer_to_body_loss'] * outer_to_body_loss
            batch_losses.update({
                'outer_to_body_loss': outer_to_body_loss.item()
            })
            self.logger.log({'outer_to_body_loss': outer_to_body_loss.item()})
        if self.loss_weight["normal_loss"] > 0:
            gt_norm = torch.linalg.norm(items['normal_img'], dim=2)
            items['normal_img'][gt_norm > 0] /= gt_norm[gt_norm > 0].unsqueeze(1)
            normal_loss = self.img_loss(gt_img=items['normal_img'], pred_img=render_output['normal_map'],  boundary_mask_img=boundary_mask_img, mask_img=items['mask_img'])
            total_loss += self.loss_weight["normal_loss"] * normal_loss
            batch_losses.update({
                'normal_loss': normal_loss.item()
            })
            self.logger.log({'normal_loss': normal_loss.item()})

        if self.loss_weight["opacity_loss"] > 0:
            gaussian_body_opacity = render_output["gaussian_body_opacity"]
            eps = 1e-6
            # opacity_loss =  -1 * (gaussian_cloth_opacity * (gaussian_cloth_opacity + eps).log() + (1-gaussian_cloth_opacity) * (1 - gaussian_cloth_opacity + eps).log())
            opacity_loss = - torch.log(gaussian_body_opacity) 
            opacity_loss = opacity_loss.mean() * 2
            total_loss += self.loss_weight['opacity_loss'] * opacity_loss
            batch_losses.update({
                'opacity_loss': opacity_loss.item()
            })
        if self.loss_weight['laplacian'] > 0.:
            gaussian_offset = render_output["body_offset"] * 1000
            # read required data, convert and send to device
            neighbor_idx = np.load(config.opt['train']['data']['data_dir'] + '/{}_body/neighbor_idx.npy'
                                      .format(self.smpl_pos_map))
            self.neighbor_idx = torch.from_numpy(neighbor_idx).to(torch.int64).to(config.device)
            neighbor_weights = np.load(config.opt['train']['data']['data_dir'] + '/{}_body/neighbor_weights.npy'
                                      .format(self.smpl_pos_map))
            self.neighbor_weights = torch.from_numpy(neighbor_weights).to(torch.float32).to(config.device)

            lap_out = gaussian_offset + (gaussian_offset[self.neighbor_idx, :] * self.neighbor_weights[:, :, None]).sum(1)
            laplacian_loss = (lap_out ** 2).sum(1)[self.avatar_net.smplx_fix_offset_idx].mean()
            total_loss += self.loss_weight['laplacian'] * laplacian_loss
            batch_losses.update({
                'laplacian_loss': laplacian_loss.item()
            })
            self.logger.log({'laplacian_loss': laplacian_loss.item() })

        if self.loss_weight['inner_cloth_laplacian'] > 0.:
            gaussian_offset = render_output["cloth_offset"] * 1000
            # read required data, convert and send to device
            neighbor_idx = np.load(config.opt['train']['data']['data_dir'] + '/{}_cloth/neighbor_idx.npy'
                                      .format(self.smpl_pos_map))
            self.neighbor_idx = torch.from_numpy(neighbor_idx).to(torch.int64).to(config.device)
            neighbor_weights = np.load(config.opt['train']['data']['data_dir'] + '/{}_cloth/neighbor_weights.npy'
                                      .format(self.smpl_pos_map))
            self.neighbor_weights = torch.from_numpy(neighbor_weights).to(torch.float32).to(config.device)

            lap_out = gaussian_offset + (gaussian_offset[self.neighbor_idx, :] * self.neighbor_weights[:, :, None]).sum(1)
            inner_cloth_laplacian_loss = (lap_out ** 2).sum(1).mean()
            total_loss += self.loss_weight['inner_cloth_laplacian'] * inner_cloth_laplacian_loss
            batch_losses.update({
                'inner_cloth_laplacian_loss': inner_cloth_laplacian_loss.item()
            })
            self.logger.log({'inner_cloth_laplacian_loss': inner_cloth_laplacian_loss.item() })
        
        if self.loss_weight['consistent_body_color'] > 0. and self.epoch_idx > 8 :
            mean_hand_color = render_output["gaussian_body_hand_color"].detach().mean(dim=0)
            irrelevant_color = torch.tensor([0., 1.0, 0.]).to(config.device)
            consistent_body_color_loss = (render_output["gaussian_body_covered_color"]- mean_hand_color).abs().mean()
            total_loss += self.loss_weight['consistent_body_color'] * consistent_body_color_loss
            batch_losses.update({
                'consistent_body_color_loss': consistent_body_color_loss.item()
            })
            self.logger.log({'consistent_body_color_loss': consistent_body_color_loss.item()})
        

        # forward_end.record()

        # backward_start.record()
        total_loss.backward()
        # backward_end.record()

        # step_start.record()
        torch.nn.utils.clip_grad_norm_(self.avatar_net.parameters(), self.opt['train']["gd_clip"])
        self.optm.step()
        self.optm.zero_grad()
        # step_end.record()

        # torch.cuda.synchronize()
        # print(f'Forward costs: {forward_start.elapsed_time(forward_end) / 1000.}, ',
        #       f'Backward costs: {backward_start.elapsed_time(backward_end) / 1000.}, ',
        #       f'Step costs: {step_start.elapsed_time(step_end) / 1000.}')

        return total_loss, batch_losses

    def pretrain(self):
        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        self.dataset = MvRgbDataset(**self.opt['train']['data'])
        batch_size = self.opt['train']['batch_size']
        num_workers = self.opt['train']['num_workers']
        batch_num = len(self.dataset) // batch_size
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers = num_workers,
                                                 drop_last = True)

        # tb writer
        smooth_interval = 10
        smooth_count = 0
        smooth_losses = {}

        for epoch_idx in range(0, 9999999):
            self.epoch_idx = epoch_idx
            for batch_idx, items in enumerate(dataloader):
                self.iter_idx = batch_idx + epoch_idx * batch_num
                items = to_cuda(items)

                # one_step_start.record()
                total_loss, batch_losses = self.forward_one_pass_pretrain(items)
                # one_step_end.record()
                # torch.cuda.synchronize()
                # print('One step costs %f secs' % (one_step_start.elapsed_time(one_step_end) / 1000.))

                # record batch loss
                for key, loss in batch_losses.items():
                    if key in smooth_losses:
                        smooth_losses[key] += loss
                    else:
                        smooth_losses[key] = loss
                smooth_count += 1

                if self.iter_idx % smooth_interval == 0:
                    log_info = 'epoch %d, batch %d, iter %d, ' % (epoch_idx, batch_idx, self.iter_idx)
                    for key in smooth_losses.keys():
                        smooth_losses[key] /= smooth_count
                        log_info = log_info + ('%s: %f, ' % (key, smooth_losses[key]))
                        self.logger.log({key:  smooth_losses[key]})
                        smooth_losses[key] = 0.
                    smooth_count = 0
                    print(log_info)


                if self.iter_idx % 200 == 0 and self.iter_idx != 0:
                    self.mini_test(pretraining = True)

                if self.iter_idx == 5000:
                    model_folder = self.opt['train']['net_ckpt_dir'] + '/pretrained'
                    os.makedirs(model_folder, exist_ok = True)
                    self.save_ckpt(model_folder, save_optm = True)
                    self.iter_idx = 0
                    return

    def train(self):
        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        self.dataset = MvRgbDataset(**self.opt['train']['data'])
        batch_size = self.opt['train']['batch_size']
        num_workers = self.opt['train']['num_workers']
        batch_num = len(self.dataset) // batch_size
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers = num_workers,
                                                 drop_last = True)

        if 'lpips' in self.opt['train']['loss_weight']:
            self.lpips = LPIPS(net = 'vgg').to(config.device)
            for p in self.lpips.parameters():
                p.requires_grad = False
        if self.opt['train']['prev_ckpt'] is not None:
            prev_ckpt_path = os.path.join(self.opt['train']['net_ckpt_dir'], self.opt['train']['prev_ckpt'] )
            start_epoch, self.iter_idx = self.load_ckpt(prev_ckpt_path, load_optm = True)
            start_epoch += 1
            self.iter_idx += 1
        else:
            prev_ckpt_path = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
            if safe_exists(prev_ckpt_path):
                start_epoch, self.iter_idx = self.load_ckpt(prev_ckpt_path, load_optm = True)
                start_epoch += 1
                self.iter_idx += 1
            else:
                if safe_exists(self.opt['train']['pretrained_dir']):
                    self.load_ckpt(self.opt['train']['pretrained_dir'], load_optm = False)
                elif safe_exists(self.opt['train']['net_ckpt_dir'] + '/pretrained'):
                    self.load_ckpt(self.opt['train']['net_ckpt_dir'] + '/pretrained', load_optm = False)
                else:
                    raise FileNotFoundError('Cannot find pretrained checkpoint!')

                self.optm.state = collections.defaultdict(dict)
                start_epoch = 0
                self.iter_idx = 0
        if self.opt['train']['inner_ckpt_dir'] is not None:
            inner_net = MultiLAvatarNet(self.opt['model'], ["body", "cloth"]).to(config.device)
            self.load_ckpt_net(self.opt['train']['inner_ckpt_dir'], inner_net, False)
            self.avatar_net.layers_nn["body"] = inner_net.layers_nn["body"]
            self.avatar_net.layers_nn["cloth"] = inner_net.layers_nn["cloth"]
            self.optm = torch.optim.Adam(self.avatar_net.parameters(), lr = self.lr_init)
            self.optm.state = collections.defaultdict(dict)


        # one_step_start = torch.cuda.Event(enable_timing = True)
        # one_step_end = torch.cuda.Event(enable_timing = True)

        # tb writer
        smooth_interval = 10
        smooth_count = 0
        smooth_losses = {}
        torch.cuda.empty_cache()
        for epoch_idx in range(start_epoch, 9999999):
            self.epoch_idx = epoch_idx
            for batch_idx, items in enumerate(dataloader):
                lr = self.update_lr()

                items = to_cuda(items)
                if ((epoch_idx + 1) % self.opt['train']['dropout_interval']) == 0 and type( items["smpl_pos_map"]) == dict:
                    items["smpl_pos_map"]["body"][:, :, self.avatar_net.cano_smpl_fix_offset_input_mask] = \
                        items["smpl_cano_pos_map"]["body"][:, :, self.avatar_net.cano_smpl_fix_offset_input_mask]
                    items["smpl_pos_map"]["cloth"] = items["smpl_cano_pos_map"]["cloth"]
                # one_step_start.record()
                total_loss, batch_losses = self.forward_one_pass(items)
                # one_step_end.record()
                # torch.cuda.synchronize()
                # print('One step costs %f secs' % (one_step_start.elapsed_time(one_step_end) / 1000.))

                # record batch loss
                for key, loss in batch_losses.items():
                    if key in smooth_losses:
                        smooth_losses[key] += loss
                    else:
                        smooth_losses[key] = loss
                smooth_count += 1

                if self.iter_idx % smooth_interval == 0:
                    log_info = 'epoch %d, batch %d, iter %d, lr %e, ' % (epoch_idx, batch_idx, self.iter_idx, lr)
                    for key in smooth_losses.keys():
                        smooth_losses[key] /= smooth_count
                        log_info = log_info + ('%s: %f, ' % (key, smooth_losses[key]))
                        self.logger.log({key:  smooth_losses[key]})
                        smooth_losses[key] = 0.
                    smooth_count = 0
                    print(log_info)
                    torch.cuda.empty_cache()

                if self.iter_idx % self.opt['train']['eval_interval'] == 0 and self.iter_idx != 0:
                    if self.iter_idx % (10 * self.opt['train']['eval_interval']) == 0:
                        eval_cano_pts = False
                    else:
                        eval_cano_pts = False
                    self.mini_test(eval_cano_pts = eval_cano_pts)

                if self.iter_idx % self.opt['train']['ckpt_interval']['batch'] == 0 and self.iter_idx != 0:
                    for folder in glob.glob(self.opt['train']['net_ckpt_dir'] + '/batch_*'):
                        shutil.rmtree(folder)
                    model_folder = self.opt['train']['net_ckpt_dir'] + '/batch_%d' % self.iter_idx
                    os.makedirs(model_folder, exist_ok = True)
                    self.save_ckpt(model_folder, save_optm = True)

                if self.iter_idx == self.iter_num:
                    print('# Training is done.')
                    return

                self.iter_idx += 1

            """ End of epoch """
            if epoch_idx % self.opt['train']['ckpt_interval']['epoch'] == 0 and epoch_idx != 0:
                model_folder = self.opt['train']['net_ckpt_dir'] + '/epoch_%d' % epoch_idx
                os.makedirs(model_folder, exist_ok = True)
                self.save_ckpt(model_folder)

            if batch_num > 50:
                latest_folder = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
                os.makedirs(latest_folder, exist_ok = True)
                self.save_ckpt(latest_folder)
    def log_image(self, tag, rgb_map, img_factor=1, gt_image=None):
        rgb_map = post_process_img(rgb_map, img_factor)
        if gt_image is not None:
            gt_image = post_process_img(gt_image, img_factor)
            rgb_map = np.concatenate([rgb_map, gt_image], 1)
        self.logger.log({tag: wandb.Image(rgb_map[:,:, ::-1])})
        return rgb_map

    @torch.no_grad()
    def mini_test(self, pretraining = False, eval_cano_pts = False):
        def test_and_log(items, folder_name, img_factor):
            gs_render = self.avatar_net.render(items, self.bg_color)
        
            # cv.imshow('rgb_map', rgb_map.cpu().numpy())
            # cv.waitKey(0)
            if not pretraining:
                output_dir = self.opt['train']['net_ckpt_dir'] + f'/eval/{folder_name}'
            else:
                output_dir = self.opt['train']['net_ckpt_dir'] + f'/eval_pretrain/{folder_name}'
            self.log_image( f"{folder_name}_rgb", gs_render["rgb_map"], img_factor, items["color_img"])
            # render body if multilayer
            if self.opt['model'].get('network', 'AvatarNet') == "MultiLAvatarNet" and  self.opt['train']["data"]["use_body_label"] == True:
                gs_body_render = self.avatar_net.render(items, bg_color = self.bg_color, layers=["body"])
                self.log_image( f"{folder_name}_body", gs_body_render["rgb_map"], img_factor, items["label_body_img"])
            if "label_rgb_map" in gs_render:
                self.log_image(f"{folder_name}_label", gs_render["label_rgb_map"], img_factor, items["label_img"])
            if "normal_map" in gs_render:
                self.log_image( f"{folder_name}_normal", gs_render["normal_map"], img_factor, items["normal_img"])
            os.makedirs(output_dir, exist_ok = True)
            if eval_cano_pts:
                os.makedirs(output_dir + '/cano_pts', exist_ok = True)
                save_mesh_as_ply(output_dir + '/cano_pts/iter_%d.ply' % self.iter_idx, (self.avatar_net.init_points + gs_render['offset']).cpu().numpy())


        self.avatar_net.eval()
        img_factor = self.opt['train'].get('eval_img_factor', 1.0)
        # training data
        # pose_idx, view_idx = self.opt['train'].get('eval_training_ids', (310, 19))
        pose_idx, view_idx = (np.random.randint(200)*2, 7)
        intr = self.dataset.intr_mats[view_idx].copy()
        intr[:2] *= img_factor
        item = self.dataset.getitem(0,
                                    pose_idx = pose_idx,
                                    view_idx = view_idx,
                                    training = True,
                                    eval = False,
                                    img_h = int(self.dataset.img_heights[view_idx] * img_factor),
                                    img_w = int(self.dataset.img_widths[view_idx] * img_factor),
                                    extr = self.dataset.extr_mats[view_idx],
                                    intr = intr,
                                    exact_hand_pose = True)
        items = net_util.to_cuda(item, add_batch = False)
        test_and_log(items, "training_1", img_factor)

        # training data
        pose_idx, view_idx = self.opt['train'].get('eval_testing_ids', (310, 19))
        item = self.dataset.getitem(0,
                                    pose_idx = pose_idx,
                                    view_idx = view_idx,
                                    training = True,
                                    eval = False,
                                    img_h = int(self.dataset.img_heights[view_idx] * img_factor),
                                    img_w = int(self.dataset.img_widths[view_idx] * img_factor),
                                    extr = self.dataset.extr_mats[view_idx],
                                    intr = intr,
                                    exact_hand_pose = True)
        items = net_util.to_cuda(item, add_batch = False)
        test_and_log(items, "training_2", img_factor)
       
        self.avatar_net.train()

    def body_loss(self, gaussian_cloth_pos, gaussian_body_pos, gaussian_body_normal, eps=5e-2):
        body_vertices = gaussian_body_pos.detach()
        # find the nn index of cloth on body
        nn_result = knn_points(gaussian_cloth_pos.detach().unsqueeze(0), body_vertices.unsqueeze(0), K=10, return_nn=True)
        nn_list = torch.reshape(nn_result.idx.squeeze(0), (-1, ))
        nn_points = nn_result.knn.squeeze(0).mean(dim=1)
        nn_normals = torch.reshape(gaussian_body_normal[nn_list], (-1, 10 ,3)).mean(dim=1)
        nn_normals = torch.nn.functional.normalize(nn_normals, dim=1)
        distance = ((gaussian_cloth_pos - nn_points) * nn_normals).sum(dim=-1)
        interpenetration = torch.maximum(eps - distance, torch.FloatTensor([0]).to(config.device))
        interpenetration = interpenetration.pow(2) 
        interpenetration = interpenetration[interpenetration > 0]
        loss = interpenetration.mean(-1)
        return loss
    
    def gen_output_dir(self, mode, exp_name, dataset, iter_idx):
        output_dir = self.opt[mode].get('output_dir', None)
        if output_dir is None:
            view_setting = config.opt[mode].get('view_setting', 'free')
            if view_setting == 'camera':
                view_folder = 'cam_%04d' % config.opt[mode]['render_view_idx']
            else:
                view_folder = view_setting + '_view'
            output_dir = f'./{mode}_results/{dataset.subject_name}/{exp_name}/{view_folder}' + '/batch_%06d' % iter_idx
        return output_dir

    def get_global_orient_center(self):
        item_0 = self.dataset.getitem(0, training = False)
        object_center = item_0['live_bounds'].mean(0)
        global_orient = item_0['global_orient'].cpu().numpy() if isinstance(item_0['global_orient'], torch.Tensor) else item_0['global_orient']
        global_orient = cv.Rodrigues(global_orient)[0]
        return global_orient, object_center

    def get_render_setting(self, view_setting, mode, object_center, global_orient, idx):
        img_scale = self.opt['test'].get('img_scale', 1.0)
        view_setting = config.opt['test'].get('view_setting', 'free')
        if view_setting == 'camera':
            # training view setting
            cam_id = config.opt['test']['render_view_idx']
            intr = self.dataset.intr_mats[cam_id].copy()
            intr[:2] *= img_scale
            extr = self.dataset.extr_mats[cam_id].copy()
            img_h, img_w = int(self.dataset.img_heights[cam_id] * img_scale), int(self.dataset.img_widths[cam_id] * img_scale)
        elif view_setting.startswith('free'):
            # free view setting
            # frame_num_per_circle = 360
            frame_num_per_circle = 220
            idx += 854
            rot_Y = -(idx % frame_num_per_circle) / float(frame_num_per_circle) * 2 * np.pi

            extr = visualize_util.calc_free_mv(object_center,
                                                tar_pos = np.array([0, 0, 2.5]),
                                                rot_Y = rot_Y,
                                                rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                global_orient = global_orient if self.opt[mode].get('global_orient', False) else None)
            intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
            intr[:2] *= img_scale
            img_h = int(1024 * img_scale)
            img_w = int(1024 * img_scale)
        elif view_setting.startswith('front'):
            # front view setting
            extr = visualize_util.calc_free_mv(object_center,
                                                tar_pos = np.array([0, 0, 2.5]),
                                                rot_Y = 0.,
                                                rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                global_orient = global_orient if self.opt[mode].get('global_orient', False) else None)
            intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
            intr[:2] *= img_scale
            img_h = int(1024 * img_scale)
            img_w = int(1024 * img_scale)
        elif view_setting.startswith('back'):
            # back view setting
            extr = visualize_util.calc_free_mv(object_center,
                                                tar_pos = np.array([0, 0, 2.5]),
                                                rot_Y = np.pi,
                                                rot_X = 0.5 * np.pi / 4. if view_setting.endswith('bird') else 0.,
                                                global_orient = global_orient if self.opt[mode].get('global_orient', False) else None)
            intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
            intr[:2] *= img_scale
            img_h = int(1024 * img_scale)
            img_w = int(1024 * img_scale)
        elif view_setting.startswith('moving'):
            # moving camera setting
            extr = visualize_util.calc_free_mv(object_center,
                                                # tar_pos = np.array([0, 0, 3.0]),
                                                # rot_Y = -0.3,
                                                tar_pos = np.array([0, 0, 2.5]),
                                                rot_Y = 0.,
                                                rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                global_orient = global_orient if self.opt[mode].get('global_orient', False) else None)
            intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
            intr[:2] *= img_scale
            img_h = int(1024 * img_scale)
            img_w = int(1024 * img_scale)
        elif view_setting.startswith('cano'):
            cano_center = self.dataset.cano_bounds.mean(0)
            extr = np.identity(4, np.float32)
            extr[:3, 3] = -cano_center
            rot_x = np.identity(4, np.float32)
            rot_x[:3, :3] = cv.Rodrigues(np.array([np.pi, 0, 0], np.float32))[0]
            extr = rot_x @ extr
            f_len = 5000
            extr[2, 3] += f_len / 512
            intr = np.array([[f_len, 0, 512], [0, f_len, 512], [0, 0, 1]], np.float32)
            # item = self.dataset.getitem(idx,
            #                             training = False,
            #                             extr = extr,
            #                             intr = intr,
            #                             img_w = 1024,
            #                             img_h = 1024)
            img_w, img_h = 1024, 1024
            # item['live_smpl_v'] = item['cano_smpl_v']
            # item['cano2live_jnt_mats'] = torch.eye(4, dtype = torch.float32)[None].expand(item['cano2live_jnt_mats'].shape[0], -1, -1)
            # item['live_bounds'] = item['cano_bounds']
        else:
            raise ValueError('Invalid view setting for animation!')
        return extr, intr, img_w, img_h
    def save_images(self, keys, output, output_dir, item):
        for key in keys:
            if key not in output:
                continue

            os.makedirs(output_dir + f'/{key}', exist_ok=True)
            map = output[key]
            if key == "filled_contour":
                cv.imwrite(output_dir + f'/{key}/{key}_%08d.png' % item['data_idx'], map)
            else:
                
                map.float().clip_(0., 1.)
                map = (map * 255).to(torch.uint8).cpu().numpy()
                cv.imwrite(output_dir + f'/{key}/{key}_%08d.png' % item['data_idx'], map)

    @torch.no_grad()
    def test(self):
        self.bg_color = (1., 1., 1.)
        self.bg_color_cuda = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)
        self.avatar_net.eval()

        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        # # training_dataset = MvRgbDataset(**self.opt['train']['data'], training = False)
        # # if self.opt['test'].get('n_pca', -1) >= 1:
        # #     training_dataset.compute_pca(n_components = self.opt['test']['n_pca'])
        # if 'pose_data' in self.opt['test']:
        #     testing_dataset = PoseDataset(**self.opt['test']['pose_data'], smpl_shape = training_dataset.smpl_data['betas'][0])
        # else:
        testing_dataset = MvRgbDataset(**self.opt['test']['data'], training = False, load_smpl_pos_map = True)
        self.opt['test']['n_pca'] = -1  # cancel PCA for training pose reconstruction

        self.dataset = testing_dataset
        iter_idx = self.load_ckpt(self.opt['test']['prev_ckpt'], False)[1]
        exp_name = os.path.basename(os.path.dirname(self.opt['test']['prev_ckpt']))
        output_dir = self.gen_output_dir(config.opt["mode"], exp_name, testing_dataset, iter_idx)

        use_pca = self.opt['test'].get('n_pca', -1) >= 1
        if use_pca:
            output_dir += '/pca_%d_sigma_%.2f' % (self.opt['test'].get('n_pca', -1), float(self.opt['test'].get('sigma_pca', 1.)))
        else:
            output_dir += "/{}".format(self.opt['test'].get("render_layers", ["both"])[0])
        print('# Output dir: \033[1;31m%s\033[0m' % output_dir)

        os.makedirs(output_dir + '/live_skeleton', exist_ok = True)
        os.makedirs(output_dir + '/mask_map', exist_ok = True)
        os.makedirs(output_dir + '/label_rgb_map', exist_ok=True)
        os.makedirs(output_dir + '/rgb_map', exist_ok=True)

        global_orient, object_center = self.get_global_orient_center()

        time_start = torch.cuda.Event(enable_timing = True)
        time_start_all = torch.cuda.Event(enable_timing = True)
        time_end = torch.cuda.Event(enable_timing = True)

        data_num = len(self.dataset)
        if self.opt['test'].get('fix_hand', False):
            os.makedirs(output_dir + '/label_rgb_map', exist_ok=True)
            self.avatar_net.layers_nn["body"] .generate_mean_hands()
        log_time = False
        view_setting = config.opt['test'].get('view_setting', 'free')
        
        getitem_func = self.dataset.getitem_fast if hasattr(self.dataset, 'getitem_fast') else self.dataset.getitem
        for idx in tqdm(range(data_num), desc = 'Rendering avatars...'):
            extr, intr, img_w, img_h = self.get_render_setting(view_setting, config.opt["mode"], object_center, global_orient, idx=idx)
            if log_time:
                time_start.record()
                time_start_all.record()
            item = getitem_func(
                idx,
                training = False,
                extr = extr,
                intr = intr,
                img_w = img_w,
                img_h = img_h
            )
            items = to_cuda(item, add_batch = False)
            # items["smpl_pos_map"]["body"][ :, self.avatar_net.cano_smpl_fix_offset_input_mask] = \
            #             items["smpl_cano_pos_map"]["body"][:, self.avatar_net.cano_smpl_fix_offset_input_mask]


            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Loading data costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if self.opt['test'].get('render_skeleton', False):
                from utils.visualize_skeletons import construct_skeletons
                skel_vertices, skel_faces = construct_skeletons(item['joints'].cpu().numpy(), item['kin_parent'].cpu().numpy())
                skel_mesh = trimesh.Trimesh(skel_vertices, skel_faces, process = False)

                if geo_renderer is None:
                    geo_renderer = Renderer(item['img_w'], item['img_h'], shader_name = 'phong_geometry', bg_color = (1, 1, 1))
                extr, intr = item['extr'], item['intr']
                geo_renderer.set_camera(extr, intr)
                geo_renderer.set_model(skel_vertices[skel_faces.reshape(-1)], skel_mesh.vertex_normals.astype(np.float32)[skel_faces.reshape(-1)])
                skel_img = geo_renderer.render()[:, :, :3]
                skel_img = (skel_img * 255).astype(np.uint8)
                cv.imwrite(output_dir + '/live_skeleton/%08d.jpg' % item['data_idx'], skel_img)

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering skeletons costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if 'smpl_pos_map' not in items:
                self.avatar_net.get_pose_map(items)

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering pose conditions costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            output = self.avatar_net.render(items, bg_color = self.bg_color,
                                            use_pca = use_pca, layers=self.opt['test'].get("render_layers", None))
            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering avatar costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if view_setting.startswith('moving') or view_setting == 'free_moving':
                current_center = items['live_bounds'].cpu().numpy().mean(0)
                delta = current_center - object_center

                object_center[0] += delta[0]
                extr, intr, img_w, img_h = self.get_render_setting(view_setting, config.opt["mode"], object_center, global_orient)
                
                # object_center[1] += delta[1]
                # object_center[2] += delta[2]

            rgb_map = output['rgb_map']
            rgb_map.clip_(0., 1.)
            rgb_map = (rgb_map * 255).to(torch.uint8).cpu().numpy()
            cv.imwrite(output_dir + '/rgb_map/%08d.png' % item['data_idx'], rgb_map)
        
            self.save_images(["offset_map", "mask_map", 
                              "label_rgb_map", "depth_body_map", "depth_cloth_map", 
                              "body_mask", "cloth_mask", "body_visible_mask",
                              "cloth_rgb", "body_rgb", "label_pixel_map", "label_body_rgb_map"], output, output_dir, item)
            
            if self.opt['test'].get('save_tex_map', False):
                os.makedirs(output_dir + '/cano_tex_map', exist_ok = True)
                cano_tex_map = output['cano_tex_map']
                cano_tex_map.clip_(0., 1.)
                cano_tex_map = (cano_tex_map * 255).to(torch.uint8)
                cv.imwrite(output_dir + '/cano_tex_map/%08d.jpg' % item['data_idx'], cano_tex_map.cpu().numpy())

            if self.opt['test'].get('save_ply', False):
                save_gaussians_as_ply(output_dir + '/posed_gaussians/%08d.ply' % item['data_idx'], output['posed_gaussians'])

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Saving images costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                print('Animating one frame costs %.4f secs' % (time_start_all.elapsed_time(time_end) / 1000.))

            torch.cuda.empty_cache()

    def save_ckpt(self, path, save_optm = True):
        os.makedirs(path, exist_ok = True)
        net_dict = {
            'epoch_idx': self.epoch_idx,
            'iter_idx': self.iter_idx,
            'avatar_net': self.avatar_net.state_dict(),
        }
        print('Saving networks to ', path + '/net.pt')
        torch.save(net_dict, path + '/net.pt')

        if save_optm:
            optm_dict = {
                'avatar_net': self.optm.state_dict(),
            }
            print('Saving optimizers to ', path + '/optm.pt')
            torch.save(optm_dict, path + '/optm.pt')

    def load_ckpt(self, path, load_optm = True):
        print('Loading networks from ', path + '/net.pt')
        net_dict = torch.load(path + '/net.pt')
        if 'avatar_net' in net_dict:
            self.avatar_net.load_state_dict(net_dict['avatar_net'])
        else:
            print('[WARNING] Cannot find "avatar_net" from the network checkpoint!')
        epoch_idx = net_dict['epoch_idx']
        iter_idx = net_dict['iter_idx']

        if load_optm and os.path.exists(path + '/optm.pt'):
            print('Loading optimizers from ', path + '/optm.pt')
            optm_dict = torch.load(path + '/optm.pt')
            if 'avatar_net' in optm_dict:
                self.optm.load_state_dict(optm_dict['avatar_net'])
            else:
                print('[WARNING] Cannot find "avatar_net" from the optimizer checkpoint!')

        return epoch_idx, iter_idx
    
    def load_ckpt_net(self, path, avatar_net, load_optm = True):
        print('Loading networks from ', path + '/net.pt')
        net_dict = torch.load(path + '/net.pt')
        if 'avatar_net' in net_dict:
            avatar_net.load_state_dict(net_dict['avatar_net'])
        else:
            print('[WARNING] Cannot find "avatar_net" from the network checkpoint!')
        epoch_idx = net_dict['epoch_idx']
        iter_idx = net_dict['iter_idx']
        if load_optm and os.path.exists(path + '/optm.pt'):
            print('Loading optimizers from ', path + '/optm.pt')
            optm_dict = torch.load(path + '/optm.pt')
            if 'avatar_net' in optm_dict:
                self.optm.load_state_dict(optm_dict['avatar_net'])
            else:
                print('[WARNING] Cannot find "avatar_net" from the optimizer checkpoint!')
        return epoch_idx, iter_idx
    
    @torch.no_grad()
    def exchange_cloth(self):
        self.avatar_net.eval()
        self.bg_color = (1., 1., 1.)

        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)

        testing_dataset_upper = MvRgbDataset(**self.opt['exchange_cloth']['data_upper'], training = False, load_smpl_pos_map = True)
        testing_dataset_lower = MvRgbDataset(**self.opt['exchange_cloth']['data_lower'], training = False, load_smpl_pos_map = True)
        testing_dataset_body = MvRgbDataset(**self.opt['exchange_cloth']['data_body'], training = False, load_smpl_pos_map = True)
        self.dataset_upper = testing_dataset_upper 
        self.dataset_lower = testing_dataset_lower 
        self.dataset_body = testing_dataset_body
        self.dataset = testing_dataset_body
        dataset_name = 'training'
        seq_name = config.opt.get("sequence_name", "smpl_params")
        upper_avatar_net = MultiLAvatarNet(self.opt['model'],
                                    self.opt['exchange_cloth']['data_upper'].get('layers', None), data_dir=self.opt['exchange_cloth']['data_upper']["data_dir"]).to(config.device)
        _, iter_idx = self.load_ckpt_net(self.opt['exchange_cloth']['upper_ckpt'], upper_avatar_net, False)
        lower_avatar_net = MultiLAvatarNet(self.opt['model'],
                        self.opt['exchange_cloth']['data_lower'].get('layers', None), data_dir=self.opt['exchange_cloth']['data_lower']["data_dir"]).to(config.device)
        self.load_ckpt_net(self.opt['exchange_cloth']['lower_ckpt'], lower_avatar_net, False)
        body_avatar_net = MultiLAvatarNet(self.opt['model'],
                                    self.opt['exchange_cloth']['data_body'].get('layers', None), data_dir=self.opt['exchange_cloth']['data_body']["data_dir"]).to(config.device)
        self.load_ckpt_net(self.opt['exchange_cloth']['body_ckpt'], body_avatar_net, False)
        combined_avatar_net = CombinedAvatarNet(self.opt['exchange_cloth'], body_avatar_net, upper_avatar_net, lower_avatar_net)

        output_dir = self.opt['exchange_cloth'].get('output_dir', None)
        if output_dir is None:
            view_setting = config.opt['exchange_cloth'].get('view_setting', 'free')
            view_folder = 'cam_%04d' % config.opt['exchange_cloth']['render_view_idx']
            exp_name = os.path.basename(os.path.dirname(self.opt['exchange_cloth']['body_ckpt'])) \
                + "_" + os.path.basename(os.path.dirname(self.opt['exchange_cloth']['upper_ckpt'])) \
                + "_" + os.path.basename(os.path.dirname(self.opt['exchange_cloth']['lower_ckpt'])) 
            output_dir = f'./exchange_results/{testing_dataset_body.subject_name}/{exp_name}/{dataset_name}_{seq_name}_{view_folder}' + '/batch_%06d' % iter_idx

        use_pca = self.opt['test'].get('n_pca', -1) >= 1
        if use_pca:
            output_dir += '/pca_%d_sigma_%.2f' % (self.opt['test'].get('n_pca', -1), float(self.opt['test'].get('sigma_pca', 1.)))
        else:
            output_dir += "/{}".format(self.opt['exchange_cloth'].get("render_layers", ["both"])[0])
        print('# Output dir: \033[1;31m%s\033[0m' % output_dir)

        os.makedirs(output_dir + '/live_skeleton', exist_ok = True)
        os.makedirs(output_dir + '/rgb_map', exist_ok = True)
        os.makedirs(output_dir + '/mask_map', exist_ok = True)
        os.makedirs(output_dir + '/label_rgb_map', exist_ok=True)

        geo_renderer = None
        item_0 = self.dataset.getitem(0, training = False)
        object_center = item_0['live_bounds'].mean(0)
        global_orient = item_0['global_orient'].cpu().numpy() if isinstance(item_0['global_orient'], torch.Tensor) else item_0['global_orient']
        global_orient = cv.Rodrigues(global_orient)[0]
  
        time_start = torch.cuda.Event(enable_timing = True)
        time_start_all = torch.cuda.Event(enable_timing = True)
        time_end = torch.cuda.Event(enable_timing = True)

        data_num = len(self.dataset)
    
        log_time = False

        for idx in tqdm(range(0, data_num), desc = 'Rendering avatars...'):
            if log_time:
                time_start.record()
                time_start_all.record()

            img_scale = self.opt['exchange_cloth'].get('img_scale', 1.0)
            view_setting = config.opt['exchange_cloth'].get('view_setting', 'free')

            # training view setting
            cam_id = config.opt['exchange_cloth']['render_view_idx']
            intr = self.dataset.intr_mats[cam_id].copy()
            intr[:2] *= img_scale
            extr = self.dataset.extr_mats[cam_id].copy()
            img_h, img_w = int(self.dataset.img_heights[cam_id] * img_scale), int(self.dataset.img_widths[cam_id] * img_scale)


            getitem_func_body = self.dataset_body.getitem_fast if hasattr(self.dataset, 'getitem_fast') else self.dataset_body.getitem
            item_body = getitem_func_body(
                idx,
                training = False,
                extr = extr,
                intr = intr,
                img_w = img_w,
                img_h = img_h
            )
            items_body = to_cuda(item_body, add_batch = False)
            getitem_func_upper = self.dataset_upper.getitem_fast if hasattr(self.dataset, 'getitem_fast') else self.dataset_upper.getitem
            item_upper = getitem_func_upper(
                idx,
                training = False,
                extr = extr,
                intr = intr,
                img_w = img_w,
                img_h = img_h
            )
            items_upper = to_cuda(item_upper, add_batch = False)
            if view_setting.startswith('moving') or view_setting == 'free_moving':
                current_center = items_body['live_bounds'].cpu().numpy().mean(0)
                delta = current_center - object_center

                object_center[0] += delta[0]
                # object_center[1] += delta[1]
                # object_center[2] += delta[2]

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Loading data costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering skeletons costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering pose conditions costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()
            if self.opt['exchange_cloth']["change_mode"] == "cross_change_inner":

                getitem_func_lower = self.dataset_lower.getitem_fast if hasattr(self.dataset, 'getitem_fast') else self.dataset_lower.getitem
                item_lower = getitem_func_lower(
                    idx,
                    training = False,
                    extr = extr,
                    intr = intr,
                    img_w = img_w,
                    img_h = img_h
                )
                items_lower = to_cuda(item_lower, add_batch = False)
                output = combined_avatar_net.render_filtered(items_body, items_upper, items_lower, bg_color = self.bg_color,
                                            layers=self.opt['exchange_cloth'].get("render_layers", None))
            elif self.opt['exchange_cloth']["change_mode"] == "full_change_inner":
                items_cloth = items_upper
                output = combined_avatar_net.render_full_filter(items_body, items_cloth, bg_color = self.bg_color,
                                            layers=self.opt['exchange_cloth'].get("render_layers", None))
            elif self.opt['exchange_cloth']["change_mode"] == "full_change_outer":
                items_cloth = items_upper
                output = combined_avatar_net.render_outer_filter(items_body, items_cloth, bg_color = self.bg_color,
                                            layers=self.opt['exchange_cloth'].get("render_layers", None))
            

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering avatar costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            rgb_map = output['rgb_map']
            rgb_map.clip_(0., 1.)
            rgb_map = (rgb_map * 255).to(torch.uint8).cpu().numpy()
            cv.imwrite(output_dir + '/rgb_map/%08d.png' % items_body['data_idx'], rgb_map)
        
            self.save_images(["depth_diff_map", "mask_map", "cloth_mask_map", "body_mask_map", "body_visible_mask", "label_rgb_map"], output, output_dir, item_body)
            if self.opt['test'].get('save_tex_map', False):
                os.makedirs(output_dir + '/cano_tex_map', exist_ok = True)
                cano_tex_map = output['cano_tex_map']
                cano_tex_map.clip_(0., 1.)
                cano_tex_map = (cano_tex_map * 255).to(torch.uint8)
                cv.imwrite(output_dir + '/cano_tex_map/%08d.jpg' % item_body['data_idx'], cano_tex_map.cpu().numpy())

            if self.opt['test'].get('save_ply', False):
                save_gaussians_as_ply(output_dir + '/posed_gaussians/%08d.ply' % item_body['data_idx'], output['posed_gaussians'])

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Saving images costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                print('Animating one frame costs %.4f secs' % (time_start_all.elapsed_time(time_end) / 1000.))

            torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.manual_seed(31359)
    np.random.seed(31359)
    torch.backends.cuda.preferred_linalg_library("magma")
    # torch.autograd.set_detect_anomaly(True)
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('-m', '--mode', type = str, help = 'Running mode.', default = 'train')
    args = arg_parser.parse_args()

    config.load_global_opt(args.config_path)
    if args.mode is not None:
        config.opt['mode'] = args.mode

    trainer = AvatarTrainer(config.opt)
    if config.opt['mode'] == 'train':
        if not safe_exists(config.opt['train']['net_ckpt_dir'] + '/pretrained') \
                and not safe_exists(config.opt['train']['pretrained_dir'])\
                and not safe_exists(config.opt['train']['prev_ckpt']):
            trainer.pretrain()
        trainer.train()
    elif config.opt['mode'] == 'test':
        trainer.test()
    elif config.opt['mode'] == 'exchange_cloth':
        trainer.exchange_cloth()
    else:
        raise NotImplementedError('Invalid running mode!')
