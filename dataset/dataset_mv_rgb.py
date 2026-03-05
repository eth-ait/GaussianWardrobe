import glob
import os
import pickle

import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
import json
from numpy.linalg import inv
import smplx
import config
import utils.nerf_util as nerf_util
import utils.visualize_util as visualize_util
import dataset.commons as commons
from utils.sh_utils import RGB2GRAY
from PIL import Image
from utils.smpl_util import smplx_model
from utils.net_util import process_pos_map
class MvRgbDatasetBase(Dataset):
    @torch.no_grad()
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs',
        layers = None,
        use_label = False,
        use_body_label = False,
        use_normal = False,
    ):
        super(MvRgbDatasetBase, self).__init__()

        self.data_dir = data_dir
        self.training = training
        self.subject_name = subject_name
        self.layers = layers
        if self.subject_name is None:
            self.subject_name = os.path.basename(self.data_dir)
        self.load_smpl_pos_map = load_smpl_pos_map
        self.load_smpl_nml_map = load_smpl_nml_map
        self.mode = mode  # '3dgs' or 'nerf'
        self.use_label = use_label
        self.use_body_label = use_body_label
        self.use_normal = use_normal
        self.load_cam_data()
        self.load_smpl_data()
        if layers is None:
            self.smpl_pos_map = config.opt.get("smpl_pos_map", "smpl_pos_map")
        else:
            self.smpl_pos_map = {}
            for layer in layers:
                self.smpl_pos_map[layer] = config.opt.get("smpl_pos_map", "smpl_pos_map") + f"_{layer}"
        self.smpl_model, self.smpl_model_fit = smplx_model(self.data_dir)

        pose_list = list(range(self.smpl_data['body_pose'].shape[0]))
        if frame_range is not None:
            if isinstance(frame_range, list):
                if len(frame_range) == 2:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]})')
                    frame_range = range(frame_range[0], frame_range[1])
                elif len(frame_range) == 3:
                    print(f'# Selected frame indices: range({frame_range[0]}, {frame_range[1]}, {frame_range[2]})')
                    frame_range = range(frame_range[0], frame_range[1], frame_range[2])
            elif isinstance(frame_range, str):
                frame_range = np.loadtxt(self.data_dir + '/' + frame_range).astype(np.int).tolist()
                print(f'# Selected frame indices: {frame_range}')
            else:
                raise TypeError('Invalid frame_range!')
            self.pose_list = list(frame_range)
        else:
            self.pose_list = pose_list

        if self.training:
            if used_cam_ids is None:
                self.used_cam_ids = list(range(self.view_num))
            else:
                self.used_cam_ids = used_cam_ids
            print('# Used camera ids: ', self.used_cam_ids)
            self.data_list = []
            for pose_idx in self.pose_list:
                for view_idx in self.used_cam_ids:
                    self.data_list.append((pose_idx, view_idx))
            # filter missing files
            self.filter_missing_files()

        print('# Dataset contains %d items' % len(self))

        # SMPL related
        ret = self.smpl_model.forward(betas = self.smpl_data['betas'][0][None],
                                      global_orient = config.cano_smpl_global_orient[None],
                                      transl = config.cano_smpl_transl[None],
                                      body_pose = config.cano_smpl_body_pose[None])
        self.cano_smpl = {k: v[0] for k, v in ret.items() if isinstance(v, torch.Tensor)}
        self.inv_cano_jnt_mats = torch.linalg.inv(self.cano_smpl['A'])
        min_xyz = self.cano_smpl['vertices'].min(0)[0]
        max_xyz = self.cano_smpl['vertices'].max(0)[0]
        self.cano_smpl_center = 0.5 * (min_xyz + max_xyz)
        min_xyz[:2] -= 0.05
        max_xyz[:2] += 0.05
        min_xyz[2] -= 0.15
        max_xyz[2] += 0.15
        self.cano_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        self.smpl_faces = self.smpl_model.faces.astype(np.int32)

        commons._initialize_hands(self)

    def __len__(self):
        if self.training:
            return len(self.data_list)
        else:
            return len(self.pose_list)

    def __getitem__(self, index):
        return self.getitem(index, self.training)

    def getitem(self, index, training = True, **kwargs):
        if training or kwargs.get('eval', False):  # training or evaluation
            pose_idx, view_idx = self.data_list[index]
            pose_idx = kwargs['pose_idx'] if 'pose_idx' in kwargs else pose_idx
            view_idx = kwargs['view_idx'] if 'view_idx' in kwargs else view_idx
            data_idx = (pose_idx, view_idx)
            if not training:
                print('data index: (%d, %d)' % (pose_idx, view_idx))
        else:  # testing
            pose_idx = self.pose_list[index]
            data_idx = pose_idx
            print('data index: %d' % pose_idx)

        # SMPL
        with torch.no_grad():
            live_smpl = self.smpl_model_fit.forward(
                betas = self.smpl_data['betas'][0][None],
                global_orient = self.smpl_data['global_orient'][pose_idx][None],
                transl = self.smpl_data['transl'][pose_idx][None],
                body_pose = self.smpl_data['body_pose'][pose_idx][None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
                left_hand_pose = self.smpl_data['left_hand_pose'][pose_idx][None],
                right_hand_pose = self.smpl_data['right_hand_pose'][pose_idx][None]
            )
            cano_smpl = self.smpl_model.forward(
                betas = self.smpl_data['betas'][0][None],
                global_orient = config.cano_smpl_global_orient[None],
                transl = config.cano_smpl_transl[None],
                body_pose = config.cano_smpl_body_pose[None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
            )
            live_smpl_woRoot = self.smpl_model_fit.forward(
                betas = self.smpl_data['betas'][0][None],
                body_pose = self.smpl_data['body_pose'][pose_idx][None],
                jaw_pose = self.smpl_data['jaw_pose'][pose_idx][None],
                expression = self.smpl_data['expression'][pose_idx][None],
                left_hand_pose=self.smpl_data['left_hand_pose'][pose_idx][None],
                right_hand_pose=self.smpl_data['right_hand_pose'][pose_idx][None]
            )

        data_item = dict()
        if self.load_smpl_pos_map:
            if self.layers is None:
                smpl_pos_map = process_pos_map(self.data_dir + '/{}/%08d.exr'.format(self.smpl_pos_map) % pose_idx)
                data_item['smpl_pos_map'] = smpl_pos_map
            else:
                smpl_pos_maps = {}
                smpl_cano_pos_maps = {}
                for layer in self.layers:
                    smpl_pos_map = process_pos_map(self.data_dir + '/{}/%08d.exr'.format(self.smpl_pos_map[layer]) % pose_idx)
                    smpl_pos_maps[layer] = smpl_pos_map

                    smpl_cano_pos_map = process_pos_map(self.data_dir + '/{}/cano_smpl_pos_map.exr'.format(self.smpl_pos_map[layer]))
                    smpl_cano_pos_maps[layer] = smpl_cano_pos_map
                data_item['smpl_cano_pos_map'] = smpl_cano_pos_maps
                data_item['smpl_pos_map'] = smpl_pos_maps
                


        if self.load_smpl_nml_map:
            smpl_nml_map = cv.imread(self.data_dir + '/smpl_nml_map/%08d.jpg' % pose_idx, cv.IMREAD_UNCHANGED)
            smpl_nml_map = (smpl_nml_map / 255.).astype(np.float32)
            nml_map_size = smpl_nml_map.shape[1] // 2
            smpl_nml_map = np.concatenate([smpl_nml_map[:, :nml_map_size], smpl_nml_map[:, nml_map_size:]], 2)
            smpl_nml_map = smpl_nml_map.transpose((2, 0, 1))
            data_item['smpl_nml_map'] = smpl_nml_map

        data_item['joints'] = live_smpl.joints[0, :22]
        data_item['kin_parent'] = self.smpl_model.parents[:22].to(torch.long)
        data_item['item_idx'] = index
        data_item['data_idx'] = data_idx
        data_item['time_stamp'] = np.array(pose_idx, np.float32)
        data_item['global_orient'] = self.smpl_data['global_orient'][pose_idx]
        data_item['transl'] = self.smpl_data['transl'][pose_idx]
        data_item['live_smpl_v'] = live_smpl.vertices[0]
        data_item['live_smpl_v_woRoot'] = live_smpl_woRoot.vertices[0]
        data_item['cano_smpl_v'] = cano_smpl.vertices[0]
        data_item['cano_jnts'] = cano_smpl.joints[0]
        data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], torch.linalg.inv(cano_smpl.A[0]))
        data_item['cano2live_jnt_mats_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], torch.linalg.inv(cano_smpl.A[0]))
        data_item['cano_smpl_center'] = self.cano_smpl_center
        data_item['cano_bounds'] = self.cano_bounds
        data_item['smpl_faces'] = self.smpl_faces
        min_xyz = live_smpl.vertices[0].min(0)[0] - 0.15
        max_xyz = live_smpl.vertices[0].max(0)[0] + 0.15
        live_bounds = torch.stack([min_xyz, max_xyz], 0).to(torch.float32).numpy()
        data_item['live_bounds'] = live_bounds

        if training:
            color_img, mask_img = self.load_color_mask_images(pose_idx, view_idx)

            color_img = (color_img / 255.).astype(np.float32)

            boundary_mask_img, mask_img = self.get_boundary_mask(mask_img)

            if self.mode == '3dgs':
                data_item.update({
                    'img_h': color_img.shape[0],
                    'img_w': color_img.shape[1],
                    'extr': self.extr_mats[view_idx],
                    'intr': self.intr_mats[view_idx],
                    'color_img': color_img,
                    'mask_img': mask_img,
                    'boundary_mask_img': boundary_mask_img
                })
                if self.use_label:
                    label_img = (self.load_image(pose_idx, view_idx, "labels_color") / 255.).astype(np.float32)
                    data_item["label_img"] = label_img
                if self.use_body_label:
                    label_body_img = (self.load_image(pose_idx, view_idx, "body_masks") / 255.).astype(np.float32)
                    data_item["label_body_img"] = label_body_img
                if self.use_normal:
                    normal_img = (self.load_image(pose_idx, view_idx, "normals") / 255.).astype(np.float32)
                    data_item["normal_img"] = normal_img
                joints_3d = live_smpl["joints"][:, [config.JOINT_MAPPER["head"]]]
                extrinsic = self.extr_mats[view_idx]
                intrinsic = self.intr_mats[view_idx]
                projection_matrices = torch.tensor(intrinsic @ extrinsic[:3])
                p = torch.einsum("ij,mnj->mni", projection_matrices[:3, :3], joints_3d) + projection_matrices[:3, 3]
                p = p[..., :2] / p[..., 2:3]
                data_item["head_pixel"] = p[0].squeeze(0)
            elif self.mode == 'nerf':
                depth_img = np.zeros(color_img.shape[:2], np.float32)
                nerf_random = nerf_util.sample_randomly_for_nerf_rendering(
                    color_img, mask_img, depth_img,
                    self.extr_mats[view_idx], self.intr_mats[view_idx],
                    live_bounds,
                    unsample_region_mask = boundary_mask_img
                )
                data_item.update({
                    'nerf_random': nerf_random,
                    'extr': self.extr_mats[view_idx],
                    'intr': self.intr_mats[view_idx]
                })
            else:
                raise ValueError('Invalid dataset mode!')
        else:
            """ synthesis config """
            img_h = 512 if 'img_h' not in kwargs else kwargs['img_h']
            img_w = 512 if 'img_w' not in kwargs else kwargs['img_w']
            intr = np.array([[550, 0, 256], [0, 550, 256], [0, 0, 1]], np.float32) if 'intr' not in kwargs else kwargs['intr']
            if 'extr' not in kwargs:
                extr = visualize_util.calc_front_mv(live_bounds.mean(0), tar_pos = np.array([0, 0, 2.5]))
            else:
                extr = kwargs['extr']

            data_item.update({
                'img_h': img_h,
                'img_w': img_w,
                'extr': extr,
                'intr': intr
            })

        if self.mode == 'nerf' or self.mode == '3dgs' and not training:
            # mano
            data_item['left_cano_mano_v'], data_item['left_cano_mano_n'], data_item['right_cano_mano_v'], data_item['right_cano_mano_n'] \
                = commons.generate_two_manos(self, self.cano_smpl['vertices'])
            data_item['left_live_mano_v'], data_item['left_live_mano_n'], data_item['right_live_mano_v'], data_item['right_live_mano_n'] \
                = commons.generate_two_manos(self, live_smpl.vertices[0])

        return data_item

    def load_cam_data(self):
        """
        Initialize:
        self.cam_names, self.view_num, self.extr_mats, self.intr_mats,
        self.img_widths, self.img_heights
        """
        raise NotImplementedError

    def load_smpl_data(self):
        """
        Initialize:
        self.smpl_data, a dict including ['body_pose', 'global_orient', 'transl', 'betas', ...]
        """
        smpl_data = np.load(self.data_dir + f'/{config.opt.get("sequence_name", "smpl_params")}.npz', allow_pickle = True)
        smpl_data_raw = dict(smpl_data)
        smpl_data = {}
        for key in smpl_data_raw:
            if smpl_data_raw[key].dtype == "float64" or smpl_data_raw[key].dtype == "float32":
                smpl_data[key] = smpl_data_raw[key]
        self.smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

    def filter_missing_files(self):
        pass

    def load_color_mask_images(self, pose_idx, view_idx):
        raise NotImplementedError
    
    def load_image(self, pose_idx, view_idx, folder_name):
        raise NotImplementedError
    

    
    @staticmethod
    def get_boundary_mask(mask, kernel_size = 3):
        """
        :param mask: np.uint8
        :param kernel_size:
        :return:
        """
        mask_bk = mask.copy()
        thres = 128
        mask[mask < thres] = 0
        mask[mask > thres] = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_erode = cv.erode(mask.copy(), kernel)
        mask_dilate = cv.dilate(mask.copy(), kernel)
        boundary_mask = (mask_dilate - mask_erode) == 1
        boundary_mask = np.logical_or(boundary_mask,
                                      np.logical_and(mask_bk > 5, mask_bk < 250))

        # boundary_mask_resized = cv.resize(boundary_mask.astype(np.uint8), (0, 0), fx = 0.5, fy = 0.5)
        # cv.imshow('boundary_mask', boundary_mask_resized.astype(np.uint8) * 255)
        # cv.waitKey(0)

        return boundary_mask, mask == 1

    def compute_pca(self, n_components = 10):
        from sklearn.decomposition import PCA
        from tqdm import tqdm
        import joblib

        if not os.path.exists(self.data_dir + '/{}/pca_%d.ckpt'.format(self.smpl_pos_map) % n_components):
            pose_conds = []
            mask = None
            for pose_idx in tqdm(self.pose_list, desc = 'Loading position maps...'):
                pose_map = cv.imread(self.data_dir + '/{}/%08d.exr'.format(self.smpl_pos_map) % pose_idx, cv.IMREAD_UNCHANGED)
                pose_map = pose_map[:, :pose_map.shape[1] // 2]
                if mask is None:
                    mask = np.linalg.norm(pose_map, axis = -1) > 1e-6
                pose_conds.append(pose_map[mask])
            pose_conds = np.stack(pose_conds, 0)
            pose_conds = pose_conds.reshape(pose_conds.shape[0], -1)
            self.pca = PCA(n_components = n_components)
            self.pca.fit(pose_conds)
            joblib.dump(self.pca, self.data_dir + '/{}/pca_%d.ckpt'.format(self.smpl_pos_map) % n_components)
            self.pos_map_mask = mask
        else:
            self.pca = joblib.load(self.data_dir + '/{}/pca_%d.ckpt'.format(self.smpl_pos_map) % n_components)
            pose_map = cv.imread(sorted(glob.glob(self.data_dir + '/smpl_pos_map{}/0*.exr'.format(self.smpl_pos_map)))[0], cv.IMREAD_UNCHANGED)
            pose_map = pose_map[:, :pose_map.shape[1] // 2]
            self.pos_map_mask = np.linalg.norm(pose_map, axis = -1) > 1e-6

    def transform_pca(self, pose_conds, sigma_pca = 2.):
        pose_conds = pose_conds.reshape(1, -1)
        lowdim_pose_conds = self.pca.transform(pose_conds)
        std = np.sqrt(self.pca.explained_variance_)
        lowdim_pose_conds = np.maximum(lowdim_pose_conds, -sigma_pca * std)
        lowdim_pose_conds = np.minimum(lowdim_pose_conds, sigma_pca * std)
        new_pose_conds = self.pca.inverse_transform(lowdim_pose_conds)
        new_pose_conds = new_pose_conds.reshape(-1, 3)
        return new_pose_conds

class MvRgbDataset4DDress(MvRgbDatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs',
        layers = None,
        use_label = False,
        use_body_label = False,
        use_normal = False,
    ):
        super(MvRgbDataset4DDress, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            training,
            subject_name,
            load_smpl_pos_map,
            load_smpl_nml_map,
            mode,
            layers,
            use_label,
            use_body_label,
            use_normal,
        )
        if subject_name is None:
            self.subject_name = os.path.basename(os.path.dirname(self.data_dir))
        if layers is not None:
            self.surface_label = ['lower']
            self.surface_label_color = RGB2GRAY([[128, 0, 255]])
            self.masklabel = dict(zip(self.surface_label, self.surface_label_color))


    def load_cam_data(self):
        import csv
        cam_names = []
        extr_mats = []
        intr_mats = []
        img_widths = []
        img_heights = []
        with open(os.path.join(self.data_dir, "cameras.pkl"), "rb") as fp:
            cameras = pickle.load(fp)
            for cam_name in cameras.keys():
                cam_names.append(cam_name)
                camera = cameras[cam_name]
                img_widths.append(940)
                img_heights.append(1280)

                extr_mat = np.identity(4, np.float32)
                extr_mat[:3, :] = camera["extrinsics"]
                extr_mats.append(extr_mat)

                intr_mat = np.identity(3, np.float32)
                intr_mat[:, :] = camera["intrinsics"]
                intr_mats.append(intr_mat)

        self.cam_names, self.img_widths, self.img_heights, self.extr_mats, self.intr_mats \
            = cam_names, img_widths, img_heights, extr_mats, intr_mats
        self.view_num = len(self.cam_names)

    def load_image(self, pose_idx, view_idx, folder_name):
        cam_name = self.cam_names[view_idx]
        img = cv.imread(os.path.join(self.data_dir, cam_name,
                                           folder_name, '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        return img
    
    def filter_missing_files(self):
        missing_data_list = []
        if os.path.exists((self.data_dir + '/missing_img_files.txt')):
            with open(self.data_dir + '/missing_img_files.txt', 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                line = line.replace('\\', '/')  # considering both Windows and Ubuntu file system
                frame_idx = int(os.path.basename(line).replace('.jpg', ''))
                for camera in self.used_cam_ids:
                    missing_data_list.append((frame_idx, camera))
            for missing_data_idx in missing_data_list:
                if missing_data_idx in self.data_list:
                    self.data_list.remove(missing_data_idx)

    
    def load_color_mask_images(self, pose_idx, view_idx):
        cam_name = self.cam_names[view_idx]
        color_img = cv.imread(os.path.join(self.data_dir, cam_name,
                                           "images", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        # label_img = cv.imread(os.path.join(self.data_dir, cam_name,
        #                                    "labels", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = cv.imread(os.path.join(self.data_dir, cam_name, "masks", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        if len(mask_img.shape) > 2:
            mask_img = mask_img[:, :, 0]
        # color_img = np.concatenate([color_img, np.expand_dims(label_img, axis=2)], axis=2)
        # Use partial image
        # if self.layers is None:
        #     mask_img = cv.imread(os.path.join(self.data_dir, cam_name,
        #                                     "masks", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        # else:
        #     selected_gray = []
        #     label_img = cv.imread(os.path.join(self.data_dir, cam_name,
        #                                     "labels", '%05d.png' % pose_idx), cv.IMREAD_GRAYSCALE)
        #     for layer in self.layers:
        #         selected_gray.append(self.masklabel[layer])
        #     mask_img = np.isin(label_img, selected_gray).astype(np.uint8) * 255
        return color_img, mask_img

class MvRgbDatasetOther(MvRgbDatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs',
    ):
        super(MvRgbDatasetOther, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            training,
            subject_name,
            load_smpl_pos_map,
            load_smpl_nml_map,
            mode,
        )
        if subject_name is None:
            self.subject_name = os.path.basename(os.path.dirname(self.data_dir))

    def load_cam_data(self):
        import csv
        cam_names = []
        extr_mats = []
        intr_mats = []
        img_widths = []
        img_heights = []
        cameras = glob.glob( os.path.join(self.data_dir,  "cameras", '*.json'),
            recursive=True)
        for camera_fn in cameras:
            with open(camera_fn) as fp:
                camera = json.load(fp)
                cam_names.append(camera_fn.split("/")[-1].split(".")[0])
                img_widths.append(1024)
                img_heights.append(1024)

                extr_mat = inv(camera["cam_param"])
                extr_mats.append(extr_mat)
                intr_mat = np.identity(3, np.float32)
                intr_mat[0, 0] = 100000
                intr_mat[1, 1] = 100000
                intr_mat[0, 2] = 512
                intr_mat[1, 2] = 512
                intr_mats.append(intr_mat)

        self.cam_names, self.img_widths, self.img_heights, self.extr_mats, self.intr_mats \
            = cam_names, img_widths, img_heights, extr_mats, intr_mats
        self.view_num = len(self.cam_names)

    def load_color_mask_images(self, pose_idx, view_idx):
        cam_name = self.cam_names[view_idx]
        color_img = cv.imread(os.path.join(self.data_dir, cam_name, '%05d.png' % (pose_idx + 1)), cv.IMREAD_UNCHANGED)[:,:,:3]
        # label_img = cv.imread(os.path.join(self.data_dir, cam_name,
        #                                    "labels", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = np.load(os.path.join(self.data_dir, cam_name, '%05d.npy' % (pose_idx + 1))).astype(np.float64) * 255
        if len(mask_img.shape) > 2:
            mask_img = mask_img[:, :, 0]
        # color_img = np.concatenate([color_img, np.expand_dims(label_img, axis=2)], axis=2)
        # Use partial image
        # if self.layers is None:
        #     mask_img = cv.imread(os.path.join(self.data_dir, cam_name,
        #                                     "masks", '%05d.png' % pose_idx), cv.IMREAD_UNCHANGED)
        # else:
        #     selected_gray = []
        #     label_img = cv.imread(os.path.join(self.data_dir, cam_name,
        #                                     "labels", '%05d.png' % pose_idx), cv.IMREAD_GRAYSCALE)
        #     for layer in self.layers:
        #         selected_gray.append(self.masklabel[layer])
        #     mask_img = np.isin(label_img, selected_gray).astype(np.uint8) * 255
        return color_img, mask_img
    
class MvRgbDatasetActorsHQ(MvRgbDatasetBase):
    def __init__(
        self,
        data_dir,
        frame_range = None,
        used_cam_ids = None,
        training = True,
        subject_name = None,
        load_smpl_pos_map = False,
        load_smpl_nml_map = False,
        mode = '3dgs',
        layers = None,
        use_label = False,
        use_body_label = False,
        use_normal = False,
    ):
        super(MvRgbDatasetActorsHQ, self).__init__(
            data_dir,
            frame_range,
            used_cam_ids,
            training,
            subject_name,
            load_smpl_pos_map,
            load_smpl_nml_map,
            mode,
            layers,
            use_label,
            use_body_label,
            use_normal,
        )

        if subject_name is None:
            self.subject_name = os.path.basename(os.path.dirname(self.data_dir))

    def load_cam_data(self):
        import csv
        cam_names = ["Cam000"]
        extr_mats = [np.identity(4, np.float32)]
        intr_mats = [np.identity(3, np.float32)]
        img_widths = [0]
        img_heights = [0]
        with open(self.data_dir + '/calibration.csv', "r", newline = "", encoding = 'utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                cam_names.append(row['name'])
                img_widths.append(int(row['w']))
                img_heights.append(int(row['h']))

                extr_mat = np.identity(4, np.float32)
                extr_mat[:3, :3] = cv.Rodrigues(np.array([float(row['rx']), float(row['ry']), float(row['rz'])], np.float32))[0]
                extr_mat[:3, 3] = np.array([float(row['tx']), float(row['ty']), float(row['tz'])])
                extr_mat = np.linalg.inv(extr_mat)
                extr_mats.append(extr_mat)

                intr_mat = np.identity(3, np.float32)
                intr_mat[0, 0] = float(row['fx']) * float(row['w'])
                intr_mat[0, 2] = float(row['px']) * float(row['w'])
                intr_mat[1, 1] = float(row['fy']) * float(row['h'])
                intr_mat[1, 2] = float(row['py']) * float(row['h'])
                intr_mats.append(intr_mat)

        self.cam_names, self.img_widths, self.img_heights, self.extr_mats, self.intr_mats \
            = cam_names, img_widths, img_heights, extr_mats, intr_mats

    def load_color_mask_images(self, pose_idx, view_idx):
        cam_name = f"Cam{view_idx:03d}"
        color_img = cv.imread(self.data_dir + '/rgbs/%s/%s_rgb%06d.jpg' % (cam_name, cam_name, pose_idx), cv.IMREAD_UNCHANGED)
        mask_img = cv.imread(self.data_dir + '/masks/%s/%s_mask%06d.png' % (cam_name, cam_name, pose_idx), cv.IMREAD_UNCHANGED)
        return color_img, mask_img
    
    def filter_missing_files(self):
        missing_data_list = []
        with open(self.data_dir + '/missing_img_files.txt', 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.replace('\\', '/')  # considering both Windows and Ubuntu file system
            frame_idx = int(os.path.basename(line).replace('.jpg', ''))
            for camera in self.used_cam_ids:
                missing_data_list.append((frame_idx, camera))
        for missing_data_idx in missing_data_list:
            if missing_data_idx in self.data_list:
                self.data_list.remove(missing_data_idx)

    def load_image(self, pose_idx, view_idx, folder_name):
        cam_name = f"Cam{view_idx:03d}"
        if folder_name == "labels_color":
            img = cv.imread(os.path.join(self.data_dir, '%s/%s/label-f%06d.png' % (folder_name, cam_name, pose_idx)), cv.IMREAD_UNCHANGED)
        elif folder_name == "normals":
            img = cv.imread(os.path.join(self.data_dir, '%s/%s/%s_rgb%06d.png' % (folder_name, cam_name, cam_name, pose_idx)), cv.IMREAD_UNCHANGED)
        else:
            img = cv.imread(os.path.join(self.data_dir, '%s/%s/%05d.png' % (folder_name, cam_name, pose_idx)), cv.IMREAD_UNCHANGED)
        return img