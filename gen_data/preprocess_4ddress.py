import glob
import os
import pickle
import shutil
import torch
import numpy
import numpy as np
import cv2 as cv
import sys
import trimesh
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
import json
from pytorch3d.structures import Meshes
sys.path.append("/local/home/zhiychen/AnimatableGaussain")
from utils.sh_utils import RGB2GRAY
from tqdm import tqdm
import re

SURFACE_LABEL = ['skin', 'upper', 'lower', 'hair', 'shoe', 'outer']
SURFACE_LABEL_COLOR = np.array([[127, 127, 127], [255, 128, 0], [128, 0, 255], [180, 50, 50], [50, 180, 50], [0, 128, 255]])
SURFACE_LABEL_GRAY = RGB2GRAY(SURFACE_LABEL_COLOR)
MASK_LABEL = dict(zip(SURFACE_LABEL, SURFACE_LABEL_GRAY))
RGB_LABEL = dict(zip(SURFACE_LABEL, SURFACE_LABEL_COLOR))
camera_list = [4, 16, 28, 36, 41, 48, 52, 60, 69, 76, 84, 92, 89, 21]
# data_dir = "/mnt/work/Clothes4D/SMLCHumans/"
data_dir = "/mnt/work/Clothes4D/SMLCHumans/"
generated_data_dir = "/data/zhiychen/4ddress/"

def get_mesh_indices(mesh_folder):
    """Extract indices from mesh filenames like mesh-f00011_smplx.pkl"""
    pattern = re.compile(r"mesh-f(\d{5})_smplx\.pkl")
    indices = set()

    for file_name in os.listdir(mesh_folder):
        match = pattern.match(file_name)
        if match:
            index = int(match.group(1))
            indices.add(index)

    return indices

def get_matching_files_by_index(folder, valid_indices, prefix, extension=".png"):
    """
    Get files like <prefix>-f00011<extension> if index is in valid_indices.
    Example: capture-f00011.png
    """
    pattern = re.compile(rf"{re.escape(prefix)}-f(\d{{5}}){re.escape(extension)}")
    matching_files = []

    for file_name in os.listdir(folder):
        match = pattern.match(file_name)
        if match:
            index = int(match.group(1))
            if index in valid_indices:
                matching_files.append(os.path.join(folder, file_name))

    return sorted(matching_files)

def read_all_png_camera(data_dir, png_type="images", png_prefix=""):
    png_list = [] 
       
    for take in TAKES:
        if png_type != "labels":
            valid_indices = get_mesh_indices(os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "SMPLX"))
            png_list += get_matching_files_by_index(
                os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "Capture", png_type, '%04d' % CAMERA), 
                valid_indices, png_prefix)
        else:
            png_list += glob.glob(
                os.path.join(data_dir, '%05d' % SUBJECT, LAYER,  f'Take{take}', "Capture", '%04d' % CAMERA, png_type, "*.png"),
                recursive=True)
    # png_list += glob.glob(
    #     os.path.join(data_dir, '%05d' % SUBJECT, LAYER,  '%04d' % CAMERA, png_type, "*.png"),
    #     recursive=True)
    png_list.sort()
    return png_list

def read_all_pose(data_dir):
    pose_list = []
    for take in TAKES:
        pose_list += glob.glob(
            os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "SMPLX", '*.pkl'),
            recursive=True)
    pose_list.sort()
    return pose_list
def read_all_mesh(data_dir):
    mesh_list = []
    for take in TAKES:
        mesh_list += glob.glob(
            os.path.join(data_dir, '%05d' % SUBJECT, LAYER, f'Take{take}', "Semantic", "clothes", '*.ply'),
            recursive=True)
    mesh_list.sort()
    return mesh_list 

def copy_filter_label(new_data_dir, png_list, png_type="labels"):
    if not os.path.exists(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, png_type)):
        os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, png_type))
    for ind in range(len(png_list)):
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, png_type, f'{ind:05d}.png')
        img = cv.imread(png_list[ind], cv.IMREAD_GRAYSCALE)
        new_img = cv.imread(png_list[ind], cv.IMREAD_UNCHANGED)
        if LAYER == "Outer":
            new_img[img == MASK_LABEL["hair"], :] = 0
            new_img[img == MASK_LABEL["shoe"], :] = 0
            new_img[img == 128, :] = 0
            new_img[img == 126, :] = 0
        else:
            new_img[img == MASK_LABEL["hair"], :] = MASK_LABEL["skin"]
            new_img[img == MASK_LABEL["shoe"], :] = MASK_LABEL["skin"]

        # new_img[img == MASK_LABEL["outer"]] = 255
        # new_img[img == MASK_LABEL["upper"]] = 255
        # new_img[img == MASK_LABEL["lower"]] = 255
        # new_img = np.full((img.shape[0], img.shape[1], 3), 128)
        # new_img[img != 255] = 0
        # new_img[img == MASK_LABEL["upper"]] = 255
        # new_img[img == MASK_LABEL["lower"]] = 255
        # new_img[img == MASK_LABEL["outer"]] = 255

        # new_img = np.full((img.shape[0], img.shape[1]), 255)
        # new_img[img != 255] = 0 * 255
        # new_img[img == MASK_LABEL["upper"]] = 1/3 * 255
        # new_img[img == MASK_LABEL["lower"]] = 2/3 * 255
        cv.imwrite(new_path, new_img) 


def copy_to_folder(new_data_dir, copy_list, folder_list, file_type):
    os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, *folder_list), exist_ok=True)
    for ind in range(len(copy_list)):
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, *folder_list, f'{ind:05d}.{file_type}')
        shutil.copy(copy_list[ind], new_path)

def combine_pose_to_npz(new_data_dir, sequence_name="smpl_params"):
    regstr_list = []
    regstr_list += glob.glob(
        os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "pose", '*.pkl'),
        recursive=True)
    regstr_list.sort()
    smpl_data = None
    for scan in regstr_list:
        try:
            with open(scan, 'rb') as f:
                regstr = pickle.load(f)
        except:
            print('corrupted npz')

        if smpl_data is None:
            del regstr['leye_pose']
            del regstr['reye_pose']
            smpl_data = regstr
            # expand dimension to concat
            for k in smpl_data.keys():
                smpl_data[k] = np.expand_dims(smpl_data[k], 0)
        else:
            for k in smpl_data.keys():
                if k == "betas":
                    continue
                smpl_data[k] = np.concatenate((smpl_data[k], np.expand_dims(regstr[k], 0)), axis=0)
    np.savez(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, f'{sequence_name}_gender.npz'), **smpl_data)
# load pytorch3d cameras from parameters: intrinsics, extrinsics
def load_pytorch_cameras(camera_params, camera_list, image_shape):
    # init camera_dict
    camera_dict = dict()
    # process all camera within camera_list
    for camera_id in camera_list:
        # assign camera intrinsic and extrinsic matrices
        intrinsic = torch.tensor((camera_params[camera_id]["intrinsics"]), dtype=torch.float32).cuda()
        extrinsic = torch.tensor(camera_params[camera_id]["extrinsics"], dtype=torch.float32).cuda()
        # assign camera image size
        image_size = torch.tensor([image_shape[0], image_shape[1]], dtype=torch.float32).unsqueeze(0).cuda()

        # assign camera parameters
        f_xy = torch.cat([intrinsic[0:1, 0], intrinsic[1:2, 1]], dim=0).unsqueeze(0)
        p_xy = intrinsic[:2, 2].unsqueeze(0)
        R = extrinsic[:, :3].unsqueeze(0)
        T = extrinsic[:, 3].unsqueeze(0)
        # coordinate system adaption to PyTorch3D
        R[:, :2, :] *= -1.0
        # camera position in world space -> world position in camera space
        T[:, :2] *= -1.0
        R = torch.transpose(R, 1, 2)  # row-major
        # assign Pytorch3d PerspectiveCameras
        camera_dict[camera_id] = PerspectiveCameras(focal_length=f_xy, principal_point=p_xy, R=R, T=T, in_ndc=False, image_size=image_size).cuda()
    # assign Pytorch3d RasterizationSettings
    raster_settings = RasterizationSettings(image_size=image_shape, blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=80000)
    return camera_dict, raster_settings

def render_mesh(mesh_path, new_data_dir):
    mesh_list = glob.glob(
        os.path.join(mesh_path, '*.ply'),
        recursive=True)
    mesh_list.sort()
    os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, "body_masks"), exist_ok=True)
    camera_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "cameras.pkl")
    cameras = pickle.load(open(camera_path, 'rb'))
    camera_dict, raster_settings = load_pytorch_cameras(cameras, ['%04d' % CAMERA], (1280, 940))
    capture_rasts = MeshRasterizer(cameras=camera_dict['%04d' % CAMERA], raster_settings=raster_settings)
    for ind in tqdm(range(len(mesh_list))):
        mesh_path = mesh_list[ind]
        mesh = trimesh.load(mesh_path)
        scan_mesh = Meshes(torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0),
                           torch.tensor(mesh.faces, dtype=torch.long).unsqueeze(0)).cuda()
        # render capture_mask(h, w) and capture_mask_image(h, w)
        capture_mask = capture_rasts(scan_mesh).pix_to_face[0, :, :, 0] > -1
        capture_mask = capture_mask.unsqueeze(-1).expand(-1, -1, 3)
        cv.imwrite(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, "body_masks", f'{ind:05d}.png'), (capture_mask * 255).detach().cpu().numpy())
def generate_camera_pkl(camera_path, camera_list, output_path):
    with open(camera_path, 'r') as file:
        camera_dict = json.load(file)
        selected_camera_dict = {'%04d' % camera: camera_dict['%04d' % camera] for camera in camera_list}
    with open(os.path.join(output_path, '%05d' % SUBJECT, LAYER, "cameras.pkl"), 'wb') as handle:
        pickle.dump(selected_camera_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
def render_covered_body_mask(new_data_dir):
    body_mask_paths = glob.glob(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, "body_masks", "*.png"))
    body_mask_paths.sort()
    body_cloth_mask_paths = glob.glob(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, "labels", "*.png"))
    body_cloth_mask_paths.sort()
    os.makedirs(os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, "body_cover_masks"), exist_ok=True)
    for ind in range(len(body_mask_paths)):
        body_mask_path = body_mask_paths[ind]
        body_cloth_mask_path = body_cloth_mask_paths[ind]
        body_mask = (cv.imread(body_mask_path, cv.IMREAD_UNCHANGED)[:, :, 0] == 255)
        body_cloth = cv.imread(body_cloth_mask_path, cv.IMREAD_UNCHANGED)
        cloth_mask = (body_cloth[:, :, 0] == 255)
        body_covered_mask = body_mask & cloth_mask
        # save the cover mask to target folder
        new_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, '%04d' % CAMERA, "body_cover_masks", f'{ind:05d}.png')
        new_img = np.zeros(body_cloth.shape)
        new_img[body_covered_mask] = 255
        cv.imwrite(new_path, new_img) 
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--camera', type=int, default="28", help = 'camera_name')
    arg_parser.add_argument('-s', '--subject', type=str, default="00187", help='Configuration output path.')
    arg_parser.add_argument('-l', '--layer', type=str, default="Inner", help='layer')
    arg_parser.add_argument('-m', '--mode', type=str, default="test", help='Train or test moce')
    arg_parser.add_argument('-j', '--job', default="image", type=str)

    args = arg_parser.parse_args()
    global SUBJECT 
    SUBJECT = int(args.subject)
    global CAMERA 
    CAMERA = int(args.camera)
    TAKES =   [8, 9] 
    global LAYER
    LAYER = args.layer
    global new_data_dir
    new_data_dir = f"/data/zhiychen/AnimatableGaussain/{args.mode}_data/multiviewRGC/4d_dress/"
    if args.job == "image":
        image_list = read_all_png_camera(data_dir, png_prefix="capture")
        mask_list = read_all_png_camera(data_dir, png_type="masks", png_prefix="mask")
        copy_to_folder(new_data_dir, image_list, ['%04d' % CAMERA, "images"], "png")
        copy_to_folder(new_data_dir, mask_list,  ['%04d' % CAMERA, "masks"], "png")
        # label_list = read_all_png_camera(generated_data_dir, png_type="labels", png_prefix="label")
        # copy_filter_label(new_data_dir, label_list, png_type="labels")
        # mesh_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "body_mesh")
        # render_mesh(mesh_path, new_data_dir)
    elif args.job == "pose":
        pose_list = read_all_pose(data_dir)
        copy_to_folder(new_data_dir, pose_list, ["pose"], "pkl")
        combine_pose_to_npz(new_data_dir)
    all_camera_path = "/data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/rgb_cameras.json"
    generate_camera_pkl(all_camera_path, camera_list, new_data_dir)



    
