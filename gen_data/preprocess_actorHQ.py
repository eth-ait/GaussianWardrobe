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
from scipy.spatial.transform import Rotation
import csv
from pytorch3d.transforms import axis_angle_to_matrix
device=torch.device('cuda:0')
SUBJECT = "Actor05"
CAMERA =  "Cam007"
SEQUENCE = 1
SURFACE_LABEL = ['skin', 'upper', 'lower', 'hair', 'shoe', 'outer']
SURFACE_LABEL_COLOR = np.array([[128, 128, 128], [180, 50, 50], [128, 0, 255], [255, 128, 0], [50, 180, 50], [0, 128, 255]])
SURFACE_LABEL_GRAY = RGB2GRAY(SURFACE_LABEL_COLOR)
MASK_LABEL = dict(zip(SURFACE_LABEL, SURFACE_LABEL_GRAY))
camera_list = ["Cam007", "Cam022", "Cam039", "Cam056", "Cam079", "Cam080", "Cam095",
                    "Cam109", "Cam112", "Cam126", "Cam127", "Cam128",
                    "Cam151", "Cam152", "Cam160"]
data_dir = "/data/zhiychen/ActorHQ/"
new_data_dir = "/data/zhiychen/ActorHQ/"
def filter_camera(camera_fn, camera_list):
    focal_len = []
    principal_point = []
    R = []
    T = []
    image_size = []
    with open(camera_fn) as csvfile:

        reader = csv.DictReader(csvfile)

        for row in reader:
            if row["name"] not in camera_list:
                continue
            focal_len.append([float(row["fx"]) * float(row["w"]), float(row["fy"]) * float(row["h"])])
            image_size.append((int(row["h"]), int(row["w"])))
            principal_point.append([int(float(row["px"]) * float(row["w"])), int(float(row["py"]) * float(row["h"]))])
            axis_angle = [float(row["rx"]), float(row["ry"]), float(row["rz"])]
            transl = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
            extr_matrix = np.linalg.inv(extrinsic_matrix_cam2world(axis_angle, transl))
            R.append(extr_matrix[:3, :3].T)
            T.append(extr_matrix[:3, 3])
        # R = torch.tensor(R)
        # R = torch.transpose(R, 1, 2)
  
    return torch.tensor(focal_len), torch.tensor(principal_point), torch.tensor(R), torch.tensor(T), image_size

def extrinsic_matrix_cam2world(r, t) -> np.array:
    """Set up the camera to world transformation matrix to be applied on homogeneous coordinates.

    Returns:transform them accordingly before using the PyTorch3D renderer.
        np.array (4 x 4): Transformation matrix going from camera to world space.
    """
    tfm_cam2world = np.eye(4)
    tfm_cam2world[:3, :3] = rotation_matrix_cam2world(r)
    tfm_cam2world[:3, 0] *= -1
    tfm_cam2world[:3, 1] *= -1 
    tfm_cam2world[:3, 3] = t

    return tfm_cam2world

def rotation_matrix_cam2world(r) -> np.array:
    """Set up the camera to world rotation matrix from the axis-angle representation.

    Returns:
        np.array (3 x 3): Rotation matrix going from camera to world space.
    """
    return Rotation.from_rotvec(r).as_matrix()

# load pytorch3d cameras from parameters: intrinsics, extrinsics
def load_pytorch_cameras(camera_fn):
    focal_len, principal_point, R, T, image_size = filter_camera(camera_fn, camera_list)
    # assign Pytorch3d RasterizationSettings
    raster_settings = RasterizationSettings(image_size=image_size[0], blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=80000)
    camera_agents = {}
    for i in range(len(camera_list)):
        camera = PerspectiveCameras(
                    focal_length=focal_len[i].unsqueeze(0),
                    principal_point=principal_point[i].unsqueeze(0), 
                    R=R[i].unsqueeze(0),
                    T=T[i].unsqueeze(0), 
                    in_ndc=False,
                    image_size=torch.tensor(image_size[i:i+1]), device=device)
        camera_agents[camera_list[i]] = camera
    
    return camera_agents, raster_settings

def render_mesh(mesh_path, new_data_dir):
    mesh_list = glob.glob(
        os.path.join(mesh_path, '*.ply'),
        recursive=True)
    mesh_list.sort()
    os.makedirs(os.path.join(new_data_dir, SUBJECT, f"Sequence{SEQUENCE}", "4x", "body_masks", CAMERA), exist_ok=True)
    camera_path = os.path.join(new_data_dir, SUBJECT, f"Sequence{SEQUENCE}", "4x", "calibration.csv")
    camera_dict, raster_settings = load_pytorch_cameras(camera_path)
    capture_rasts = MeshRasterizer(cameras=camera_dict[CAMERA], raster_settings=raster_settings)
    for ind in tqdm(range(0, len(mesh_list), 2)):
        mesh_path = mesh_list[ind]
        ind_frame = os.path.basename(mesh_path).split(".")[0]
        mesh = trimesh.load(mesh_path)
        scan_mesh = Meshes(torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0),
                           torch.tensor(mesh.faces, dtype=torch.long).unsqueeze(0)).cuda()
        # render capture_mask(h, w) and capture_mask_image(h, w)
        capture_mask = capture_rasts(scan_mesh).pix_to_face[0, :, :, 0] > -1
        capture_mask = capture_mask.unsqueeze(-1).expand(-1, -1, 3)
        cv.imwrite(os.path.join(new_data_dir, SUBJECT, f"Sequence{SEQUENCE}", "4x", "body_masks", CAMERA, f'{ind_frame}.png'), (capture_mask * 255).detach().cpu().numpy())

def copy_filter_label(new_data_dir):
    for camera in camera_list:
        png_list = glob.glob(
        os.path.join(data_dir, SUBJECT,  f"Sequence{SEQUENCE}", "4x", "labels_unfiltered", camera, 'label*.png'),
        recursive=True)
        os.makedirs(os.path.join(new_data_dir, SUBJECT, f"Sequence{SEQUENCE}", "4x", "labels", camera), exist_ok=True)
        for png_path in png_list:
            ind_frame = os.path.basename(png_path).split(".")[0]
            new_path = os.path.join(os.path.join(new_data_dir, SUBJECT, f"Sequence{SEQUENCE}", "4x", "labels", camera), f'{ind_frame}.png')
            img = cv.imread(png_path, cv.IMREAD_GRAYSCALE)
            new_img = np.full((img.shape[0], img.shape[1], 3), 128)
            new_img[img != 255] = 0
            new_img[img == MASK_LABEL["upper"]] = 255
            # new_img[img == MASK_LABEL["upper"]] = 255
            
            cv.imwrite(new_path, new_img)         
if __name__ == '__main__':

    mesh_path = os.path.join(new_data_dir,  SUBJECT, f"Sequence{SEQUENCE}", "4x",  "body_mesh")
    # render_mesh(mesh_path, new_data_dir)
    copy_filter_label(new_data_dir)
    
