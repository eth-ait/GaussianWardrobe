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

SUBJECT_1 = 127
SUBJECT_2 = 1
LAYER = "Outer"
def exchange_beta(data_dir, output_dir):
    smpl_path_1 = os.path.join(data_dir, '%05d' % SUBJECT_1, LAYER, "smpl_params.npz")
    smpl_sub_1 = dict(np.load(smpl_path_1))
    smpl_path_2 = os.path.join(data_dir, '%05d' % SUBJECT_2, LAYER, "smpl_params.npz")
    smpl_sub_2 = dict(np.load(smpl_path_2))
    output_folder_1 = "_".join(["cloth", '%05d' % SUBJECT_1, "body", '%05d' % SUBJECT_2])
    output_folder_2 = "_".join(["body", '%05d' % SUBJECT_1, "cloth", '%05d' % SUBJECT_2])
    os.makedirs(os.path.join(output_dir, output_folder_1), exist_ok=True)
    os.makedirs(os.path.join(output_dir, output_folder_2), exist_ok=True)
    output_dir_1 = os.path.join(output_dir, output_folder_1, LAYER, "smpl_params.npz")
    output_dir_2 = os.path.join(output_dir, output_folder_2, LAYER, "smpl_params.npz")
    beta_sub_1 = smpl_sub_1["betas"]
    smpl_sub_1["betas"] = smpl_sub_2["betas"] 
    smpl_sub_2["betas"] = beta_sub_1
    np.savez(output_dir_1, **smpl_sub_2)
    np.savez(output_dir_2, **smpl_sub_1)
def update_beta(data_dir, beta_dir):
    smpl_path = os.path.join(data_dir, "00185_outer_test.npz")
    smpl_data = dict(np.load(smpl_path, allow_pickle=True))
    beta_path = os.path.join(beta_dir,  "smpl_params.npz")
    beta_data = np.load(beta_path, allow_pickle=True)["betas"]

    output_dir = os.path.join(data_dir,  "00185_outer_test.npz")
    smpl_data["betas"] = np.reshape(beta_data, (1, -1))
    np.savez(output_dir, **smpl_data)
def update_facial_hand_pose(new_pose_dir, new_pose_sequence, train_pose_dir):
    os.makedirs(new_pose_dir, exist_ok=True)

    new_pose_path = os.path.join(new_pose_dir, new_pose_sequence + ".npz")
    new_pose = dict(np.load(new_pose_path, allow_pickle=True))
    train_pose_path = os.path.join(train_pose_dir,  "smpl_params.npz")
    train_pose = dict(np.load(train_pose_path, allow_pickle=True))
    n_pose = new_pose["body_pose"].shape[0]
    use_train = ["expression", "left_hand_pose", "right_hand_pose"]


    for key in use_train:
        cur_feature = np.reshape(train_pose[key][0], (1, -1))
        new_pose[key] = np.repeat(cur_feature, n_pose, axis=0)
    

    np.savez(os.path.join(new_pose_dir, new_pose_sequence + "_fixed.npz"), **new_pose)

if __name__ == '__main__':
    train_pose_dir = "/data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/00187/Inner/"
    new_pose_dir = "/data/zhiychen/4ddress_sequence"
    new_pose_sequence = "actorhq"

    # output_dir = "/data/zhiychen/AnimatableGaussain/test_data/multiviewRGC/4d_dress/"
    # update_facial_hand_pose(new_pose_dir, new_pose_sequence, train_pose_dir)
    # update_facial_expression(data_dir, facial_expression_dir)
    
    update_beta(data_dir="/data/zhiychen/AnimatableGaussain/test_data/multiviewRGC/4d_dress/00185/Inner", beta_dir="/data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/00185/Inner")
