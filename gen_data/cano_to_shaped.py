import math
import os
import pickle

import torch
import numpy as np
import trimesh
import json
import argparse
import pickle
from smplxd import SMPLX

from torch import einsum
from utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# smplx_body_model_fit = SMPLX('body_models/smplx', gender='neutral', num_pca_comps=12,
#                             use_pca=True, flat_hand_mean=True, use_face_contour=False).to(device)

# smplx_body_model = SMPLX('body_models/smplx', gender='neutral', num_pca_comps=12,
#                             use_pca=True, flat_hand_mean=True, use_face_contour=False).to(device)
smplx_body_model_fit = SMPLX('smpl_files/smplx', gender='neutral',
                            use_pca=False, flat_hand_mean=True, use_face_contour=False).to(device)

smplx_body_model = SMPLX('smpl_files/smplx', gender='neutral',
                            use_pca=False, flat_hand_mean=True, use_face_contour=False).to(device)

seg = json.load(open('smplx_vert_segmentation.json'))
hand_id = torch.tensor(sorted(seg['rightHand']+ \
                              seg['leftHand'] + \
                              seg['rightHandIndex1'] + \
                              seg['leftHandIndex1'] + \
                              seg['leftEye'] + \
                              seg['rightEye'] + \
                              seg['eyeballs']), device=device)
def rotate_front(mesh_V, global_orientation):
    rot_mat = batch_rodrigues(global_orientation)
    return mesh_V.matmul(rot_mat)

def prepare_template_data_smplx(smplx_data, J_0):

    body_pose_t = torch.zeros((1, 63), device=device)
    # body_pose_t[:, 2] = torch.pi / 6
    # body_pose_t[:, 5] = -torch.pi / 6
    body_pose_t[:, 2] = math.radians(25)
    body_pose_t[:, 5] = math.radians(-25)
    #
    body_pose_t[:, 57] = -torch.pi / 2
    body_pose_t[:, 60] = -torch.pi / 2



    smplx_output = smplx_body_model(body_pose=body_pose_t,
                              betas=torch.from_numpy(smplx_data['betas']).view(1, -1).to(device),
                              transl = -J_0.view(1, 3).to(device))
    smplx_MANO_vertex_indices = pickle.load(
        open("MANO_SMPLX_vertex_ids.pkl", "rb"))
    smplx_hand_indices = np.concatenate(
        [smplx_MANO_vertex_indices["left_hand"], smplx_MANO_vertex_indices["right_hand"]])
    vertex_color = np.zeros_like(smplx_output.vertices[0].detach().cpu())
    vertex_color[smplx_hand_indices] = [255, 255, 255]
    t = trimesh.Trimesh((smplx_output.vertices + J_0.view(1, 3).to(device)).squeeze(0).detach().cpu().numpy(),
                        smplx_body_model.faces,
                        vertex_colors=vertex_color,
                        process=False)

    ply_filename = f"{args.output_dir}/smplx_beta_pose_cano.ply"
    t.export(ply_filename)

    A_template = smplx_output.A.clone().detach()
    vs_template = smplx_output.vertices.clone().detach()

    smplx_pose_cano = smplx_body_model(body_pose=body_pose_t)
    t_cano = trimesh.Trimesh(smplx_pose_cano.vertices.squeeze(0).detach().cpu().numpy(),
                        smplx_body_model.faces,
                        process=False)
    ply_filename = f"{args.output_dir}/smplx_pose_cano.ply"
    t_cano.export(ply_filename)
    return vs_template, A_template

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    mesh_data = trimesh.load(os.path.join(args.template_dir, f"template_{args.part}_offset.ply"), process=False)
    smplx = os.path.join(args.smplx_dir, "smpl_params.npz") 

    pred_smplx = dict(np.load(open(smplx, 'rb'), allow_pickle=True))

    smplx_output_shaped = smplx_body_model_fit(
        betas=torch.from_numpy(pred_smplx['betas']).view(1, -1).to(device),
    )

    smplx_output_cano = smplx_body_model_fit(
        betas=torch.zeros((10, )).view(1, -1).to(device)
    )


    # Calculate shape offset
    total_offsets = (smplx_output_shaped["shape_offsets"]).clone().float().detach()
    offsets = diffuse_offset_smplx(smplx_output_cano["vertices"], total_offsets, resolution=256)
    mesh_V = torch.from_numpy(mesh_data.vertices).float().to(device).unsqueeze(0)
    mesh_offset = query_weights(mesh_V, offsets)
    V_saved = mesh_V + mesh_offset
    t_beta = trimesh.Trimesh(V_saved.squeeze(0).detach().cpu().numpy(),
                                mesh_data.faces,
                                process=False, )

    ply_filename = f"{args.output_dir}/beta_template_{args.part}_offset.ply"
    t_beta.export(ply_filename)
    with open(f"{args.output_dir}/template_{args.part}_offset.pkl", 'wb') as f:
        pickle.dump(mesh_offset.squeeze(0).detach().cpu().numpy(), f)

    with open(f"{args.template_dir}/template_{args.part}_offset.pkl", 'wb') as f:
        pickle.dump(mesh_offset.squeeze(0).detach().cpu().numpy(), f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--template_dir', type=str, default='/data/zhiychen/AnimatableGaussain/test_data/multiviewRGC/4d_dress/upper_00134_lower_00134_body_00140/upper')
    parser.add_argument('-p', '--part', type=str, default='outer')
    parser.add_argument('-o', '--output_dir', type=str, default='shaped_template')
    parser.add_argument('-s', '--smplx_dir', type=str, default='/data/zhiychen/AnimatableGaussain/test_data/multiviewRGC/4d_dress/00140/Inner')
    args = parser.parse_args()
    main(args)