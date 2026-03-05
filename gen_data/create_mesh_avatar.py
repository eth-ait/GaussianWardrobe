import math
import os
import pickle

import torch
import numpy as np
import trimesh
import pickle as pkl
import json
import argparse
import pickle

from tqdm import tqdm

from torch import einsum
from PIL import Image
from gen_data.utils import *
from smplxd import SMPLX

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

smplx_body_model_fit = SMPLX('smpl_files/smplx', gender='neutral', num_pca_comps=12,
                            use_pca=True, flat_hand_mean=False, use_face_contour=False).to(device)

smplx_body_model = SMPLX('smpl_files/smplx', gender='neutral', num_pca_comps=12,
                            use_pca=True, flat_hand_mean=True, use_face_contour=False).to(device)
# smplx_body_model_fit = SMPLX('smpl_files/smplx', gender='neutral', use_pca=False,
#                             flat_hand_mean=False, use_face_contour=True).to(device)

# smplx_body_model = SMPLX('smpl_files/smplx', gender='neutral',
#                             use_pca=False, flat_hand_mean=True, use_face_contour=True).to(device)

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
    os.makedirs(args.output_dir, exist_ok=True)
    ply_filename = f"{args.output_dir}/smplx_beta_pose_cano.ply"
    t.export(ply_filename)

    A_template = smplx_output.A.clone().detach()
    vs_template = smplx_output.vertices.clone().detach()

    smplx_pose_cano = smplx_body_model(body_pose=body_pose_t)
    t_cano = trimesh.Trimesh(smplx_pose_cano.vertices.squeeze(0).detach().cpu().numpy(),
                        smplx_body_model.faces,
                        vertex_colors=vertex_color,
                        process=False)
    ply_filename = f"output_path/smplx_pose_cano.ply"
    t_cano.export(ply_filename)
    return vs_template, A_template
def select_smplx_pose(pred_smplx_poses, index):
    pred_smplx = {}
    for key in pred_smplx_poses:
        if key == "betas":
            pred_smplx[key] = pred_smplx_poses[key].squeeze(0)
        else:
            pred_smplx[key] = pred_smplx_poses[key][index]
    return pred_smplx
def main(args):

    smplx_beta = dict(np.load(open(os.path.join(args.smplx_dir, "smpl_params.npz"), 'rb'), allow_pickle=True))


    pred_smplx = select_smplx_pose(smplx_beta, args.index)
    label = pkl.load(open(args.label_path, 'rb'))


    J_0 = smplx_body_model_fit( betas=
            torch.from_numpy(pred_smplx['betas']).view(1, -1).to(device)).joints[:, 0, :].clone().detach()


    smplx_output = smplx_body_model_fit(
        body_pose=torch.from_numpy(pred_smplx['body_pose']).view(1, -1).to(device),
        betas=torch.from_numpy(pred_smplx['betas']).view(1, -1).to(device),
        # global_orient=torch.from_numpy(pred_smplx['global_orient']).view(1, -1).to(device),
        left_hand_pose=torch.from_numpy(pred_smplx['left_hand_pose']).view(1, -1).to(device),
        right_hand_pose=torch.from_numpy(pred_smplx['right_hand_pose']).view(1, -1).to(device),
        jaw_pose=torch.from_numpy(pred_smplx['jaw_pose']).view(1, -1).to(device),
        expression=torch.from_numpy(pred_smplx['expression']).view(1, -1).to(device),
        transl=-J_0.view(1, 3).to(device)
    )

    temp_V, temp_A = prepare_template_data_smplx(pred_smplx, J_0)

    smplx_V = smplx_output.vertices.clone().detach()
    lbs_weights = smplx_body_model_fit.lbs_weights.clone().float().detach().unsqueeze(0)
    w = diffuse_weights_smplx(smplx_V, lbs_weights, resolution=184)
    # Calculate shape offset
    total_offsets = (smplx_output["shape_offsets"]).clone().float().detach()
    offsets = diffuse_offset_smplx(smplx_V, total_offsets, resolution=256)
    parts = list(label.keys()) + ["body"]

    for part in parts:
        if part == "body":
            mesh_data = {"vertices": smplx_V, "faces": smplx_body_model.faces}
            mesh_V = mesh_data["vertices"].float().to(device)


        else:
            mesh_data = label[part]
            mesh_V = torch.from_numpy(mesh_data["vertices"]).float().to(device).unsqueeze(0) - \
                        torch.from_numpy(pred_smplx['transl']).view(1, -1).to(device) - J_0.view(1, 3).to(device)
            mesh_V = rotate_front(mesh_V, torch.from_numpy(pred_smplx['global_orient']).view(1, -1).to(device))
            # mesh_colors =  mesh_data["colors"]
        trimesh.Trimesh(mesh_V.squeeze(0).detach().cpu().numpy(),
                            mesh_data["faces"],
                            process=False).export(os.path.join(args.output_dir, part + '.ply'))
        # color body to black
        mesh_colors = np.array([[0, 0, 0, 255]] * mesh_V.shape[1])
        mesh_weight = query_weights(mesh_V, w)
        # Calculate offset to zero-shape canonical
        mesh_offset = query_weights(mesh_V, offsets)

        mesh_weight_knn, idx = knn_weights(mesh_V, smplx_V, lbs_weights, K=20)
        is_hand = torch.any(torch.isin(idx, hand_id), dim=-1).view(1,-1,1).expand(1,-1,55)

        mesh_weight[is_hand] = mesh_weight_knn[is_hand]

        # Do inverse LBS
        tsf = temp_A @ torch.inverse(smplx_output.A)
        V_transformed = skinning_mask(mesh_V, mesh_weight, tsf)
        # V_transformed = mesh_V
        mesh_offset = skinning_mask_rotation_only(mesh_offset, mesh_weight, temp_A)

        # clean up memory
        torch.cuda.empty_cache()

        # Redo skinning weights again
        w2 = diffuse_weights_smplx(temp_V, lbs_weights, resolution=128)
        mesh_weight2 = query_weights(V_transformed, w2)
        mesh_weight_knn2, idx = knn_weights(V_transformed, temp_V, lbs_weights, K=20)

        mesh_weight2[is_hand] = mesh_weight_knn2[is_hand]

        out_file_name = os.path.basename(args.label_path).split('-')[1]
        os.makedirs(args.output_dir, exist_ok=True)

        np.save(os.path.join(args.output_dir, out_file_name + '_weights.npy'),
                    mesh_weight2.squeeze(0).detach().cpu().numpy())
        V_saved_beta = V_transformed + J_0.view(1, 3).to(device)

        save_mesh(V_saved_beta, mesh_data["faces"], weights2colors(mesh_weight2.squeeze(0).detach().cpu().numpy()),
                os.path.join(args.output_dir, out_file_name + '_weights.obj'))
 
        save_mesh(V_saved_beta, mesh_data["faces"], mesh_colors, os.path.join(args.output_dir, f"beta_template_{part}.ply"))
        

        V_saved = V_saved_beta - mesh_offset
        save_mesh(V_saved, mesh_data["faces"], mesh_colors, os.path.join(args.output_dir, f"template_{part}.ply"))

        # save mesh offset
        with open(f"output_path/template_{part}_offset.pkl", 'wb') as f:
            pickle.dump(mesh_offset, f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--smplx_dir', type=str, default='/data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/00185/Inner')
    parser.add_argument('-l', '--label_path', type=str, default='/mnt/work/Clothes4D/SMLCHumans/00185/Inner/Take1/Semantic/clothes/cloth-f00011.pkl')
    parser.add_argument('-o', '--output_dir', type=str, default='output_path')
    parser.add_argument('-ind', '--index', type=int, default=0)

    args = parser.parse_args()
    main(args)