import math
import os
import pickle as pkl
import torch
import numpy as np
import trimesh
import argparse
from smplxd import SMPLX
from smplxd.lbs import blend_shapes


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # SMPLX_gender = SMPLX('body_models/smplx', gender='male',
    #                              use_pca=False, flat_hand_mean=False, use_face_contour=False).to(device)
    # SMPLX_neutral = SMPLX('body_models/smplx', gender='neutral',
    #                              use_pca=False, flat_hand_mean=False, use_face_contour=False).to(device)
    
    SMPLX_gender = SMPLX('body_models/smplx', gender='female', num_pca_comps=12,
                            use_pca=True, flat_hand_mean=False, use_face_contour=False).to(device)
    SMPLX_neutral = SMPLX('body_models/smplx', gender='neutral',num_pca_comps=12,
                            use_pca=True, flat_hand_mean=False, use_face_contour=False).to(device)
    pred_smplx = dict(np.load(open(os.path.join(args.input_dir, f"{args.sequence_name}_gender.npz"), 'rb'), allow_pickle=True))

    # convert betas
    ori_betas = torch.from_numpy(pred_smplx['betas']).float().to(device)
    ori_mesh = SMPLX_gender(betas=ori_betas)
    V = ori_mesh.vertices[0]
    ori_v_template = SMPLX_gender.v_template
    ori_blendshape = SMPLX_gender.shapedirs

    blend_shape = torch.einsum('bl,mkl->bmk', [ori_betas, ori_blendshape])
    blend_V = ori_v_template + blend_shape[0]

    t = trimesh.Trimesh(vertices = V.cpu().detach().numpy(), faces = SMPLX_gender.faces, process=False)
    os.makedirs(args.output_dir, exist_ok=True)
    t.export(os.path.join(args.output_dir,  args.file_prefix + "_ori.ply"))
    tar_v_template = SMPLX_neutral.v_template
    tar_blendshape = SMPLX_neutral.shapedirs

    A = tar_blendshape.view(-1, 10)
    B = blend_V.view(-1, 1) - tar_v_template.view(-1, 1)
    X_beta = torch.linalg.lstsq(A, B).solution

    res_mesh = SMPLX_neutral(betas = X_beta.view(-1, 10))


    t = trimesh.Trimesh(vertices = res_mesh.vertices[0].detach().cpu().numpy(), faces = SMPLX_neutral.faces, process=False)
    t.export(os.path.join(args.output_dir, args.file_prefix + "_neutral_mesh.ply"))

    

    # convert expression
    ori_expr = torch.from_numpy(pred_smplx["expression"]).float().to(device)
    n_pose = ori_expr.shape[0]
    ori_mesh_beta = SMPLX_gender(betas=ori_betas)
    ori_mesh_beta_expr = SMPLX_gender(betas=ori_betas.expand(n_pose, -1), 
                                    expression=ori_expr,
                                    body_pose=torch.zeros(pred_smplx['body_pose'].shape).to(device),
                                    left_hand_pose=torch.zeros(pred_smplx['left_hand_pose'].shape).to(device),
                                    right_hand_pose=torch.zeros(pred_smplx['right_hand_pose'].shape).to(device),
                                    leye_pose=torch.zeros((n_pose, 3)).to(device),
                                    reye_pose=torch.zeros((n_pose, 3)).to(device),
                                    jaw_pose=torch.zeros(pred_smplx['jaw_pose'].shape).to(device),
                                    global_orient=torch.zeros(pred_smplx['global_orient'].shape).to(device),
                                    transl=torch.zeros(pred_smplx['transl'].shape).to(device))
    
    tar_expr_shape = SMPLX_neutral.expr_dirs.reshape(-1, 10).expand(n_pose, -1, -1)
    A = tar_expr_shape
    B = ori_mesh_beta_expr.vertices.reshape(n_pose, -1) - ori_mesh_beta.vertices.expand(n_pose, -1, -1).reshape(n_pose, -1)
    X_expr = torch.linalg. lstsq(A, B).solution

    # convert right hand pca
    # ori_right_hand = torch.from_numpy(pred_smplx["right_hand_pose"]).float().to(device)
    # ori_right_hand_pose = torch.einsum('bi,ij->bj', [ori_right_hand, SMPLX_gender.right_hand_components])
    # A = SMPLX_neutral.right_hand_components.T.expand(n_pose, -1, -1)
    # B = ori_right_hand_pose
    # X_right_hand = torch.linalg.lstsq(A, B).solution

    # convert left hand pca
    # ori_left_hand = torch.from_numpy(pred_smplx["left_hand_pose"]).float().to(device)
    # ori_left_hand_pose = torch.einsum('bi,ij->bj', [ori_left_hand, SMPLX_gender.left_hand_components])
    # A = SMPLX_neutral.left_hand_components.T.expand(n_pose, -1, -1)
    # B = ori_left_hand_pose
    # X_left_hand = torch.linalg.lstsq(A, B).solution
    
    pred_smplx["betas"] = X_beta.T.detach().cpu().numpy()
    pred_smplx["expression"] = X_expr.detach().cpu().numpy()
    # pred_smplx["right_hand_pose"] = X_right_hand.detach().cpu().numpy()
    # pred_smplx["left_hand_pose"] = X_left_hand.detach().cpu().numpy()
    with open(os.path.join(args.output_dir, f"{args.sequence_name}.npz"), 'wb') as file:
        pkl.dump(pred_smplx, file)
    with open(os.path.join(args.input_dir, f"{args.sequence_name}.npz"), 'wb') as file:
        pkl.dump(pred_smplx, file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/00185/Outer')
    parser.add_argument('-o', '--output_dir', type=str, default='neutral_SMPLX')
    parser.add_argument('-p', '--file_prefix', type=str, default="00187")
    parser.add_argument('-seq', '--sequence_name', type=str, default="smpl_params_neutral")

    args = parser.parse_args()
    main(args)