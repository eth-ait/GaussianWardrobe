import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.insert(0, project_root)
from copyreg import pickle

import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import trimesh
import yaml
import tqdm
from trimesh.graph import neighbors
import smplxd as smplx
from network.volume import CanoBlendWeightVolume
from utils.renderer.renderer_pytorch3d import Renderer
from utils.smpl_util import smplx_model
import config
import pickle as pkl
import pytorch3d.ops as ops


def find_neighbor(pixel_idx, pix_to_face, mask, mesh, x, y):
    dx_l = [-1, -1, 1, 1]
    dy_l = [1, -1, 1, -1]
    cur_face = pix_to_face[x,y]
    img_h, img_w = pix_to_face.shape
    neighbors = set({pixel_idx[x, y]})
    for dx in dx_l:
        for dy in dy_l:
            if x+dx < 0 or x+dx >= img_h or y+dy < 0 or y+dy >=img_w or not mask[x+dx, y+dy]:
                continue
            if set(mesh.faces[pix_to_face[x+dx,y+dy]]).intersection(mesh.faces[cur_face]):
                neighbors.add(pixel_idx[x+dx,y+dy])
    return neighbors

def find_face(pixel_idx, mask, x, y):
    img_h, img_w = pixel_idx.shape
    valid_faces = []
    # Upper right face
    if x - 1 >= 0 and mask[x - 1, y] and y + 1 < img_w and mask[x, y + 1]:
        valid_faces.append([pixel_idx[x - 1, y], pixel_idx[x, y], pixel_idx[x, y + 1]])
    # Lower right face
    if y + 1 < img_w and mask[x, y + 1] and x + 1 < img_h and mask[x + 1, y]:
        valid_faces.append([pixel_idx[x, y + 1], pixel_idx[x, y], pixel_idx[x + 1, y]])
    # Lower left face
    if x + 1 < img_h and mask[x + 1, y] and y - 1 >= 0 and mask[x, y - 1]:
        valid_faces.append([pixel_idx[x + 1, y], pixel_idx[x, y], pixel_idx[x, y - 1]])
    # Upper left face
    if y - 1 >= 0 and mask[x, y - 1] and x - 1 >= 0 and mask[x - 1, y]:
        valid_faces.append([pixel_idx[x, y - 1], pixel_idx[x, y], pixel_idx[x - 1, y]])
    return valid_faces
def flip_faces(valid_faces):
    flipped_faces = []
    for face in valid_faces:
        flipped_faces.append(face[::-1])
    return flipped_faces

def get_faces(pixel_idx, mask):
    img_h, img_w = mask.shape
    # generate adjacency list
    valid_faces = []
    with_face = []
    for x in range(img_h):
        for y in range(img_w):
            if mask[x, y]:
                cur_valid_face = find_face(pixel_idx, mask, x, y)
                if y > img_w / 2:
                    cur_valid_face = flip_faces(cur_valid_face)
                valid_faces += cur_valid_face
                with_face.append(len(cur_valid_face) > 0)
    return np.array(valid_faces), np.array(with_face)

def get_neighbors(pixel_idx, mask, pix_to_face, mesh, neighbor_max_num=8):
    n_init_point = mask.sum()
    img_h, img_w = mask.shape
    # generate adjacency list
    adj = {}
    for x in range(img_h):
        for y in range(img_w):
            if mask[x, y]:
                adj[pixel_idx[x,y]] = find_neighbor(pixel_idx, pix_to_face, mask, mesh, x, y)
    with_neighbors = []
    neighbor_idxs = np.tile(np.arange(n_init_point)[:, None], (1, neighbor_max_num))
    neighbor_weights = np.zeros((n_init_point, neighbor_max_num), dtype=np.float32)
    for idx in range(n_init_point):
        neighbor_num = min(len(adj[idx]), neighbor_max_num)
        neighbor_idxs[idx, :neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]
        neighbor_weights[idx, :neighbor_num] = -1.0 / neighbor_num
        if neighbor_num == 1:
            with_neighbors.append(False)
        else:
            with_neighbors.append(True)
    return neighbor_idxs, neighbor_weights, np.array(with_neighbors)


def save_pos_map(pos_map, path):
    mask = np.linalg.norm(pos_map, axis = -1) > 0.
    positions = pos_map[mask]
    print('Point nums %d' % positions.shape[0])
    pc = trimesh.PointCloud(positions)
    pc.export(path)

def generate_cano_maps(cano_smpl_v_dup, cano_smpl_attr_dup, attr):
    cano_renderer.set_model(cano_smpl_v_dup, cano_smpl_attr_dup)
    cano_renderer.set_camera(front_mv)
    front_cano_attr_map, front_cano_pix_to_face = cano_renderer.render()
    front_cano_attr_map = front_cano_attr_map[:, :, :3]

    cano_renderer.set_camera(back_mv)
    back_cano_attr_map, back_cano_pix_to_face = cano_renderer.render()
    back_cano_attr_map = back_cano_attr_map[:, :, :3]
    back_cano_attr_map = cv.flip(back_cano_attr_map, 1)
    cano_attr_map = np.concatenate([front_cano_attr_map, back_cano_attr_map], 1)
    cv.imwrite(data_dir + f'/{args.output_path}/cano_smpl_{attr}_map.exr', cano_attr_map)
    # cano_attr_map_normalized = (cano_attr_map - cano_attr_map.min())
    # cano_attr_map_normalized /= cano_attr_map_normalized.max()
    # cv.imwrite(data_dir + f'/{args.output_path}/cano_smpl_{attr}_map_normalized.jpg', cano_attr_map_normalized * 255)
    return front_cano_pix_to_face, back_cano_pix_to_face, cano_attr_map


def interpolate_lbs(pts, vertices, faces, vertex_lbs):
    from utils.posevocab_custom_ops.nearest_face import nearest_face_pytorch3d
    from utils.geo_util import barycentric_interpolate
    dists, indices, bc_coords = nearest_face_pytorch3d(
        torch.from_numpy(pts).to(torch.float32).cuda()[None],
        torch.from_numpy(vertices).to(torch.float32).cuda()[None],
        torch.from_numpy(faces).to(torch.int64).cuda()
    )
    # print(dists.mean())
    lbs = barycentric_interpolate(
        vert_attris = vertex_lbs[None].to(torch.float32).cuda(),
        faces = torch.from_numpy(faces).to(torch.int64).cuda()[None],
        face_ids = indices,
        bc_coords = bc_coords
    )
    return lbs[0].cpu().numpy()


map_size = 1024


if __name__ == '__main__':
    from argparse import ArgumentParser
    import importlib

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('-o', '--output_path', type=str, help='Configuration output path.')
    arg_parser.add_argument('-t', '--template_path', type=str, help='Configuration template path.')
    arg_parser.add_argument('-rc', '--render_color', action='store_true', default=False, help='Render color or not.')
    arg_parser.add_argument('-ro', '--render_offset', action='store_true', default=False, help='Render offset or not.')
    arg_parser.add_argument('-lw', '--lbs_weight', default="cano_weight_volume", help='Render color or not.')
    arg_parser.add_argument('-seq', '--sequence_name', default="smpl_params", help='Name of the rendering sequence')
    arg_parser.add_argument('-d', '--data_dir', default="", help='specify the data directory')
    args = arg_parser.parse_args()
    config.load_global_opt(args.config_path)

    opt = yaml.load(open(args.config_path, encoding = 'UTF-8'), Loader = yaml.FullLoader)
    dataset_module = opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    if args.data_dir == "":
        data_dir = config.opt["train"]["data"]["data_dir"]
    else:
        data_dir = args.data_dir

    os.makedirs(data_dir + f'/{args.output_path}', exist_ok=True)

    cano_renderer = Renderer(map_size, map_size, shader_name = 'vertex_attribute')

    smpl_model, smpl_model_fit = smplx_model(data_dir)
    smpl_data = dict(np.load(data_dir + f'/{args.sequence_name}.npz', allow_pickle=True))
    frame_list = [i for i in range(len(smpl_data["body_pose"]))]
    if args.lbs_weight != "cano_weight_volume":
        smpl_data["betas"] = np.zeros_like(smpl_data["betas"])
    smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

    with torch.no_grad():
        cano_smpl = smpl_model.forward(
            betas = smpl_data['betas'],
            global_orient = config.cano_smpl_global_orient[None],
            transl = config.cano_smpl_transl[None],
            body_pose = config.cano_smpl_body_pose[None]
        )
        cano_smpl_v = cano_smpl.vertices[0].cpu().numpy()
        cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
        cano_smpl_v_min = cano_smpl_v.min()
        smpl_faces = smpl_model.faces.astype(np.int64)

    if os.path.exists(data_dir + f'/{args.template_path}.ply'):
        print('# Loading template from %s' % (data_dir + f'/{args.template_path}.ply'))
        template = trimesh.load(data_dir + f'/{args.template_path}.ply', process = False)
        using_template = True
    else:
        import pickle
        print(f'# Cannot find {args.template_path}.ply from {data_dir}, using SMPL-X as template')
        template = trimesh.Trimesh(cano_smpl_v, smpl_faces, process = False)
        using_template = False

    cano_smpl_v = template.vertices.astype(np.float32)
    smpl_faces = template.faces.astype(np.int64)
    cano_smpl_v_dup = cano_smpl_v[smpl_faces.reshape(-1)]
    cano_smpl_n_dup = template.vertex_normals.astype(np.float32)[smpl_faces.reshape(-1)]
    cano_smpl_c_dup = template.visual.vertex_colors.astype(np.float32)[smpl_faces.reshape(-1)][:, :3] / 255

    # define front & back view matrices
    front_mv = np.identity(4, np.float32)
    front_mv[:3, 3] = -cano_center + np.array([0, 0, -10], np.float32)
    front_mv[1:3] *= -1

    back_mv = np.identity(4, np.float32)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ cano_center + np.array([0, 0, -10], np.float32)
    back_mv[1:3] *= -1

    # render hand mask
    if args.render_color:
        if "cloth" in args.template_path:
            generate_cano_maps(cano_smpl_v_dup, cano_smpl_c_dup, attr="segment")
        elif "smplx_upper" in args.template_path:
            generate_cano_maps(cano_smpl_v_dup, cano_smpl_c_dup, attr="smplx_upper")
        elif "smplx_lower" in args.template_path:
            generate_cano_maps(cano_smpl_v_dup, cano_smpl_c_dup, attr="smplx_lower")
        else:
            generate_cano_maps(cano_smpl_v_dup, cano_smpl_c_dup, attr="smplx_fix_offset")
        

    # render offset map
    if args.render_offset:
        with open((data_dir + f'/{args.template_path}.pkl'), 'rb') as f:
            mesh_offset = pkl.load(f)
        cano_smpl_off_dup = mesh_offset[smpl_faces.reshape(-1)]
        _, _, cano_offset_map = generate_cano_maps(cano_smpl_v_dup, cano_smpl_off_dup, attr="offset")

    # render canonical smpl position maps

    front_cano_pix_to_face, back_cano_pix_to_face, cano_pos_map = generate_cano_maps(cano_smpl_v_dup, cano_smpl_v_dup, attr="pos")
    # Generate neighbor idx and neighbor weights

    cano_pix_to_face_map = np.concatenate([front_cano_pix_to_face, back_cano_pix_to_face], 1)
    cano_smpl_mask = torch.linalg.norm(torch.from_numpy(cano_pos_map).to(torch.float32), dim=-1) > 0.
    pixel_idx = np.full(cano_pix_to_face_map.shape, -1, dtype=np.int64)
    pixel_idx[cano_smpl_mask] = np.arange(cano_smpl_mask.sum())
    print("Generate neighbor map")
    neighbor_idxs, neighbor_weights, with_neighbor = get_neighbors(pixel_idx, cano_smpl_mask, cano_pix_to_face_map, template)
    # save as npy file and use later
    with open(data_dir + f'/{args.output_path}/neighbor_idx.npy', 'wb') as f:
        np.save(f, neighbor_idxs)
    with open(data_dir + f'/{args.output_path}/neighbor_weights.npy', 'wb') as f:
        np.save(f, neighbor_weights)
    with open(data_dir + f'/{args.output_path}/with_neighbor.npy', 'wb') as f:
        np.save(f, with_neighbor)

    # Generate mesh-like faces
    print("Generate face map")
    valid_faces, with_face = get_faces(pixel_idx, cano_smpl_mask)
    with open(data_dir + f'/{args.output_path}/valid_faces.npy', 'wb') as f:
        np.save(f, valid_faces)
    with open(data_dir + f'/{args.output_path}/with_face.npy', 'wb') as f:
        np.save(f, with_face)


    # render canonical smpl normal maps
    generate_cano_maps(cano_smpl_v_dup, cano_smpl_n_dup, attr="nml")


    body_mask = np.linalg.norm(cano_pos_map, axis = -1) > 0.
    cano_pts = cano_pos_map[body_mask]

    if using_template:
        weight_volume = CanoBlendWeightVolume(data_dir + f'/{args.lbs_weight}.npz')
        pts_lbs = weight_volume.forward_weight(torch.from_numpy(cano_pts)[None].cuda())[0]
        verts_lbs = weight_volume.forward_weight(torch.from_numpy(cano_smpl_v)[None].cuda())[0]
        if args.lbs_weight != "cano_weight_volume":
            cano_offset = cano_offset_map[body_mask]
            save_weight_volume = CanoBlendWeightVolume(data_dir + f'/cano_weight_volume.npz')
            save_pts_lbs = save_weight_volume.forward_weight(torch.from_numpy(cano_pts + cano_offset)[None].cuda())[0]
            np.save(data_dir + f'/{args.output_path}/init_pts_lbs.npy', save_pts_lbs.cpu().numpy())
        else:
            np.save(data_dir + f'/{args.output_path}/init_pts_lbs.npy', pts_lbs.cpu().numpy())
    else:
        pts_lbs = interpolate_lbs(cano_pts, cano_smpl_v, smpl_faces, smpl_model.lbs_weights)
        pts_lbs = torch.from_numpy(pts_lbs).cuda()
        np.save(data_dir + f'/{args.output_path}/init_pts_lbs.npy', pts_lbs.cpu().numpy())

    inv_cano_smpl_A = torch.linalg.inv(cano_smpl.A).cuda()
    body_mask = torch.from_numpy(body_mask).cuda()
    cano_pts = torch.from_numpy(cano_pts).cuda()
    pts_lbs = pts_lbs.cuda()
    verts_lbs = verts_lbs.cuda()

    for pose_idx in tqdm.tqdm(frame_list, desc = 'Generating positional maps...'):
        with torch.no_grad():
            live_smpl_woRoot = smpl_model_fit.forward(
                betas = smpl_data['betas'],
                # global_orient = smpl_data['global_orient'][pose_idx][None],
                # transl = smpl_data['transl'][pose_idx][None],
                body_pose = smpl_data['body_pose'][pose_idx][None],
                jaw_pose = smpl_data['jaw_pose'][pose_idx][None],
                expression = smpl_data['expression'][pose_idx][None],
                left_hand_pose = smpl_data['left_hand_pose'][pose_idx][None],
                right_hand_pose = smpl_data['right_hand_pose'][pose_idx][None]
            )

        cano2live_jnt_mats_woRoot = torch.matmul(live_smpl_woRoot.A.cuda(), inv_cano_smpl_A)[0]
        pt_mats = torch.einsum('nj,jxy->nxy', pts_lbs, cano2live_jnt_mats_woRoot)
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
        vert_mats = torch.einsum('nj,jxy->nxy', verts_lbs, cano2live_jnt_mats_woRoot)
        live_verts = torch.einsum('nxy,ny->nx', vert_mats[..., :3, :3], torch.from_numpy(cano_smpl_v).cuda()) + vert_mats[..., :3, 3]
        # cloud=trimesh.Trimesh(live_verts.detach().cpu().numpy(), template.faces)
        # ply_filename = os.path.join(data_dir, f'{args.output_path}', '%08d' % pose_idx + '.ply')
        # cloud.export(ply_filename)
        live_pos_map = torch.zeros((map_size, 2 * map_size, 3)).to(live_pts)
        live_pos_map[body_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = live_pos_map.permute(1, 2, 0).cpu().numpy()

        cv.imwrite(data_dir + f'/{args.output_path}/%08d.exr' % pose_idx, live_pos_map)
