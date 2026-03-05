import torch
from torch import einsum
import numpy as np
from pytorch3d import ops
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy as sp
import trimesh


def weights2colors(weights):

    cmap = plt.get_cmap('Paired')

    color_map = [ 'white', #0
                    'blue', #1
                    'green', #2
                    'red', #3
                    'white', #4
                    'white', #5
                    'white', #6
                    'green', #7
                    'blue', #8
                    'red', #9
                    'white', #10
                    'white', #11
                    'white', #12
                    'blue', #13
                    'green', #14
                    'red', #15
                    'green', #16
                    'blue', #17
                    'blue', #18
                    'green', #19
                    'lightpink', #20
                    'lightpink', #21
                    'brown', #22
                    'darkyellow', #23
                    'green', #24
                    'red', #25
                    'lightpink', #26
                    'red', #27
                    'green', #28
                    'lightpink', #29
                    'green', #30
                    'darkyellow', #31
                    'lightpink', #32
                    'darkyellow', #33
                    'blue', #34
                    'lightpink', #35
                    'blue', #36
                    'brown', #37
                    'lightpink', #38
                    'brown', #39
                    'red', #40
                    'lightpink', #41
                    'red', #42
                    'green', #43
                    'lightpink', #44
                    'green', #45
                    'darkyellow', #46
                    'lightpink', #47
                    'darkyellow', #48
                    'blue', #49
                    'lightpink', #50
                    'blue', #51
                    'brown', #52
                    'lightpink', #53
                    'brown', #54
        ]
    
    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'lightpink': cmap.colors[4],
                    'darkgreen': cmap.colors[1],
                    'darkyellow': cmap.colors[7],
                    'brown': cmap.colors[11],
                    'green':cmap.colors[3],
                    'white': [1,1,1],
                    'red':cmap.colors[5],
                    }
    
    colors = []
    num_bones = weights.shape[1]
    for i in range(num_bones):
        colors.append( np.array(color_mapping[color_map[i]]) )

    colors = np.stack(colors)[None, ...]
    verts_colors = weights[..., None] * colors
    verts_colors = verts_colors.sum(1)

    return verts_colors

def batch_rodrigues(rot_vecs, epsilon = 1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def skinning_mask(x, lbs_weights, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    # 5. Do skinning:
    # W is N x V x (J + 1)
    batch_size = x.shape[0]
    W = lbs_weights.expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = W.shape[-1]
    T = torch.matmul(W, tfs.reshape(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, x.shape[1], 1],
                               dtype=x.dtype, device=x.device)
    v_posed_homo = torch.cat([x, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    x_h = v_homo[:, :, :3, 0]
    return x_h
def skinning_mask_rotation_only(x, lbs_weights, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points with only rotations. shape: [B, N, D]
    """
    # 5. Do skinning:
    # W is N x V x (J + 1)
    batch_size = x.shape[0]
    W = lbs_weights.expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = W.shape[-1]
    T = torch.matmul(W, tfs.reshape(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    T[0, :, :, 3] = 0
    T[0, :, 3, 3] = 1
    homogen_coord = torch.ones([batch_size, x.shape[1], 1],
                               dtype=x.dtype, device=x.device)
    v_posed_homo = torch.cat([x, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    x_h = v_homo[:, :, :3, 0]
    return x_h
def query_weights(xc, lbs_voxel, mode='bilinear'):
    shape = xc.shape
    N = 1
    xc = xc.view(1, -1, 3)
    w = F.grid_sample(lbs_voxel.expand(N, -1, -1, -1, -1),
                            xc[:, :, None, None],
                            align_corners=True,
                            mode=mode,
                            padding_mode='border')
    w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
    w = w.view(*shape[:-1], -1)
    return w

def knn_weights(x, smpl_verts, smpl_weights, K=1):
    dist, idx, _ = ops.knn_points(x, smpl_verts.detach(), K=K)
    dist = dist.sqrt().clamp_(0.0001, 1.)
    weights = smpl_weights[0, idx]
    ws = 1. / dist
    ws = ws / ws.sum(-1, keepdim=True)
    weights = (ws[..., None] * weights).sum(-2)

    return weights.detach(), idx.detach()

def diffuse_weights_smplx(smpl_verts, smpl_weights, resolution=128, iter=30):
    # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py

    b, c, d, h, w = 1, 55, resolution // 4, resolution, resolution
    x_range = (torch.linspace(-1, 1, steps=w, device=smpl_verts.device)).view(1, 1, 1, w).expand(1, d, h, w)
    y_range = (torch.linspace(-1, 1, steps=h, device=smpl_verts.device)).view(1, 1, h, 1).expand(1, d, h, w)
    z_range = (torch.linspace(-1, 1, steps=d, device=smpl_verts.device)).view(1, d, 1, 1).expand(1, d, h, w)
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(b, 3, -1).permute(0, 2, 1)

    dist, idx, _ = ops.knn_points(grid, smpl_verts.detach(), K=30)
    dist = dist.sqrt().clamp_(0.0001, 1.)
    weights = smpl_weights[0, idx]

    ws = 1. / dist
    ws = ws / ws.sum(-1, keepdim=True)
    weights = (ws[..., None] * weights).sum(-2)

    b, c, d, h, w = 1, 55, resolution // 4, resolution, resolution
    weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
    for _ in range(iter):
        mean=(weights[:,:,2:,1:-1,1:-1]+weights[:,:,:-2,1:-1,1:-1]+\
              weights[:,:,1:-1,2:,1:-1]+weights[:,:,1:-1,:-2,1:-1]+\
              weights[:,:,1:-1,1:-1,2:]+weights[:,:,1:-1,1:-1,:-2])/6.0
        weights[:, :, 1:-1, 1:-1, 1:-1] = (weights[:, :, 1:-1, 1:-1, 1:-1] - mean) * 0.7 + mean
        sums = weights.sum(1, keepdim=True)
        weights = weights / sums
    return weights.detach()

def diffuse_offset_smplx(smpl_verts, smpl_offset, resolution=128, iter=30):
    # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py

    b, c, d, h, w = 1, 3, resolution // 4, resolution, resolution
    x_range = (torch.linspace(-1, 1, steps=w, device=smpl_verts.device)).view(1, 1, 1, w).expand(1, d, h, w)
    y_range = (torch.linspace(-1, 1, steps=h, device=smpl_verts.device)).view(1, 1, h, 1).expand(1, d, h, w)
    z_range = (torch.linspace(-1, 1, steps=d, device=smpl_verts.device)).view(1, d, 1, 1).expand(1, d, h, w)
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(b, 3, -1).permute(0, 2, 1)

    dist, idx, _ = ops.knn_points(grid, smpl_verts.detach(), K=30)
    dist = dist.sqrt().clamp_(0.0001, 1.)
    offsets = smpl_offset[0, idx]

    ws = 1. / dist
    ws = ws / ws.sum(-1, keepdim=True)
    offsets = (ws[..., None] * offsets).sum(-2)

    b, c, d, h, w = 1, 3, resolution // 4, resolution, resolution
    offsets = offsets.permute(0, 2, 1).reshape(b, c, d, h, w)

    return offsets.detach()

def compute_surface_metrics(mesh_pred, mesh_gt):
    """Compute surface metrics (chamfer distance and f-score) for one example.
    Args:
    mesh: trimesh.Trimesh, the mesh to evaluate.
    Returns:
    chamfer: float, chamfer distance.
    fscore: float, f-score.
    """
    # Chamfer
    point_gt = mesh_gt["vertices"]
    point_gt = point_gt.astype(np.float32)

    point_pred = mesh_pred.vertices
    point_pred = point_pred.astype(np.float32)

    dist_pred_to_gt = distance_field_helper(point_pred, point_gt)

    # TODO: subdivide by 2 following OccNet
    # https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/im2mesh/eval.py#L136

    # Following the tradition to scale chamfer distance up by 10.
    return dist_pred_to_gt

def distance_field_helper(source, target):
    target_kdtree = sp.spatial.cKDTree(target)
    distances, idx = target_kdtree.query(source, workers=-1)
    return distances

def generateRGBA(distance):
    distance = np.array(distance)
    mean_dist = np.expand_dims(np.mean(distance, axis=0), axis=1)
    rgb = np.repeat(mean_dist, 3, axis=1)
    return rgb

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def render_mesh(img_w, img_h):
    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=50000,
    )

def save_mesh(vertices, faces, colors=None, f_name="output.obj"):
    if colors is not None:
        mesh = trimesh.Trimesh(vertices.squeeze(0).detach().cpu().numpy(),
                                faces,
                                vertex_colors=colors,
                                process=False)
    else:
        mesh = trimesh.Trimesh(vertices.squeeze(0).detach().cpu().numpy(),
                        faces,
                        process=False)

    mesh.export(f_name)