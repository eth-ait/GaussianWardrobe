from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import ops
from network.network import ImplicitNetwork as ImplicitNet


class DeformationGraph(nn.Module):
    def __init__(self, opt=None, K=5):
        super().__init__()
        self.deformation_graph = ImplicitNet(**opt)
        self.K = K

    def update_deformation_nodes(self, deformation_graph_verts, cano_mesh):
        """
        Here the deformation nodes should be progressively updated based on the new cloth template
        """
        if not torch.is_tensor(deformation_graph_verts):
            self.deformation_graph_verts = torch.from_numpy(deformation_graph_verts).float().cuda().detach()
        else:
            assert False, "deformation_graph_verts should be a numpy array"
            self.deformation_graph_verts = deformation_graph_verts.detach()
        self.cano_mesh = cano_mesh

    def get_transformation(self, nodes, cond):
        n_batch = cond["smpl"].shape[0]
        _, n_joint, n_dim = nodes.shape
        y = torch.zeros((n_batch, n_joint, n_dim))
        nodes_input, _ = torch.broadcast_tensors(nodes, y)
        transformation = self.deformation_graph(nodes_input, cond)  # predict the 6 DoF
        rot = transformation[:, :, :3]  # TODO try to use the quaternion representation
        trans = transformation[:, :, 3:]

        rot_mat = batch_rodrigues(rot.reshape(n_batch*n_joint, 3)) # from axis-angle to rotation matrix
        # rot_mat = rot_mat.reshape(n_batch, n_joint, 3, 3)
        # rot_mat = euler2rotmat(rot)
        transform_mat = to_transform_mat(rot_mat, trans.reshape(n_batch*n_joint, 3, 1))
        transform_mat = transform_mat.reshape(n_batch, n_joint, 4, 4)
        return transform_mat

    def forward(self, x=None, cond=None, nodes=None, nodes_deformed=None, lbs=True,
                smpl_tfs=None, inverse=False, return_transform_mat=False,
                smpl_root_orient=None, smpl_trans=None, scale=None,
                time_enc=None):
        """
        Obtain the warped points
        """
        n_batch, n_point, n_dim = x.shape
        if nodes is None:
            transform_mat = self.get_transformation(self.deformation_graph_verts[None], cond)
        else:
            transform_mat = self.get_transformation(nodes, cond)
        if lbs:
            # Version 1
            # TODO double-check the correctness of the following code
            # Confirmed: the following code is incorrect
            # global_orient = smpl_tfs[:, :1].detach().clone()
            # transform_mat = global_orient @ transform_mat

            # Version 2
            smpl_root_orient_mat = batch_rodrigues(smpl_root_orient)
            smpl_root_orient_mat = to_transform_mat(smpl_root_orient_mat, torch.zeros(
                [smpl_root_orient_mat.shape[0], 3, 1]).cuda()).unsqueeze(1).detach()
            # TODO double-check the correctness of the following code
            transform_mat = torch.matmul(smpl_root_orient_mat.expand(-1, transform_mat.shape[1], -1, -1), transform_mat)
            transform_mat[:, :, :3, :] = transform_mat[:, :, :3, :] * scale.unsqueeze(1).unsqueeze(1)
            transform_mat[:, :, :3, 3] = transform_mat[:, :, :3, 3] + smpl_trans.unsqueeze(1) * scale.unsqueeze(1)
            if return_transform_mat:
                return transform_mat
            if inverse:

                # assert nodes_deformed is not None
                if nodes_deformed is None:
                    skinning_weights_self = torch.eye(self.deformation_graph_verts.shape[0]).cuda()
                    nodes_deformed = skinning(self.deformation_graph_verts[None], skinning_weights_self[None],
                                              transform_mat, inverse=False, return_T=False).detach()
                distance_squared, nn_index, _ = ops.knn_points(x, nodes_deformed, K=self.K,
                                                               return_nn=False)
            else:
                distance_squared, nn_index, _ = ops.knn_points(x.reshape(1, n_batch*n_point, 3),
                                                               self.deformation_graph_verts.unsqueeze(0), K=self.K,
                                                               return_nn=False)

            distance = torch.sqrt(distance_squared)
            distance = distance.reshape(1, n_batch*n_point, self.K)
            nn_index = nn_index.reshape(1, n_batch * n_point, self.K)
            least_distance = distance[0, :,
                             0]  # distance is naturally sorted, we choose the zero index as the least distance

            distance = torch.clamp(distance, max=1)
            weights = -torch.log(distance - 1e-6)[0]
            weights = weights / weights.sum(dim=-1, keepdim=True)

            skinning_weights = torch.zeros(
                (n_batch*n_point, self.deformation_graph_verts.shape[0])).cuda()  # TODO sparse matrix is better
            skinning_weights.scatter_(1, nn_index[0], weights)  # TODO double-check scatter_
            skinning_weights = skinning_weights.reshape(n_batch, n_point, self.deformation_graph_verts.shape[0])
            xc, _ = skinning(x, skinning_weights, transform_mat, inverse=inverse, return_T=True)
            mask = least_distance < 0.2
            mask = mask.reshape(n_batch, n_point, 1)

            mask_true = torch.full((n_batch, n_point, 1), True, device=x.device)
            return xc, mask_true, nodes_deformed

    def forward_graph(self, nodes=None, cond=None, smpl_tfs=None, smpl_root_orient=None, smpl_trans=None, scale=None):
        if nodes is None:
            nodes = self.deformation_graph_verts
        transform_mat = self.get_transformation(nodes, cond)

        # Version 1
        # smpl_root_orient = smpl_tfs[:, 0:1]
        # transform_mat = smpl_root_orient @ transform_mat

        # Version 2
        smpl_root_orient_mat = batch_rodrigues(smpl_root_orient)
        smpl_root_orient_mat = to_transform_mat(smpl_root_orient_mat,
                                                torch.zeros([smpl_root_orient_mat.shape[0], 3, 1]).cuda()).unsqueeze(
            0).detach()

        transform_mat = torch.matmul(smpl_root_orient_mat.expand(-1, transform_mat.shape[1], -1, -1), transform_mat)
        transform_mat[:, :, :3, :] = transform_mat[:, :, :3, :] * scale.unsqueeze(1).unsqueeze(1)
        transform_mat[:, :, :3, 3] = transform_mat[:, :, :3, 3] + smpl_trans.unsqueeze(1) * scale.unsqueeze(1)
        nodes = nodes.squeeze(0)
        skinning_weights_self = torch.eye(nodes.shape[0]).cuda()
        nodes_deformed = skinning(nodes[None], skinning_weights_self[None], transform_mat, inverse=False,
                                  return_T=False)
        return nodes_deformed

    def warping(self, xc, influence_nodes_v_all, rot_mat_all, trans_all, weights):
        """
        warping canonical points based on the deformation nodes that influence them. Here we choose the nearest 5 nodes.
        """
        # deformation graph warping
        warped_xc_all = (torch.einsum('bij, bkj->bki', rot_mat_all,
                                      (xc.repeat_interleave(self.K, dim=0) - influence_nodes_v_all).unsqueeze(
                                          1)).squeeze(1) \
                         + influence_nodes_v_all + trans_all).reshape((xc.shape[0], self.K, -1)) * weights.unsqueeze(-1)
        # weighted average
        warped_xc = warped_xc_all.sum(dim=1)
        # TODO as rigid as possible loss
        return warped_xc

    def inverse_warping(self, xc, influence_nodes_v_all, rot_mat_all, trans_all, weights):
        """
        inverse warping canonical points based on the deformation nodes that influence them. Here we choose the nearest 5 nodes.
        """

        # inverse deformation graph warping
        warped_xc_all = ()


def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    # rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                   2], norm_quat[:,
                                                       3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
        dim=1).view(batch_size, 3, 3)
    return rotMat


def skinning(x, w, tfs, inverse=False, return_T=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
        w_tf = None
    if return_T:
        return x_h[:, :, :3], w_tf
    else:
        return x_h[:, :, :3]


def to_transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def euler2rotmat(euler):
    # the similar function in pytorch3d.transforms has significant math error compared to this one
    # don't know why
    sx = torch.sin(euler[:, 0]).view((-1, 1))
    sy = torch.sin(euler[:, 1]).view((-1, 1))
    sz = torch.sin(euler[:, 2]).view((-1, 1))
    cx = torch.cos(euler[:, 0]).view((-1, 1))
    cy = torch.cos(euler[:, 1]).view((-1, 1))
    cz = torch.cos(euler[:, 2]).view((-1, 1))

    mat_flat = torch.hstack([cy * cz,
                             sx * sy * cz - sz * cx,
                             sy * cx * cz + sx * sz,
                             sz * cy,
                             sx * sy * sz + cx * cz,
                             sy * sz * cx - sx * cz,
                             -sy,
                             sx * cy,
                             cx * cy])
    return mat_flat.view((-1, 3, 3))