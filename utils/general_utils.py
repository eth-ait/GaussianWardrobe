#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2 as cv
import config
from gaussians.gaussian_renderer import render3
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def rectify_depth(render, max_value=10):
    new_depth = render["depth"].clone()
    fix_depth_mask = (render["mask"] > 1e-4)
    new_depth[fix_depth_mask] = new_depth[fix_depth_mask] / render["mask"][fix_depth_mask] 
    new_depth[(new_depth < 1e-4) | (render["mask"] < 0.8)] = max_value
    return new_depth

def merge_gaussian_val(gaussian_vals_1, gaussian_vals_2):
    gaussian_vals = {}
    for key in gaussian_vals_1.keys():
        if key == "max_sh_degree":
            gaussian_vals[key] = gaussian_vals_1[key]
        elif key == "gaussian_nml":
            continue
        else:
            gaussian_vals[key] = torch.concat([gaussian_vals_1[key],
                                                            gaussian_vals_2[key]], dim=0)
    return gaussian_vals

from scipy.ndimage import convolve

def masked_mean_filter(image, mask, kernel_size=5):
    # Ensure mask is binary float (0 or 1)
    mask = (mask > 0).detach().cpu().numpy().astype(np.float32)
    image_np = image.detach().cpu().numpy().astype(np.float32)

    # Create averaging kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    # Prepare output image
    filtered = np.zeros_like(image.detach().cpu().numpy(), dtype=np.float32)

    for c in range(image.shape[2]):  # Apply per channel
        img_c = image_np[:, :, c].astype(np.float32)
        
        # Convolve image * mask and the mask itself
        masked_img = img_c * mask
        sum_vals = convolve(masked_img, kernel, mode='constant', cval=0.0)
        count_vals = convolve(mask, kernel, mode='constant', cval=0.0)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_vals = np.where(count_vals > 0, sum_vals / count_vals, img_c)

        filtered[:, :, c] = mean_vals

    # Keep original pixels outside the mask
    mask_3ch = np.stack([mask]*3, axis=-1)
    output = np.where(mask_3ch == 1, filtered, image.detach().cpu().numpy())
    
    return torch.tensor(output)

def filter_gaussian(upper_gaussian, lower_gaussian, label_rgb_map, rgb_map, mask_map, upper_channel=0, buffer=0.05):
    # process depth map
    lower_gaussian["depth"][lower_gaussian["depth"] < 1e-4] = 10
    upper_gaussian["depth"][upper_gaussian["depth"] < 1e-4] = 10
    depth_upper_map = upper_gaussian["depth"].clone()
    fix_upper_depth_mask = (upper_gaussian["mask"] > 0.)
    depth_upper_map[fix_upper_depth_mask] = depth_upper_map[fix_upper_depth_mask] / upper_gaussian["mask"][fix_upper_depth_mask] 


    depth_lower_map = lower_gaussian["depth"].clone()
    fix_lower_depth_mask = (lower_gaussian["mask"] > 0.) 
    depth_lower_map[fix_lower_depth_mask] =  depth_lower_map[fix_lower_depth_mask] / lower_gaussian["mask"][fix_lower_depth_mask]

    # set threshold to 0.5
    label_rgb_map_np = (label_rgb_map[upper_channel, :, :]*255).detach().cpu().numpy()   
    label_rgb_map_np[mask_map[0,:,:].detach().cpu().numpy()  < 0.5] = 0   

    # Ensure it's uint8
    label_rgb_map_np = label_rgb_map_np.astype(np.uint8)

    # Threshold to binary
    _, thresh = cv.threshold(label_rgb_map_np, 127, 255, cv.THRESH_BINARY)

    # Find contours and hierarchy
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create blank image to draw filled result
    filled = np.zeros_like(label_rgb_map_np)

    # Draw filled contours
    for i, h in enumerate(hierarchy[0]):
        parent = h[3]
        if parent == -1:
            # Top-level (outer) contour: fill white
            cv.drawContours(filled, contours, i, 255, cv.FILLED)
        else:
            # Hole or inner contour: fill black
            cv.drawContours(filled, contours, i, 255, cv.FILLED)

    # Convert to BGR for visualization
    output = cv.cvtColor(filled, cv.COLOR_GRAY2BGR)

    # Draw only top-level contours in green
    for i, h in enumerate(hierarchy[0]):
        if h[3] == -1:  # No parent = top-level
            cv.drawContours(output, contours, i, (0, 0, 0), 3)
    # cv.imwrite('output.jpg', output)
    # find potential penetrate
    body_penetrate_mask = torch.tensor(output[:, :, 0] > 128).to(config.device) & (upper_gaussian["mask"] > 0.95)[0, :, :] & (depth_lower_map >= (depth_upper_map - buffer))[0,:,:]
    filtered_rgb_map = rgb_map.clone()
    # filtered_rgb_map[:, body_mask] = body_rgb_map[:, body_mask]
    filtered_rgb_map[:, body_penetrate_mask] = upper_gaussian["render"][3:, :, :][:, body_penetrate_mask]
    visibility_mask = label_rgb_map.clone()
    # visibility_mask[:, body_mask] = 0
    visibility_mask[:, body_penetrate_mask] = upper_gaussian["render"][:3, :, :][:, body_penetrate_mask]

    depth_diff = depth_upper_map - depth_lower_map
    depth_diff -= depth_diff.min()
    depth_diff /= depth_diff.max()
    return filtered_rgb_map, visibility_mask, depth_diff

def render_gaussian(gaussian_vals, items, bg_color):
    gaussian_vals["colors"] = torch.concat([gaussian_vals["label_colors"], gaussian_vals["colors"]], dim=1)
    render_ret = render3(
    gaussian_vals,
    bg_color.repeat(2),
    items['extr'],
    items['intr'],
    items['img_w'],
    items['img_h']
    )
    return render_ret

def create_bounding_box(center, box_size, image_size):
    """
    Create a clipped bounding box from a center point and box size.

    Parameters:
    - center (tuple of float): (center_x, center_y)
    - box_size (tuple of int): (box_width, box_height)
    - image_size (tuple of int): (image_width, image_height)

    Returns:
    - (x_min, y_min, x_max, y_max): Clipped bounding box coordinates
    """
    center_y, center_x = center.round()
    box_width, box_height = box_size
    image_width, image_height = image_size



    # Calculate box corners
    x_min = int(center_x - box_width / 2)
    y_min = int(center_y - box_height / 2)
    x_max = int(center_x + box_width / 2)
    y_max = int(center_y + box_height / 2)

    # Clamp to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)

    return [x_min, y_min, x_max, y_max]

