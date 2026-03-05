# Preparing Training Data
0. SET CAMERA, LAYER and SUBJECT (gen_data.preprocess_4ddress.py)
1. Copy capatured images and masks for all selected cameras (gen_data.preprocess_4ddress.py)
```
image_list = read_all_png_camera(data_dir, png_prefix="capture")
mask_list = read_all_png_camera(data_dir, png_type="masks", png_prefix="mask")
copy_to_folder(new_data_dir, image_list, ['%04d' % CAMERA, "images"], "png")
copy_to_folder(new_data_dir, mask_list,  ['%04d' % CAMERA, "masks"], "png")
```
2. Copy pose parameters and compress to a .npz file (gen_data.preprocess_4ddress.py) and transform it to neutral space 
```
pose_list = read_all_pose(data_dir)
copy_to_folder(new_data_dir, pose_list, ["pose"], "pkl")
combine_pose_to_npz(new_data_dir)
```
3. create cameras.pkl file for selected cameras (gen_data.preprocess_4ddress.py)
```
all_camera_path = "/data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/rgb_cameras.json"
selected_camera_path = "/data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/"
generate_camera_pkl(all_camera_path, camera_list=camera_list, output_path=selected_camera_path)
```
4. Generate label images (using 4ddress code)
5. Copy and process label images into cloth-body label images for all selected cameras (gen_data.preprocess_4ddress.py)
```
label_list = read_all_png_camera(generated_data_dir, png_type="labels", png_prefix="label")
copy_filter_label(new_data_dir, label_list, png_type="labels")
```
6. Do inverse LBS and deform back to canonical pose for all parts with and without shape offset and save into *template_\*.ply* amd *beta_template_\*.ply* (using meshavatar code)
7. Merge upper and lower layer without shape offset and save into *template_cloth_offset.ply* and *template_cloth_offset.pkl*  (using meshavatar code local)
8. Merge *template_hair.ply*, *template_body.ply*, *template_shoes.ply* and *template_skin.ply* into *template_body.ply*  (using meshavatar code local)
9. Set *template.yaml*, download *smpl_params.npz* and Generate *cano_weight_volume.npz* (using gen_data.gen_weight_volume.py local, check gender!!!!)
```
# in template.yaml
train:
  data:
    subject_name: 169
    data_dir: /home/zhiychen/Desktop/train_data/multiviewRGC/4d_dress/00169/Inner
```
10. Set shape to zero and generate *cano_weight_volume_shape_zero.npz* (using gen_data.gen_weight_volume.py local)
```
# line 80 and 81
# smpl_shape = torch.from_numpy(smpl_params['betas'][0]).to(torch.float32)
smpl_shape = torch.zeros((10,)).to(torch.float32)
```
11. Deform *beta_template_body_offser.ply* for all the poses in *smpl_params.npz* into the *body_mesh* folder (using meshavatar generate_meshes.py)
12. Upload *template_cloth_offset.ply*, *template_cloth_offset.pkl*, *template_body.ply*, *cano_weight_volume.npz* and *cano_weight_volume_shape_zero.npz* and *body_mesh*
13. Render *body_masks* for all cameras (gen_data.preprocess_4ddress.py)
```
mesh_path = os.path.join(new_data_dir, '%05d' % SUBJECT, LAYER, "body_mesh")
render_mesh(mesh_path, new_data_dir)
```
14. Generate *smpl_pos_map_body* with *cano_weight_volume.npz* (using gen_data.gen_pos_map.py)
```
# launch.json 
{
    "name": "gen_pos_maps",
    "type": "debugpy",
    "request": "launch",
    "program": "gen_data/gen_pos_maps.py",
    "console": "integratedTerminal",
    "args": [
        "-c",
        "configs/4d_dress/avatar.yaml",
        "-rc",
        "-o",
        "smpl_pos_map_body",
        "-t",
        "template_body",
        "-lw",
        "cano_weight_volume"
    ],

    "cwd": "/local/home/zhiychen/AnimatableGaussain/",
    "env": {
        "PYTHONPATH": "${cwd}",
        "CUDA_VISIBLE_DEVICES": "6",
        "OPENCV_IO_ENABLE_OPENEXR": "1"
    }
}
```
```
# avatar.yaml
train:
  dataset: MvRgbDataset4DDress
  data:
    layers: ["body", "cloth"]
    subject_name: 159
    data_dir: /data/zhiychen/AnimatableGaussain/train_data/multiviewRGC/4d_dress/00169/Inner
```
15. Generate *smpl_pos_map_cloth* with *cano_weight_volume_shape_zero_{gender}.npz* (using gen_data.gen_pos_map.py)
```
# launch.json 
{
    "name": "gen_pos_maps",
    "type": "debugpy",
    "request": "launch",
    "program": "gen_data/gen_pos_maps.py",
    "console": "integratedTerminal",
    "args": [
        "-c",
        "configs/4d_dress/avatar.yaml",
        "-rc",
        "-ro",
        "-o",
        "smpl_pos_map_cloth_rotate_offset_shape_zero",
        "-t",
        "template_cloth_offset",
        "-lw",
        "cano_weight_volume_shape_zero_{male/female}"
    ],

    "cwd": "/local/home/zhiychen/AnimatableGaussain/",
    "env": {
        "PYTHONPATH": "${cwd}",
        "CUDA_VISIBLE_DEVICES": "6",
        "OPENCV_IO_ENABLE_OPENEXR": "1"
    }
}
```

# Preparing Testing Data
1. Copy pose parameters and compress to a .npz file (gen_data.preprocess_4ddress.py)
```
pose_list = read_all_pose(data_dir)
copy_to_folder(new_data_dir, pose_list, ["pose"], "pkl")
combine_pose_to_npz(new_data_dir)
```
2. copy *cameras.pkl*, *template_body.ply*, *template_cloth_offset.ply*, *template_cloth.pkl*, *cano_weight_volume_shape_zero.npz*, *cano_weight_volume.npz* to the test data folder
3. Generate *smpl_pos_map_body* with *cano_weight_volume.npz* (using gen_data.gen_pos_map.py)
```
# launch.json 
{
    "name": "gen_pos_maps",
    "type": "debugpy",
    "request": "launch",
    "program": "gen_data/gen_pos_maps.py",
    "console": "integratedTerminal",
    "args": [
        "-c",
        "configs/4d_dress/avatar.yaml",
        "-rc",
        "-o",
        "smpl_pos_map_body",
        "-t",
        "template_body",
        "-lw",
        "cano_weight_volume"
    ],

    "cwd": "/local/home/zhiychen/AnimatableGaussain/",
    "env": {
        "PYTHONPATH": "${cwd}",
        "CUDA_VISIBLE_DEVICES": "6",
        "OPENCV_IO_ENABLE_OPENEXR": "1"
    }
}
```
4. Generate *smpl_pos_map_cloth* with *cano_weight_volume_shape_zero.npz* (using gen_data.gen_pos_map.py)
```
# launch.json 
{
    "name": "gen_pos_maps",
    "type": "debugpy",
    "request": "launch",
    "program": "gen_data/gen_pos_maps.py",
    "console": "integratedTerminal",
    "args": [
        "-c",
        "configs/4d_dress/avatar.yaml",
        "-rc",
        "-ro",
        "-o",
        "smpl_pos_map_cloth_rotate_offset_shape_zero",
        "-t",
        "template_cloth_offset",
        "-lw",
        "cano_weight_volume_shape_zero"
    ],

    "cwd": "/local/home/zhiychen/AnimatableGaussain/",
    "env": {
        "PYTHONPATH": "${cwd}",
        "CUDA_VISIBLE_DEVICES": "6",
        "OPENCV_IO_ENABLE_OPENEXR": "1"
    }
}
```
# Prepare Exchange Data

1. Do step 1-2 from Prepare Test Data