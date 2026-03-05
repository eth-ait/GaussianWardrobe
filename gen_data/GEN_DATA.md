# Training Dataset

## Data Organization
The structure of the training dataset we used is the following
```
data_dir
├── body_masks
|   └── 0004
│   └── 0016
├── images
|   └── 0004
│   └── 0016
├── labels
|   └── 0004
│   └── 0016
├── masks
│   └── 0004
│   └── 0016
├── cameras.pkl
├── smpl_params.npz
```


## Preprocessing

0. Reconstruct a template of the cloth using the label images.
- save it as ``template_body.py``

1. Reconstruct a template of the body using the smpl parameters.
- save it as ``template_cloth.py``

2. Given the pose and body shape from ``smpl_params.npz``, generate ``template_body_offset.py`` and ``template_cloth_offset.py``, which stores the body and cloth meshes in canonical shape-zero space. The side product ``template_body_offset.pkl`` and ``template_cloth_offset.pkl`` stores the shape offset information

3. Generate original beta-shaped and shape-zero weight volume:
    1. For shape-zero: ```gen_data/gen_weight_volume.py -c configs/4d_dress/template.yaml -z```
    2. For original-shape: ```gen_data/gen_weight_volume.py -c configs/4d_dress/template.yaml```

4. Generate position map for both body and cloth using:
    1.  ```gen_data/gen_pos_maps.py -c configs/4d_dress/avatar.yaml -rc -ro -o smpl_pos_map_offset_body -t template_body_offset -lw cano_weight_volume_shape_zero```
    2.  ```gen_data/gen_pos_maps.py -c configs/4d_dress/avatar.yaml -rc -ro -o smpl_pos_map_offset_cloth -t template_cloth_offset -lw cano_weight_volume_shape_zero```

# Testing Dataset

## Data Organization
The structure of the testing dataset we used is similar as the training data:

## Preprocessing

0. Reuse the following files from training dataset for the same subject
    - cameras.pkl
    - cano_weight_volume_shape_zero.npz
    - cano_weight_volume.npz
    - template_body_offset.py
    - template_cloth_offset.py
    - template_body_offset.pkl 
    - template_cloth_offset.pkl

1. Prepare the test pose sequence and save as ``smpl_params.npz``

2. Generate position map for both body and cloth using:
    0.  ```gen_data/gen_pos_maps.py -c configs/4d_dress/avatar.yaml -rc -ro -o smpl_pos_map_offset_body -t template_body_offset -lw cano_weight_volume_shape_zero```
    1.  ```gen_data/gen_pos_maps.py -c configs/4d_dress/avatar.yaml -rc -ro -o smpl_pos_map_offset_cloth -t template_cloth_offset -lw cano_weight_volume_shape_zero```


    


