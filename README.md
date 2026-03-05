
# <b> Gaussian Wardrobe: </b>:Compositional 3D Gaussian Avatars for Free-Form Virtual Try-on

**Abstract**: 

We introduce Gaussian Wardrobe, a novel framework to digitalize compositional 3D neural avatars from multi-view videos.
Existing methods for 3D neural avatars typically treat the human body and clothing as an inseparable entity. 
However, this paradigm fails to capture the dynamics of complex free-form garments and limits the reuse of clothing across different individuals.
To overcome these problems, we develop a novel, compositional 3D Gaussian representation to build avatars from multiple layers of free-form garments.
The core of our method is decomposing neural avatars into bodies and layers of shape-agnostic neural garments.
To achieve this, our framework learns to disentangle each garment layer from multi-view videos and canonicalizes it into a shape-independent space.
In experiments, our method models photorealistic avatars with high-fidelity dynamics, achieving new state-of-the-art performance on novel pose synthesis benchmarks.
In addition, we demonstrate that the learned compositional garments contribute to a versatile digital wardrobe, enabling a practical virtual try-on application where clothing can be freely transferred to new subjects.


# Installation
0. Clone this repo.
1. Install environments.
```
# install requirements
pip install -r requirements.txt

# install diff-gaussian-rasterization-depth-alpha
cd gaussians/diff_gaussian_rasterization_depth_alpha
python setup.py install
cd ../..

# install styleunet
cd network/styleunet
python setup.py install
cd ../..

# install pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

```
2. Download [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) model, and place pkl files to ```./smpl_files/smplx```.
3. Download [Lpips-weight](https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips/weights/v0.1) and place pth files to ```.network/lpips/weights/v0.1```

# Data Preparation
We have experimented with [4D-Dress](https://eth-ait.github.io/4d-dress/) and [ActorsHQ](https://www.actors-hq.com/dataset) datasets
Following [GEN_DATA.md](./gen_data/GEN_DATA.md)

Note for ActorsHQ dataset: 1. **SMPL-X Registration.** We used the smplx registration offered [here](https://drive.google.com/file/d/1DVk3k-eNbVqVCkLhGJhD_e9ILLCwhspR/view?usp=sharing) by [Animatable Gaussians](http://animatable-gaussians.github.io/)

# Avatar Training
Take the subject 00134 from [4D-Dress](https://eth-ait.github.io/4d-dress/) as an example:
0. Prepare the training dataset using the instruction from the previous step
1. Download its checkpoint or start from scratch
2. Set the corresponding data_dir and net_ckpt_dir in the train section in ./configs/4d_dress/avatar.yaml
3. Run:
```
python main_avatar.py -c configs/4d_dress/avatar.yaml --mode=train
```

# Avatar Animation
Take the subject 00134 from [4D-Dress](https://eth-ait.github.io/4d-dress/) as an example:
0. Download the checkpoint for the subject
1. Prepare the testing dataset according to [GEN_DATA.md](./gen_data/GEN_DATA.md)
2. Set the corresponding data_dir and prev_ckpt in the test section in ./configs/4d_dress/avatar.yaml
2. Run:
```
python main_avatar.py -c configs/4d_dress/avatar.yaml --mode=test
```

Some example test animation examples are provided under ``assets`` folder, e.g. 156_test.mp4 and 185_ours.mp4

# Virtual Try-on
Take the subject 00134 and 00140 from [4D-Dress](https://eth-ait.github.io/4d-dress/) as an example
We provided a script ``generate_pos_script.py`` for generating the exchange dataset for [4D-Dress](https://eth-ait.github.io/4d-dress/) subjects:
0. Update the macros in ``generate_pos_script.py``
1. Run ``generate_pos_script.py`` for the target combination
2. Run:
```
python main_avatar.py -c configs/4d_dress/exchange.yaml --mode=exchange_cloth 
```

Some example exchange examples are provided under ``assets`` folder, e.g. 185_134_full.mp4 and 127_134_outer.mp4

For other combinations please follow the format in the ``configs/4d_dress/exchange.yaml`` configuration files

# Evaluation
We provide evaluation metrics in [eval/eval_metrics.py](eval/eval_metrics.py).

0. Generate the testing pose images
1. Then update the data_dir macros in [eval/eval_metrics.py](eval/eval_metrics.py).
2. Run
```
python eval/eval_metrics.py
```
# Acknowledgement
Our code is based on the following repos:
- [Animatable Gaussians](http://animatable-gaussians.github.io/)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization) and its [adapted version](https://github.com/ashawkey/diff-gaussian-rasterization)
- [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar)


