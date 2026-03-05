# To compute FID, first install pytorch_fid
# pip install pytorch-fid

import os
import cv2 as cv
from tqdm import tqdm
import shutil

from eval.score import *

cam_id = 18
AG_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/actor01_novel_view/actor01_AG/cam_0128/batch_204961/both/rgb_map'
ours_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/actor01_novel_view/actor01_body_loss/cam_0128/batch_231281/both/rgb_map'
layga_dir = '/local/home/zhiychen/AnimatableGaussain/test_results/actor01_novel_view/actor01_layga/cam_0128/batch_185432/both/rgb_map'
gt_dir = '/data/zhiychen/ActorHQ/Actor01/Sequence1/4x/rgbs/Cam128' 
mask_dir = '/data/zhiychen/ActorHQ/Actor01/Sequence1/4x/masks/Cam128'

frame_list = list(range(46, 900, 2)) + list(range(1200, 2177, 2))


if __name__ == '__main__':
    ours_metrics = Metrics()
    # posevocab_metrics = Metrics()
    # slrf_metrics = Metrics()
    layga_metrics = Metrics()
    # tava_metrics = Metrics()
    AG_metrics = Metrics()


    os.makedirs('./tmp_quant', exist_ok=True)
    shutil.rmtree('./tmp_quant')
    os.makedirs('./tmp_quant/ours', exist_ok = True)
    os.makedirs('./tmp_quant/AG', exist_ok=True)
    os.makedirs('./tmp_quant/gt', exist_ok = True)
    AG_psnr = []
    ours_psnr = []
    for frame_id in tqdm(frame_list):
        ours_img = (cv.imread(ours_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        AG_img = (cv.imread(AG_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        layga_img = (cv.imread(layga_dir + '/%08d.png' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        gt_img = (cv.imread(gt_dir + '/Cam128_rgb%06d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        mask_img = cv.imread(mask_dir + '/Cam128_mask%06d.png' % frame_id, cv.IMREAD_UNCHANGED) > 128
        gt_img[~mask_img] = 1.
        ours_img[~mask_img] = 1.
        AG_img[~mask_img] = 1.
        layga_img[~mask_img] = 1.
        ours_img_cropped, layga_img_cropped, AG_img_cropped, gt_img_cropped = \
            crop_image(
                mask_img,
                512,
                ours_img,
                layga_img,
                AG_img,
                gt_img
            )

        cv.imwrite('./tmp_quant/ours/%08d.png' % frame_id, (ours_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/AG/%08d.png' % frame_id, (AG_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/layga/%08d.png' % frame_id, (layga_img_cropped * 255).astype(np.uint8))
        cv.imwrite('./tmp_quant/gt/%08d.png' % frame_id, (gt_img_cropped * 255).astype(np.uint8))

        if ours_img is not None:
            ours_metrics.psnr += compute_psnr(ours_img, gt_img)
            ours_metrics.ssim += compute_ssim(ours_img, gt_img)
            ours_metrics.lpips += compute_lpips(ours_img_cropped, gt_img_cropped)
            ours_metrics.count += 1

        if AG_img is not None:
            AG_metrics.psnr += compute_psnr(AG_img, gt_img)
            AG_metrics.ssim += compute_ssim(AG_img, gt_img)
            AG_metrics.lpips += compute_lpips(AG_img_cropped, gt_img_cropped)
            AG_metrics.count += 1

        if layga_img is not None:
            layga_metrics.psnr += compute_psnr(layga_img, gt_img)
            layga_metrics.ssim += compute_ssim(layga_img, gt_img)
            layga_metrics.lpips += compute_lpips(layga_img_cropped, gt_img_cropped)
            layga_metrics.count += 1
    print('Ours metrics: ', ours_metrics)
    print('AG metrics: ', AG_metrics)
    print('LayGA metrics', layga_metrics)

    print('--- Ours ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/ours', './tmp_quant/gt'))
    print('--- AG ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/AG', './tmp_quant/gt'))
    print('--- PoseVocab ---')
    os.system('python -m pytorch_fid --device cuda {} {}'.format('./tmp_quant/layga', './tmp_quant/gt'))
