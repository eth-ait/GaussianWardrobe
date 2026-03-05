
import os
import imageio
import numpy as np


def create_video_with_imageio(image_paths, output_path, fps=24):
    if not image_paths:
        raise ValueError("Image path list is empty.")

    # Open the FFmpeg writer explicitly
    with imageio.get_writer(output_path, fps=fps) as writer:
        for path in image_paths:
            try:
                image = imageio.imread(path)
                writer.append_data(image)
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
    
    print(f"Video saved to {output_path}")

# Define the directory containing images
image_folder = "/local/home/zhiychen/AnimatableGaussain/test_results/00185_no_virtual_bone/layered_full_stage_1_less_opacity_loss/cam_0001/batch_239040/both/rgb_map"
video_name = './video/00185_no_virtual_bones.mp4'
# Get all the image files in the folder
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png")]
# images = [img for img in os.listdir(image_folder) if img.endswith(“normal.png”) or img.endswith(“.jpg”)]
images.sort()  # Optional, to ensure the images are in the correct sequence
# front = [images[0]]*30
# images = front + images
create_video_with_imageio(images, video_name, fps=30)
print(f'Video {video_name} created successfully.')