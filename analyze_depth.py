from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_cumulative_distribution(image_path, save_path):
    # Load and convert image to RGB
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Flatten channels
    cumulative_data = {}
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        channel = img_array[..., i].flatten()
        
        # Compute histogram
        hist, bins = np.histogram(channel, bins=256, range=(0, 255))
        
        # Cumulative sum: number of pixels ≤ each value
        cdf = np.cumsum(hist)
        cumulative_data[color] = cdf

    # Plotting
    plt.figure(figsize=(10, 6))
    for color in ['Red', 'Green', 'Blue']:
        plt.plot(range(256), cumulative_data[color], label=f'{color} channel', color=color.lower())

    plt.title('Cumulative Pixel Count per Channel')
    plt.xlabel('Pixel Intensity (0–255)')
    plt.ylabel('Number of Pixels ≤ Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def get_rapid_increase_values(image_path, save_path, threshold_ratio=0.01):
    # Load image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    height, width, _ = img_array.shape
    total_pixels = height * width

    print(f"Image size: {width}x{height} ({total_pixels} total pixels)")

    result = {}

    channel = img_array[..., 0].flatten()
    channel = channel[channel < 255]
    # Compute histogram and CDF
    hist, _ = np.histogram(channel, bins=256, range=(0, 255))
    cdf = np.cumsum(hist)

    # Derivative (how fast pixel count increases at each value)
    diff = np.diff(cdf, prepend=0)

    # Normalize diff to be a ratio of total pixels
    diff_ratio = diff / total_pixels
    plt.plot(range(256), diff)

    plt.title('Derivative of Cumulative Distribution (Pixel Count Change)')
    plt.xlabel('Pixel Intensity (0–255)')
    plt.ylabel('Increase of Pixels at Each Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Derivative plot saved to: {save_path}")
    plt.close()

    # Find where the slope is above a threshold
    rapid_indices = np.where(diff_ratio > threshold_ratio)[0]

    result = {
        'rapid_increase_values': rapid_indices.tolist(),
        'max_increase_value': int(np.argmax(diff)),
        'max_increase_count': int(diff.max()),
    }

    print(f"Rapid changes at pixel values {rapid_indices.tolist()}")
    print(f"Max increase at value {np.argmax(diff)} with {diff.max()} pixels")

    return result

if __name__ == "__main__":
    image_path = "/local/home/zhiychen/AnimatableGaussain/test_results/render_output/neutral_new_pose_param/cam_0000/batch_156421/both/depth_body_map/depth_body_map_00000004.jpg"
    save_path = "depth_cdf.png"
    get_rapid_increase_values(image_path, save_path)



