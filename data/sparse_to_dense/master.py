import os
import glob
import numpy as np
import scipy
import skimage
from pypardiso import spsolve
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import concurrent.futures


from fill_depth_colorization import fill_depth_colorization

def process_image(input_rgb_path, input_depth_path, output_depth_path):
    imgRgb = np.array(Image.open(input_rgb_path)) / 255.0
    imgDepthInput = np.array(Image.open(input_depth_path))

    output = fill_depth_colorization(imgRgb, imgDepthInput, alpha=1)

    denseDepth = np.interp(output, (output.min(), output.max()), (0, 255))
    denseDepth = Image.fromarray(np.uint8(denseDepth))
    denseDepth.save(output_depth_path)




def process_images_in_parallel(input_rgb_folder, input_depth_folder, output_folder, num_workers=None):
    os.makedirs(output_folder, exist_ok=True)

    input_rgb_files = sorted(glob.glob(os.path.join(input_rgb_folder, "*.png")))
    input_depth_files = sorted(glob.glob(os.path.join(input_depth_folder, "*.png")))

    if len(input_rgb_files) != len(input_depth_files):
        raise ValueError("The number of RGB images and depth images must be the same.")

    total_images = len(input_rgb_files)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image, input_rgb_path, input_depth_path,
                            os.path.join(output_folder, os.path.basename(input_depth_path)))
            for input_rgb_path, input_depth_path in zip(input_rgb_files, input_depth_files)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=total_images, unit="image"):
            future.result()


if __name__ == "__main__":
    input_rgb_folder = "D:/luizg/Documents/dataSets/kitti/depth_selection/test_depth_completion_anonymous/image"
    input_depth_folder = "D:/luizg/Documents/dataSets/kitti/depth_selection/test_depth_completion_anonymous/velodyne_raw"
    output_folder = "D:/luizg/Documents/dataSets/kitti/depth_selection/test_depth_completion_anonymous/dense_depth"

    process_images_in_parallel(input_rgb_folder, input_depth_folder, output_folder)
