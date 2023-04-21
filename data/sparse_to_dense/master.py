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
import re


from fill_depth_colorization import fill_depth_colorization

def process_image(input_rgb_path, input_depth_path, output_depth_path):
    imgRgb = np.array(Image.open(input_rgb_path)) / 255.0
    imgDepthInput = np.array(Image.open(input_depth_path))

    output = fill_depth_colorization(imgRgb, imgDepthInput, alpha=1)

    denseDepth = np.interp(output, (output.min(), output.max()), (0, 255))
    denseDepth = Image.fromarray(np.uint8(denseDepth))
    denseDepth.save(output_depth_path)




def find_matching_rgb_image(depth_path, input_rgb_folder):
    # Extract the common part of the path, e.g., "2011_09_26_drive_0001_sync"
    match = re.search(r'(\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)', depth_path)
    if not match:
        raise ValueError(f"Could not find a matching pattern in the depth path: {depth_path}")

    common_part = match.group(1)
    image_number = os.path.basename(depth_path)

    # Construct the path to the corresponding RGB image
    rgb_path = os.path.join(input_rgb_folder, common_part, f"{common_part[:10]}", common_part, "image_02", "data", image_number)
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"Could not find the corresponding RGB image for the depth path: {depth_path}")

    return rgb_path



def process_images_in_parallel(input_rgb_folder, input_depth_folder, output_folder, num_workers=None):
    os.makedirs(output_folder, exist_ok=True)

    input_depth_files = sorted(glob.glob(os.path.join(input_depth_folder, "**", "*.png"), recursive=True))

    total_images = len(input_depth_files)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for input_depth_path in input_depth_files:
            # Find the corresponding RGB image
            input_rgb_path = find_matching_rgb_image(input_depth_path, input_rgb_folder)

            # Compute the output depth file path while preserving the directory structure.
            relative_depth_path = os.path.relpath(input_depth_path, input_depth_folder)
            output_depth_path = os.path.join(output_folder, relative_depth_path)
            os.makedirs(os.path.dirname(output_depth_path), exist_ok=True)

            futures.append(executor.submit(process_image, input_rgb_path, input_depth_path, output_depth_path))

        for future in tqdm(concurrent.futures.as_completed(futures), total=total_images, unit="image"):
            future.result()


if __name__ == "__main__":
    main_path = "D:/luizg/Documents/dataSets/pasta_KITTI/KITTI/teste"
    input_rgb_folder = main_path+"/data_raw"
    input_depth_folder = main_path+"/data_depth_annotated"
    output_folder = main_path+"/denseDepth"

    process_images_in_parallel(input_rgb_folder, input_depth_folder, output_folder)
