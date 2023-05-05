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

def process_image(input_rgb_path, input_depth_path, output_depth_path, error_log=None):
    try:
        imgRgb = np.array(Image.open(input_rgb_path)) / 255.0
        imgDepthInput = np.array(Image.open(input_depth_path))

        output = fill_depth_colorization(imgRgb, imgDepthInput, alpha=1)

        denseDepth = np.interp(output, (output.min(), output.max()), (0, 255))
        denseDepth = Image.fromarray(np.uint8(denseDepth))
        denseDepth.save(output_depth_path)
    except Exception as e:
        if error_log is not None:
            with open(error_log, 'a') as log_file:
                log_file.write(f"Error processing {input_depth_path}: {e}\n")


def find_matching_rgb_image(input_depth_path, rgb_folder):
    depth_file_name = os.path.basename(input_depth_path)
    potential_folders = glob.glob(os.path.join(rgb_folder, "**", "image_02", "data"), recursive=True)
    for folder in potential_folders:
        potential_file_path = os.path.join(folder, depth_file_name)
        if os.path.exists(potential_file_path):
            return potential_file_path
    return None


def process_images_in_parallel(input_rgb_folder, input_depth_folder, output_folder, num_workers=None):
    os.makedirs(output_folder, exist_ok=True)

    input_depth_files = sorted(glob.glob(os.path.join(input_depth_folder, "**", "*.png"), recursive=True))

    processed_images = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for input_depth_path in input_depth_files:
            # Find the corresponding RGB image
            input_rgb_path = find_matching_rgb_image(input_depth_path, input_rgb_folder)

            # Compute the output depth file path while preserving the directory structure.
            relative_depth_path = os.path.relpath(input_depth_path, input_depth_folder)
            output_depth_path = os.path.join(output_folder, relative_depth_path)
            os.makedirs(os.path.dirname(output_depth_path), exist_ok=True)

            # Skip the image if it has already been processed.
            if os.path.exists(output_depth_path):
                continue

            futures.append(executor.submit(process_image, input_rgb_path, input_depth_path, output_depth_path, error_log))
            processed_images += 1

        for future in tqdm(concurrent.futures.as_completed(futures), total=processed_images, unit="image"):
            future.result()



if __name__ == "__main__":
    main_path = "D:/luizg/Documents/dataSets/pasta_KITTI/KITTI/teste"
    input_rgb_folder = main_path+"/data_raw"
    input_depth_folder = main_path+"/data_depth_annotated"
    output_folder = main_path+"/denseDepth"
    error_log = main_path + "/error_log.txt"
    num_workers = None

    process_images_in_parallel(input_rgb_folder, input_depth_folder, output_folder, num_workers)

