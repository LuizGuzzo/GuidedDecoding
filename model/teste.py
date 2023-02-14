import numpy as np
import cv2

def depth_estimation(left_img, window_size=5):
    """
    Estimate depth map from a single image.
    :param left_img: grayscale image
    :param window_size: size of the window to use for SAD comparison
    :return: depth map
    """
    depth_map = np.zeros_like(left_img, dtype=np.float32)
    pad = window_size // 2

    # Shift the image in x direction and compute the SAD for each pixel
    for i in range(pad, left_img.shape[1] - pad):
        for j in range(pad, left_img.shape[0] - pad):
            window = left_img[j - pad:j + pad + 1, i - pad:i + pad + 1]
            min_sad = np.inf
            min_disp = 0

            # For each disparity level
            for d in range(1, window_size + 1):
                sad = np.sum(np.abs(window - left_img[j, i - d:i - d + window_size]))
                if sad < min_sad:
                    min_sad = sad
                    min_disp = d

            depth_map[j, i] = min_disp

    return depth_map

# Load an image
img = cv2.imread("C:/Users/luizg/Documents/repositorios/GuidedDecoding/model/tabby_tiger_cat.jpg", 0)

# Estimate depth map
depth_map = depth_estimation(img)

# Save the depth map
cv2.imwrite('depth_map.jpg', depth_map)
