import cv2
import numpy as np


def transform_sobel_value(value):
    abs_value = np.abs(value)
    return np.uint8(255*abs_value / np.max(abs_value))


def abs_sobel_threshold(grayscale_imgs, orient='x', sobel_kernel=3, min_threshold=0, max_threshold=255):
    if orient == 'x':
        scaled_gradients = [transform_sobel_value(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
                            for img in grayscale_imgs]
    else:
        scaled_gradients = [transform_sobel_value(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
                            for img in grayscale_imgs]

    sobel_binaries = []
    for scaled_gradient in scaled_gradients:
        sobel_binary = np.zeros_like(scaled_gradient)
        sobel_binary[(min_threshold <= scaled_gradient) & (scaled_gradient <= max_threshold)] = 1
        sobel_binaries.append(sobel_binary)

    return np.array(sobel_binaries)


def magnitude_threshold(grayscale_imgs, sobel_kernel=3, min_threshold=0, max_threshold=255):
    x_abs_sobels = [np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)) for img in grayscale_imgs]
    y_abs_sobels = [np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)) for img in grayscale_imgs]

    scaled_magnitudes = [transform_sobel_value(np.sqrt(sobel_pair[0]**2 + sobel_pair[1]**2))
                         for sobel_pair in zip(x_abs_sobels, y_abs_sobels)]

    sobel_magnitude_binaries = []
    for scaled_magnitude in scaled_magnitudes:
        magnitude_binary = np.zeros_like(scaled_magnitude)
        magnitude_binary[(min_threshold <= scaled_magnitude) & (scaled_magnitude <= max_threshold)] = 1
        sobel_magnitude_binaries.append(magnitude_binary)

    return np.array(sobel_magnitude_binaries)


def direction_threshold(grayscale_imgs, sobel_kernel=3, min_threshold=0, max_threshold=np.pi/2):
    x_sobels = [cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel) for img in grayscale_imgs]
    y_sobels = [cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel) for img in grayscale_imgs]

    abs_direction_gradients = [np.arctan2(np.abs(sobel_pair[0]), np.abs(sobel_pair[1]))
                               for sobel_pair in zip(x_sobels, y_sobels)]

    sobel_direction_binaries = []
    for direction_gradient in abs_direction_gradients:
        direction_binary = np.zeros_like(direction_gradient)
        direction_binary[(min_threshold <= direction_gradient) & (direction_gradient <= max_threshold)] = 1
        sobel_direction_binaries.append(direction_binary)

    return np.array(sobel_direction_binaries)

#
# def sobel_based_transforms(grayscale_imgs, sobel_config_dict):
#     abs_threshold = False
#     mag_threshold = False
#     dir_threshold = False
#
#     if 'abs_sobel_orient' in sobel_config_dict:
#         abs_threshold = True
#         sobel_binaries = abs_sobel_threshold(grayscale_imgs, sobel_config_dict['abs_sobel_orient'],
#                                              sobel_config_dict['abs_sobel_kernel'], sobel_config_dict['abs_sobel_min'],
#                                              sobel_config_dict['abs_sobel_max'])
#     if 'mag_kernel' in sobel_config_dict:
#         mag_threshold = True
#         magnitude_binaries = magnitude_threshold(grayscale_imgs, sobel_config_dict['mag_kernel'],
#                                                  sobel_config_dict['mag_min'], sobel_config_dict['mag_max'])
#
#     if 'dir_kernel' in sobel_config_dict:
#         dir_threshold = True
#         direction_binaries = direction_threshold(grayscale_imgs, sobel_config_dict['dir_kernel'],
#                                                  sobel_config_dict['dir_min'], sobel_config_dict['dir_max'])
#
#     if abs_threshold and mag_threshold and dir_threshold:
#         combined_binaries = []
#         for i, sobel_binary in enumerate(sobel_binaries):
#             mag_binary = magnitude_binaries[i]
#             dir_binary = direction_binaries[i]
#
#             combined_binary = np.zeros_like(sobel_binary)
#             combined_binary[()]