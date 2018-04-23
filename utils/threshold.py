import cv2
import numpy as np


def transform_sobel_value(value):
    abs_value = np.abs(value)
    return np.uint8(255*abs_value / np.max(abs_value))


def abs_sobel_threshold(imgs, orient='x', sobel_kernel=3, min_threshold=0, max_threshold=255):

    if type(imgs) != np.ndarray or type(imgs) != list:
        imgs = np.array(imgs)

    grayscale_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]

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


def direction_threshold(imgs, sobel_kernel=3, min_threshold=0, max_threshold=np.pi/2):

    if type(imgs) != np.ndarray or type(imgs) != list:
        imgs = np.array(imgs)

    grayscale_imgs = [cv2.cvtColor(img, cv2.COLOR_RBG2GRAY) for img in imgs]

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


def magnitude_threshold(imgs, sobel_kernel=3, min_threshold=0, max_threshold=255):

    if type(imgs) != np.ndarray or type(imgs) != list:
        imgs = np.array(imgs)

    grayscale_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]

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


def hls_threshold(channel, imgs, min_threshold=0, max_threshold=255):

    if type(imgs) != np.ndarray or type(imgs) != list:
        imgs = np.array(imgs)

    saturation_binaries = []

    channel_index = {'h': 0, 'l': 1, 's': 2}

    for img in imgs:
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls_img[:,:,channel_index[channel]]
        binary = np.zeros_like(s_channel)
        binary[(s_channel > min_threshold) & (s_channel <= max_threshold)] = 1
        saturation_binaries.append(binary)

    return np.array(saturation_binaries)


def yellow_threshold(imgs, min_threshold=0, max_threshold=255):

    if type(imgs) != np.ndarray or type(imgs) != list:
        imgs = np.array(imgs)

    yellow_binaries = []

    for img in imgs:
        r_channel = img[:,:,0]
        g_channel = img[:,:,1]
        binary = np.zeros_like(r_channel)
        binary[((r_channel > min_threshold) & (r_channel <= max_threshold)) &
               ((g_channel > min_threshold) & (g_channel <= max_threshold))] = 1
        yellow_binaries.append(binary)

    return yellow_binaries


def threshold_transforms(imgs, sobel_config_dict):

    if type(imgs) != np.ndarray or type(imgs) != list:
        imgs = np.array(imgs)

    abs_threshold_x = sobel_config_dict['abs_sobel_orient_x']
    abs_threshold_y = sobel_config_dict['abs_sobel_orient_y']
    mag_threshold = sobel_config_dict['mag']
    dir_threshold = sobel_config_dict['dir']
    light_threshold = sobel_config_dict['lightness']
    sat_threshold = sobel_config_dict['saturation']
    color_threshold = sobel_config_dict['yellow']

    if abs_threshold_x:
        sobel_binaries_x = abs_sobel_threshold(imgs,
                                               'x',
                                               sobel_config_dict['abs_sobel_kernel'],
                                               sobel_config_dict['abs_sobel_min'],
                                               sobel_config_dict['abs_sobel_max'])

    if abs_threshold_y:
        sobel_binaries_y = abs_sobel_threshold(imgs,
                                               'y',
                                               sobel_config_dict['abs_sobel_kernel'],
                                               sobel_config_dict['abs_sobel_min'],
                                               sobel_config_dict['abs_sobel_max'])

    if mag_threshold:
        magnitude_binaries = magnitude_threshold(imgs,
                                                 sobel_config_dict['mag_kernel'],
                                                 sobel_config_dict['mag_min'],
                                                 sobel_config_dict['mag_max'])

    if dir_threshold:
        direction_binaries = direction_threshold(imgs,
                                                 sobel_config_dict['dir_kernel'],
                                                 sobel_config_dict['dir_min'],
                                                 sobel_config_dict['dir_max'])

    if light_threshold:
        light_binaries = hls_threshold('l',
                                       imgs,
                                       sobel_config_dict['light_min'],
                                       sobel_config_dict['light_max'])

    if sat_threshold:
        sat_binaries = hls_threshold('s',
                                     imgs,
                                     sobel_config_dict['sat_min'],
                                     sobel_config_dict['sat_max'])

    if color_threshold:
        yellow_binaries = yellow_threshold(imgs,
                                           sobel_config_dict['yellow_min'],
                                           sobel_config_dict['yellow_max'])

    combined_binaries = []
    for i, sobel_binary_x in enumerate(sobel_binaries_x):
        combined_binary = np.zeros_like(sobel_binary_x)
        if abs_threshold_x & abs_threshold_y:
            sobel_binary_y = sobel_binaries_y[i]
            combined_binary[(sobel_binary_x == 1) & (sobel_binary_y == 1)] = 1
        if mag_threshold:
            mag_binary = magnitude_binaries[i]
            combined_binary[mag_binary == 1] = 1
        if dir_threshold:
            dir_binary = direction_binaries[i]
            combined_binary[dir_binary == 1] = 1
        if light_threshold and yellow_threshold:
            light_binary = light_binaries[i]
            yellow_binary = yellow_binaries[i]
            combined_binary[(light_binary == 1) & (yellow_binary == 1)] = 1
        if sat_threshold:
            sat_binary = sat_binaries[i]
            combined_binary[sat_binary == 1] = 1
        combined_binaries.append(combined_binary)

    return np.array(combined_binaries)