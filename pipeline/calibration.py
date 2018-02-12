import cv2
import glob
import numpy as np
from os import path


def read_chessboard_images(img_dir, include_names=False, color_imgs=False):
    """
    Helper function to read chessboard images to calibrate the camera.

    Args:
        img_dir:  Path to the directory containing the chessboard images.

    Returns:
        If `include_names`, return a list of (File name, Grayscale Image) binaries.  Otherwise,
        return a list of grayscale image binaries.
    """
    img_files = path.join(img_dir, '*.jpg')
    file_names = glob.glob(img_files)
    if include_names and color_imgs:
        return [(file_name, cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)) for file_name in file_names]
    elif include_names:
        return [(file_name, cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY)) for file_name in file_names]
    elif color_imgs:
        return [cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB) for file_name in file_names]
    else:
        return [cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY) for file_name in file_names]


def find_chessboard_corners(img_dir, chessboard_size=(9,6), data_for_drawing=False):
    """
    Locates the chessboard corners from the .

    Args:
        img_dir:  Path to the directory containing the chessboard images.

    Returns:
        [Object points] and [Image points] lists.
    """
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_p = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    if data_for_drawing:
        used_imgs_indices = []

    grayscale_imgs = read_chessboard_images(img_dir)
    for i, grayscale_img in enumerate(grayscale_imgs):
        ret, corners = cv2.findChessboardCorners(grayscale_img, chessboard_size)

        if ret:
            obj_points.append(obj_p)
            cv2.cornerSubPix(grayscale_img, corners, (11,11), (-1,-1), term_criteria)
            img_points.append(corners)

            if data_for_drawing:
                used_imgs_indices.append(i)

    if data_for_drawing:
        return used_imgs_indices, obj_points, img_points
    return obj_points, img_points


def calibrate_camera(img_dir, img_shape, chessboard_size=(9,6)):
    obj_points, img_points = find_chessboard_corners(img_dir, chessboard_size)

    return cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)


# def undistort_imgs(img_dir, img_shape, chessboard_size=(9,6)):
#     ret, matrix, dist, rotation_vecs, translation_vecs = calibrate_camera(img_dir, img_shape, chessboard_size)
#     matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, img_shape, 1, img_shape)
#
#     return