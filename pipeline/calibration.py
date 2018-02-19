import cv2
import glob
import numpy as np
from os import path


def read_images(img_dir, include_names=False, color_imgs=False, color_mode='BGR'):
    """
    Helper function to read chessboard images to calibrate the camera.

    Args:
        img_dir:  Path to the directory containing the chessboard images.
        include_names:  Return the file names.
        color_imgs:  Return RGB images.

    Returns:
        If `include_names`, return a list of (File name, Image) binaries.  Otherwise,
        return a list of image binaries.
    """
    img_files = path.join(img_dir, '*.jpg')
    file_names = glob.glob(img_files)
    if include_names and color_imgs:
        return [(file_name, cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)) for file_name in file_names]
    elif include_names:
        return [(file_name, cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY)) for file_name in file_names]
    elif color_imgs:
        if 'BGR':
            return [cv2.imread(file_name) for file_name in file_names]
        else:
            return [cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB) for file_name in file_names]
    else:
        return [cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY) for file_name in file_names]


def find_chessboard_corners(img_dir, chessboard_size=(9,6), data_for_drawing=False):
    """
    Locates the chessboard corners.

    Args:
        img_dir:  Path to the directory containing the chessboard images.

    Returns:
        [Object points], [Image points], [Grayscale images] lists.
    """
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_p = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    if data_for_drawing:
        used_imgs_indices = []

    grayscale_imgs = read_images(img_dir)
    for i, grayscale_img in enumerate(grayscale_imgs):
        ret, corners = cv2.findChessboardCorners(grayscale_img, chessboard_size)

        if ret:
            obj_points.append(obj_p)
            cv2.cornerSubPix(grayscale_img, corners, (11,11), (-1,-1), term_criteria)
            img_points.append(corners)

            if data_for_drawing:
                used_imgs_indices.append(i)

    if data_for_drawing:
        return used_imgs_indices, obj_points, img_points, grayscale_imgs
    return obj_points, img_points


def calibrate_camera(img_dir, img_shape, chessboard_size=(9,6), data_for_drawing=False):
    """
    Calibrates the camera based on chessboard images.

    Args:
         img_dir:  Path to the directory containing the chessboard images.
         include_names:  Return the file names.
         color_imgs:  Return RGB images.

    Returns:
        Returns the camera matrix, distortion coefficients, rotation and translation vectors,
        and [Grayscale images].
    """
    if data_for_drawing:
        used_imgs_indices, obj_points, img_points, grayscale_imgs = find_chessboard_corners(img_dir,
                                                                                            chessboard_size,
                                                                                            data_for_drawing)
    else:
        obj_points, img_points = find_chessboard_corners(img_dir, chessboard_size)

    if data_for_drawing:
        return cv2.calibrateCamera(obj_points, img_points, img_shape, None, None), grayscale_imgs
    return cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)


def undistort_chessboard_imgs(img_dir, img_shape, chessboard_size=(9,6)):

    (ret, matrix, dist, rotation_vecs, translation_vecs), grayscale_imgs = calibrate_camera(img_dir, img_shape,
                                                                                            chessboard_size, True)
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, img_shape, 1, img_shape)

    return [cv2.undistort(img, matrix, dist, None, new_matrix) for img in grayscale_imgs]


def undistort_imgs(calibration_dict, img_dir):

    color_imgs = read_images(img_dir, color_imgs=True)
    return [cv2.undistort(img, calibration_dict['matrix'], calibration_dict['dist'],
                          calibration_dict['new_matrix']) for img in color_imgs]