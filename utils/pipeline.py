import cv2
import numpy as np
from utils.calibration import calibrate_camera, read_images, undistort_imgs
from utils.detection import hist_lane_detection
from utils.perspective import inverse_transform_perspective, transform_perspective
from utils.threshold import threshold_transforms
import matplotlib.pyplot as plt

class Pipeline():
    def __init__(self):
        self.sobel_config = {
            'abs_sobel_orient_x': True,
            'abs_sobel_orient_y': False,
            'abs_sobel_kernel': 9,
            'abs_sobel_min': 40,
            'abs_sobel_max': 100,
            'mag': False,
            'mag_kernel': 9,
            'mag_min': 80,
            'mag_max': 200,
            'dir': False,
            'dir_kernel': 9,
            'dir_min': 0.01,
            'dir_max': 0.1,
            'lightness': True,
            'light_min': 150,
            'light_max': 255,
            'saturation': True,
            'sat_min': 170,
            'sat_max': 255,
            'yellow': True,
            'yellow_min': 200,
            'yellow_max': 255
        }
        self.calibration = {
            'matrix': None,
            'dist': None,
            'new_matrix': None
        }
        self.frame_index = 0
        self.left_array = []
        self.right_array = []

    def set_camera_calibration(self, chessboard_dir):
        chessboard_imgs = read_images(img_dir=chessboard_dir, color_imgs=True)
        chessboard_img_shape = chessboard_imgs[1].shape[1::-1]
        _, matrix, dist, _, _ = calibrate_camera(chessboard_dir, chessboard_img_shape)
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, dist, chessboard_img_shape, 1, chessboard_img_shape)
        self.calibration['matrix'] = matrix
        self.calibration['dist'] = dist
        self.calibration['new_matrix'] = new_matrix

    def process_img(self, img):
        undistorted_img = undistort_imgs(img, calibration_dict=self.calibration, color_mode='RGB')
        img_threshold = threshold_transforms(undistorted_img, self.sobel_config)
        transformed_imgs, Ms = transform_perspective(img_threshold)

        # Parse transformed image and matrix
        transformed_img = transformed_imgs[0]
        M = Ms[0]

        fit_data, position_data = hist_lane_detection(transformed_img)

        # Parse fit and position data from detections
        left_fitx, right_fitx, ploty = fit_data

        if self.frame_index < 10:
            self.left_array.append(left_fitx)
            self.right_array.append(right_fitx)
            self.frame_index += 1
        else:
            self.left_array.pop(0)
            self.left_array.append(left_fitx)

            self.right_array.pop(0)
            self.right_array.append(right_fitx)
            self.frame_index += 1

        avg_left_fitx = np.mean(self.left_array, axis=0)
        avg_right_fitx = np.mean(self.right_array, axis=0)

        curvature_rad, dist_from_middle = position_data

        # Plot lane detection on original images
        undistorted_img = undistorted_img[0]
        img_result = inverse_transform_perspective(undistorted_img, transformed_img, avg_left_fitx, avg_right_fitx,
                                                   ploty, M)
        cv2.putText(img_result, 'Radius of curvature: {} m'.format(round(curvature_rad, 5)), (100, 80),
                    cv2.QT_FONT_NORMAL, 2, (255, 255, 255), 3)
        dist_from_middle = round(dist_from_middle, 5)
        if dist_from_middle == 0:
            cv2.putText(img_result, 'Vehicle is at center', (100, 150), cv2.QT_FONT_NORMAL, 2, (255, 255, 255), 3)
        elif dist_from_middle > 0:
            cv2.putText(img_result, 'Vehicle is {} m right of center'.format(dist_from_middle), (100, 150),
                        cv2.QT_FONT_NORMAL, 2, (255, 255, 255), 3)
        else:
            cv2.putText(img_result, 'Vehicle is {} m left of center'.format(dist_from_middle), (100, 150),
                        cv2.QT_FONT_NORMAL, 2, (255, 255, 255), 3)
        return img_result
