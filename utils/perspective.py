import cv2
import matplotlib.pyplot as plt
import numpy as np

def transform_perspective(imgs):

    if type(imgs) != np.ndarray or type(imgs) != list:
        imgs = np.array(imgs)

    transformed_imgs = []
    Ms = []

    for img in imgs:
        src = np.float32([[435, img.shape[0]/2 + 200],
                          [895, img.shape[0]/2 + 200],
                          [1180, img.shape[0]],
                          [200, img.shape[0]]])
        dst = np.float32([[275, img.shape[0]/2 + 200],
                          [1055, img.shape[0]/2 + 200],
                          [1055 , img.shape[0]],
                          [275, img.shape[0]]])

        image_size = (img.shape[1], img.shape[0])

        M = cv2.getPerspectiveTransform(src, dst)
        Ms.append(M)
        transformed_img = cv2.warpPerspective(img, M, image_size, flags=cv2.INTER_LINEAR)
        transformed_imgs.append(transformed_img)

    return np.array(transformed_imgs), np.array(Ms)


def inverse_transform_perspective(orig_img, pers_img, left_fitx, right_fitx, ploty, M, plt_result=False):
    # Create base image
    orig_img_copy = np.copy(orig_img)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(pers_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.pinv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_img_copy.shape[1], orig_img_copy.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(orig_img_copy, 1, newwarp, 0.3, 0)
    if plt_result:
        plt.imshow(result)
    else:
        plt.imshow(result)
        return result