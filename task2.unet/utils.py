# --coding:utf-8--
import re
import sys
import cv2
import numpy as np


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def transform_ctdata(image, windowwidth, windowcenter, normal=False):
    """
    注意，这个函数的self.image一定得是float类型的，否则就无效！
    return: trucated image according to window center and window width
    """
    minwindow = float(windowcenter) - 0.5 * float(windowwidth)
    new_img = (image - minwindow) / float(windowwidth)
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    if not normal:
        new_img = (new_img * 255).astype('uint8')
    return new_img


def getRangImageDepth(image):
    """
    args:
    image ndarray of shape (depth, height, weight)
    """
    first_flag = True
    start_position = 0
    end_position = 0
    for z in range(image.shape[0]):
        not_zero_flag = np.max(image[z])
        if not_zero_flag and first_flag:
            start_position = z
            first_flag = False
        if not_zero_flag:
            end_position = z
    return start_position, end_position


def clahe_equalized(imgs, start, end):
    assert (len(imgs.shape) == 3)  # 3D arrays
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(start, end + 1):
        imgs_equalized[i, :, :] = clahe.apply(np.array(imgs[i, :, :], dtype=np.uint8))
    return imgs_equalized
