# --coding:utf-8--
import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from HDF5DatasetWriter import HDF5DatasetWriter
from utils import *


outputPath = "/home/huiying/Workspace/task/task2/task2.referrence/val_liver.h5"
if os.path.exists(outputPath):
    os.remove(outputPath)

full_images2 = []
full_livers2 = []
for i in range(17, 20):  # 后3个人作为测试样本
    label_path = '/mnt/sdc1/EZY/3Dircadb/3Dircadb1.%d/MASKS_DICOM/MASKS_DICOM/liver' % i
    data_path = '/mnt/sdc1/EZY/3Dircadb/3Dircadb1.%d/PATIENT_DICOM/PATIENT_DICOM' % i
    liver_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
    liver_slices.sort(key=lambda x: int(x.InstanceNumber))
    livers = np.stack([s.pixel_array for s in liver_slices])
    start, end = getRangImageDepth(livers)
    total = (end - 4) - (start + 4) + 1
    print("%d person, total slices %d" % (i, total))

    image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
    image_slices.sort(key=lambda x: int(x.InstanceNumber))

    images = get_pixels_hu(image_slices)
    images = transform_ctdata(images, 500, 150)
    images = clahe_equalized(images, start, end)
    images /= 255.
    images = images[start + 5:end - 5]
    print("%d person, images.shape:(%d,)" % (i, images.shape[0]))
    livers[livers > 0] = 1
    livers = livers[start + 5:end - 5]

    full_images2.append(images)
    full_livers2.append(livers)

full_images2 = np.vstack(full_images2)
full_images2 = np.expand_dims(full_images2, axis=-1)
full_livers2 = np.vstack(full_livers2)
full_livers2 = np.expand_dims(full_livers2, axis=-1)

dataset = HDF5DatasetWriter(image_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            mask_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            outputPath=outputPath)

dataset.add(full_images2, full_livers2)

print("total images in val ", dataset.close())
