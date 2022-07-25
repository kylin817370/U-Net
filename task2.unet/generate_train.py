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


seed = 1
data_gen_args = dict(rotation_range=3,
                     width_shift_range=0.01,
                     height_shift_range=0.01,
                     shear_range=0.01,
                     zoom_range=0.01,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
print('build ImageDataGenerator finished.')

outputPath = "/home/huiying/Workspace/task/task2/task2.referrence/train_liver.h5"

if os.path.exists(outputPath):
    os.remove(outputPath)
dataset = HDF5DatasetWriter(image_dims=(2782, 512, 512, 1),
                            mask_dims=(2782, 512, 512, 1),
                            outputPath=outputPath)

print('build HDF5DatasetWriter finished')

for i in range(1, 17):  # 前16个人作为测试集
    full_images = []  # 后面用来存储目标切片的列表
    full_livers = []  # 功能同上

    label_path = '/mnt/sdc1/EZY/3Dircadb/3Dircadb1.%d/MASKS_DICOM/MASKS_DICOM/liver' % i
    data_path = '/mnt/sdc1/EZY/3Dircadb/3Dircadb1.%d/PATIENT_DICOM/PATIENT_DICOM' % i
    liver_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
    # 需要排序，即使文件夹中显示的是有序的，读进来后就是随机的了
    liver_slices.sort(key=lambda x: int(x.InstanceNumber))
    # s.pixel_array 获取dicom格式中的像素值
    livers = np.stack([s.pixel_array for s in liver_slices])

    image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
    image_slices.sort(key=lambda x: int(x.InstanceNumber))

    images = get_pixels_hu(image_slices)

    images = transform_ctdata(images, 500, 150)

    start, end = getRangImageDepth(livers)
    images = clahe_equalized(images, start, end)

    images /= 255.
    # 仅提取腹部所有切片中包含了肝脏的那些切片

    total = (end - 4) - (start + 4) + 1
    print("%d person, total slices %d" % (i, total))
    # 首和尾目标区域都太小，舍弃
    images = images[start + 5:end - 5]
    print("%d person, images.shape:(%d,)" % (i, images.shape[0]))

    livers[livers > 0] = 1

    livers = livers[start + 5:end - 5]

    full_images.append(images)
    full_livers.append(livers)

    full_images = np.vstack(full_images)
    full_images = np.expand_dims(full_images, axis=-1)
    full_livers = np.vstack(full_livers)
    full_livers = np.expand_dims(full_livers, axis=-1)

    image_datagen.fit(full_images, augment=True, seed=seed)
    mask_datagen.fit(full_livers, augment=True, seed=seed)
    image_generator = image_datagen.flow(full_images, seed=seed)
    mask_generator = mask_datagen.flow(full_livers, seed=seed)

    train_generator = zip(image_generator, mask_generator)
    x = []
    y = []
    i = 0
    for x_batch, y_batch in train_generator:
        i += 1
        x.append(x_batch)
        y.append(y_batch)
        if i >= 2:  # 不需要太多的数据
            break
    x = np.vstack(x)
    y = np.vstack(y)

    dataset.add(full_images, full_livers)
    dataset.add(x, y)
    print('add once finished.')

dataset.close()
