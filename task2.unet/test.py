# --coding:utf-8--
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from skimage import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from HDF5DatasetWriter import HDF5DatasetWriter
from HDF5DatasetGenerator import HDF5DatasetGenerator
from utils import *
from Unet import *


TOTAL = 2616  # 总共的训练数据
TOTAL_VAL = 152  # 总共的测试数据

outputPath = '/home/huiying/Workspace/task/task2/task2.referrence/train_liver.h5'  # 训练文件
val_outputPath = '/home/huiying/Workspace/task/task2/task2.referrence/val_liver.h5'

BATCH_SIZE = 10

print('-' * 30)
print('Loading and preprocessing test data...')
test_reader = HDF5DatasetGenerator(dbpath=val_outputPath, batch_size=BATCH_SIZE)
test_iter = test_reader.generator()
fixed_test_images, fixed_test_masks = test_iter.__next__()
print('-' * 30)

print('-' * 30)
model = get_unet()
print('Loading saved weights...')
print('-' * 30)
model.load_weights('/home/huiying/Workspace/task/task2/task2.referrence/models/weights_unet-06--0.87.h5')

print('-' * 30)
print('Predicting masks on test data...')
imgs_mask_test = model.predict(fixed_test_images, verbose=1)
print('-' * 30)

print('-' * 30)
print('Saving predicted masks to files...')
np.save('imgs_mask_test.npy', imgs_mask_test)
print('-' * 30)

pred_dir = 'preds'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

i = 0
for image in imgs_mask_test:
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    gt = (fixed_test_masks[i, :, :, 0] * 255.).astype(np.uint8)
    ini = (fixed_test_images[i, :, :, 0] * 255.).astype(np.uint8)
    io.imsave(os.path.join(pred_dir, str(i) + '_ini.png'), ini)
    io.imsave(os.path.join(pred_dir, str(i) + '_pred.png'), image)
    io.imsave(os.path.join(pred_dir, str(i) + '_gt.png'), gt)
    i += 1

print("total images in test ", str(i))
