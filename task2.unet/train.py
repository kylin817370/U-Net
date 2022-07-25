# --coding:utf-8--
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from skimage import io
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from HDF5DatasetWriter import HDF5DatasetWriter
from HDF5DatasetGenerator import HDF5DatasetGenerator
from Unet import *
from utils import *


def lr_schedule(epoch):
    return 0.0005 * 0.4**epoch


TOTAL = 2616  # 总共的训练数据
TOTAL_VAL = 152  # 总共的测试数据

outputPath = '/home/huiying/Workspace/task/task2/task2.referrence/train_liver.h5'  # 训练文件
val_outputPath = '/home/huiying/Workspace/task/task2/task2.referrence/val_liver.h5'

BATCH_SIZE = 2

reader = HDF5DatasetGenerator(dbpath=outputPath, batch_size=BATCH_SIZE)
train_iter = reader.generator()

val_reader = HDF5DatasetGenerator(dbpath=val_outputPath, batch_size=BATCH_SIZE)
val_iter = val_reader.generator()

model = get_unet()

model_checkpoint = ModelCheckpoint(
    filepath='/home/huiying/Workspace/task/task2/task2.referrence/models/weights_unet-{epoch:02d}-{val_loss:.2f}.h5',
    monitor='val_loss', save_best_only=False, save_weights_only=False)
learning_rate = np.array([lr_schedule(i) for i in range(500)])
reduce_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
early_stop = EarlyStopping(patience=4, verbose=1)
tensor_board = TensorBoard(log_dir='/home/huiying/Workspace/task/task2/task2.referrence/models/logs')
callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

model.fit(train_iter, steps_per_epoch=int(TOTAL / BATCH_SIZE), verbose=1, epochs=500, shuffle=True,
          validation_data=val_iter, validation_steps=int(TOTAL_VAL / BATCH_SIZE), callbacks=callbacks)

reader.close()
