# --coding:utf-8--
import os
import h5py
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HDF5DatasetGenerator:

    def __init__(self, dbpath, batch_size, preprocessors=None,
                 aug=None, binarize=True, classes=2):
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(dbpath, mode='r')
        self.numImages = self.db["images"].shape[0]
        # self.numImages = total
        print("total images:", self.numImages)
        self.num_batches_per_epoch = int((self.numImages - 1) / batch_size) + 1

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            shuffle_indices = np.arange(self.numImages)
            shuffle_indices = np.random.permutation(shuffle_indices)
            for batch_num in range(self.num_batches_per_epoch):

                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, self.numImages)

                # h5py get item by index,参数为list，而且必须是增序
                batch_indices = sorted(list(shuffle_indices[start_index:end_index]))

                images = self.db["images"][batch_indices, :, :, :]
                labels = self.db["masks"][batch_indices, :, :, :]

                if self.preprocessors is not None:
                    proc_images = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        proc_images.append(image)

                    images = np.array(proc_images)

                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels,
                                                          batch_size=self.batch_size))
                yield images, labels

            epochs += 1

    def close(self):
        self.db.close()
