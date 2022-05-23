import tensorflow as tf
import keras
import scipy.io
import numpy as np
import pandas as pd
import keras
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from keras.preprocessing.image import ImageDataGenerator

def preprocess_data_facenet_without_aging(X_train):
  X_train = X_train.astype('float32')

  return X_train

def get_augmented_datasets():
    augmentation_generator = ImageDataGenerator(horizontal_flip=False, # Randomly flip images
                                        vertical_flip=False, # Randomly flip images
                                        rotation_range = None,
                                        validation_split=0.25,
                                        brightness_range=None,
                                        preprocessing_function=preprocess_data_facenet_without_aging) #Randomly rotate

    return augmentation_generator

class ViTAgeDBDataLoader(AgeDBDataset):

    def __init__(self, logger, metadata_file,
               list_IDs, color_mode='grayscale', augmentation_generator=None, data_dir=None, classes=[], batch_size=64, dim=(72*72), n_channels=1, n_classes=2, shuffle=True, valid=True):

        super(ViTAgeDBDataLoader, self).__init__(logger, metadata_file,
               list_IDs, batch_size=batch_size, dim=dim, n_channels=n_channels, n_classes=n_classes, shuffle=shuffle, valid=valid)

        self.iterator = self.get_iterator(color_mode, batch_size, data_dir, augmentation_generator, classes, x_col='filename',
                                          y_col='identity')

    def get_iterator(self, color_mode, batch_size, data_dir, augmentation_generator, classes=None, x_col='filename', y_col='name'):
        train_iterator = augmentation_generator.flow_from_dataframe(self.metadata,
                                                                    x_col=x_col,
                                                                    y_col=y_col,
                                                                    directory=data_dir,
                                                                    target_size=(160, 160),
                                                                    color_mode=color_mode,
                                                                    batch_size=batch_size,
                                                                    subset='training',
                                                                    class_mode='categorical', classes=classes,
                                                                    shuffle=False)

        return train_iterator

    def get_validation_iterator(self, color_mode, batch_size, data_dir, augmentation_generator, classes=None, x_col='filename', y_col='name'):
        validation_iterator = augmentation_generator.flow_from_dataframe(self.metadata,
                                                                    x_col=x_col,
                                                                    y_col=y_col,
                                                                    directory=data_dir,
                                                                    target_size=(160, 160),
                                                                    color_mode=color_mode,
                                                                    batch_size=batch_size,
                                                                    subset='validation',
                                                                    class_mode='categorical', classes=classes,
                                                                    shuffle=False)

        return validation_iterator

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.iterator), len(self.validation_iterator)

    def __getitem__(self, i):
        if self.valid:
            X, y = self.validation_iterator[i]

        else:
            X, y = self.iterator[i]

        return X, y

    def on_epoch_end(self):
        self.iterator.on_epoch_end()

class ViTCACD2000DataLoader(CACD2000Dataset):

    def __init__(self, logger, metadata_file,
               list_IDs, color_mode='grayscale', augmentation_generator=None, data_dir=None, classes=[], batch_size=64, dim=(72*72), n_channels=1, n_classes=2, shuffle=True, valid=True):

        super(ViTCACD2000DataLoader, self).__init__(logger, metadata_file,
               list_IDs, batch_size=batch_size, dim=dim, n_channels=n_channels, n_classes=n_classes, shuffle=shuffle, valid=valid)

        self.classes = classes

        if self.augmentation_generator:
            self.augmentation_generator = augmentation_generator
            self.iterator = self.get_iterator(color_mode, batch_size, data_dir, augmentation_generator, classes, x_col='filename',
                                          y_col='identity')
            self.validation_iterator = self.get_validation_iterator(color_mode, batch_size, data_dir, augmentation_generator, classes, x_col='filename',
                                          y_col='identity')

    def get_iterator(self, color_mode, batch_size, data_dir, augmentation_generator, classes=None, x_col='filename', y_col='identity'):
        train_iterator = augmentation_generator.flow_from_dataframe(self.metadata,
                                                                    x_col=x_col,
                                                                    y_col=y_col,
                                                                    directory=data_dir,
                                                                    target_size=(160, 160),
                                                                    color_mode=color_mode,
                                                                    batch_size=batch_size,
                                                                    subset='training',
                                                                    class_mode='categorical', classes=classes,
                                                                    shuffle=False)

        return train_iterator

    def get_validation_iterator(self, color_mode, batch_size, data_dir, augmentation_generator, classes=None, x_col='filename', y_col='identity'):
        validation_iterator = augmentation_generator.flow_from_dataframe(self.metadata,
                                                                    x_col=x_col,
                                                                    y_col=y_col,
                                                                    directory=data_dir,
                                                                    target_size=(160, 160),
                                                                    color_mode=color_mode,
                                                                    batch_size=batch_size,
                                                                    subset='validation',
                                                                    class_mode='categorical', classes=classes,
                                                                    shuffle=False)

        return validation_iterator

    def set_metadata(self, metadata):
        self.metadata = metadata
        self.iterator = self.get_iterator(self.color_mode, self.batch_size, self.data_dir, self.augmentation_generator, self.classes)
        self.validation_iterator = self.get_validation_iterator(self.color_mode, self.batch_size, self.data_dir,
                                                                self.augmentation_generator, self.classes)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.iterator), len(self.validation_iterator)

    def __getitem__(self, i):
        if self.valid:
            X, y = self.validation_iterator[i]

        else:
            X, y = self.iterator[i]

        return X, y

    def on_epoch_end(self):
        self.iterator.on_epoch_end()


class ViTFGNETDataLoader(FGNETDataset):

    def __init__(self, logger, metadata_file,
               list_IDs, color_mode='grayscale', augmentation_generator=None, data_dir=None, classes=[], batch_size=64, dim=(72*72), n_channels=1, n_classes=2, shuffle=True, valid=True):

        super(ViTFGNETDataLoader, self).__init__(logger, metadata_file,
               list_IDs, batch_size=batch_size, dim=dim, n_channels=n_channels, n_classes=n_classes, shuffle=shuffle, valid=valid)

        self.classes = classes

        if self.augmentation_generator:
            self.augmentation_generator = augmentation_generator
            self.iterator = self.get_iterator(color_mode, batch_size, data_dir, augmentation_generator, classes, x_col='filename',
                                          y_col='fileno')
            self.validation_iterator = self.get_validation_iterator(color_mode, batch_size, data_dir, augmentation_generator, classes, x_col='filename',
                                          y_col='fileno')

    def get_iterator(self, color_mode, batch_size, data_dir, augmentation_generator, classes=None, x_col='filename', y_col='fileno'):
        train_iterator = augmentation_generator.flow_from_dataframe(self.metadata,
                                                                    x_col=x_col,
                                                                    y_col=y_col,
                                                                    directory=data_dir,
                                                                    target_size=(160, 160),
                                                                    color_mode=color_mode,
                                                                    batch_size=batch_size,
                                                                    subset='training',
                                                                    class_mode='categorical', classes=classes,
                                                                    shuffle=False)

        return train_iterator

    def get_validation_iterator(self, color_mode, batch_size, data_dir, augmentation_generator, classes=None, x_col='filename', y_col='fileno'):
        validation_iterator = augmentation_generator.flow_from_dataframe(self.metadata,
                                                                    x_col=x_col,
                                                                    y_col=y_col,
                                                                    directory=data_dir,
                                                                    target_size=(160, 160),
                                                                    color_mode=color_mode,
                                                                    batch_size=batch_size,
                                                                    subset='validation',
                                                                    class_mode='categorical', classes=classes,
                                                                    shuffle=False)

        return validation_iterator

    def set_metadata(self, metadata):
        self.metadata = metadata
        self.iterator = self.get_iterator(self.color_mode, self.batch_size, self.data_dir, self.augmentation_generator, self.classes)
        self.validation_iterator = self.get_validation_iterator(self.color_mode, self.batch_size, self.data_dir,
                                                                self.augmentation_generator, self.classes)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.iterator), len(self.validation_iterator)

    def __getitem__(self, i):
        if self.valid:
            X, y = self.validation_iterator[i]

        else:
            X, y = self.iterator[i]

        return X, y

    def on_epoch_end(self):
        self.iterator.on_epoch_end()