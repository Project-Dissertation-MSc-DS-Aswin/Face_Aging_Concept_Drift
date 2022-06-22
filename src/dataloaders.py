import tensorflow as tf
import keras
import scipy.io
import numpy as np
import pandas as pd
import keras

"""
Reference:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=64, dim=(72,72), n_channels=1,
                 n_classes=2, shuffle=False, valid=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.valid = valid
        
    def get_iterator(self, color_mode, batch_size, data_dir, augmentation_generator, x_col='filename', y_col='name'):
        train_iterator = augmentation_generator.flow_from_dataframe(self.metadata, 
                                                                x_col=x_col, 
                                                                y_col=y_col,
                                                                directory=data_dir, 
                                                                target_size=self.dim,
                                                                color_mode=color_mode, 
                                                                batch_size=batch_size, 
                                                                class_mode='categorical', classes=None, shuffle=self.shuffle)
        
        return train_iterator
    
    def get_iterator_face_classificaton(self, color_mode, batch_size, data_dir, augmentation_generator, x_col='filename', y_cols=None):
        train_iterator = augmentation_generator.flow_from_dataframe(self.metadata, 
                                                                x_col=x_col, 
                                                                y_col=y_cols,
                                                                directory=data_dir, 
                                                                target_size=self.dim,
                                                                color_mode=color_mode, 
                                                                batch_size=batch_size, 
                                                                class_mode='multi_output', classes=None, shuffle=self.shuffle)
        
        return train_iterator
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X, y = self.iterator[index]

        return X, y
      
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
