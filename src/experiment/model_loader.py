import sys
import os
from matplotlib import image
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
import cv2
from experiment.yunet import YuNet
from pipeline.context import Constants
from tensorflow.keras import backend as K
constants = Constants()

sys.path.append(constants.LATS_REPO)

def preprocess_data_facenet_without_aging(X_train):
  """
  Preprocess Data with Facenet without Aging
  @param X_train:
  @return:
  """
  X_train = X_train.astype('float32')

  return X_train

def get_augmented_datasets():
  """
  Get Augmented Datasets
  @return: ImageDataGenerator
  """
  # Create image augmentation
  augmentation_generator = ImageDataGenerator(horizontal_flip=False, # Randomly flip images
                                    vertical_flip=False, # Randomly flip images
                                    rotation_range = None, 
                                    validation_split=0.0,
                                    brightness_range=None,
                                    preprocessing_function=preprocess_data_facenet_without_aging) #Randomly rotate

  return augmentation_generator

class KerasModelLoader:
  """
  Keras Model Loader
  """
  dimensions = 128
  
  def __init__(self, logger, model_path, input_shape=None, image_data_format='channels_last'):
    """
    __init__ function
    @param logger:
    @param model_path:
    @param input_shape:
    @param image_data_format:
    """
    self.logger = logger
    self.model_path = model_path
    self.type = 'keras'
    if image_data_format == 'channels_last':
      (b, input_w, input_h, n_channels) = input_shape
    elif image_data_format == 'channels_first':
      (b, n_channels, input_w, input_h) = input_shape
    self.input_w = input_w
    self.input_h = input_h
    self.input_shape = input_shape

  def load_model(self):
    """
    Load the Keras Model from model_path
    @return:
    """
    print(os.path.isfile(self.model_path))
    self.model = load_model(self.model_path, compile=False)
    self.logger.log({
      "keras_model_summary": self.model.summary()
    })
    
  def infer(self, data):
    """
    data: Image
    @param data:
    @return:
    """
    return self.model.predict(data)
  
  def resize(self, data):
    """
    Resize the Image by width and height
    @param data:
    @return:
    """
    return cv2.resize(data, (self.input_w, self.input_h))
  
class FaceNetKerasModelLoader(KerasModelLoader):
  pass

from keras import regularizers

class ArcFace(keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=regularizers.l2(0.0001), **kwargs):
        """
        __init__ function
        @param n_classes:
        @param s:
        @param m:
        @param regularizer:
        @param kwargs:
        """
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        """
        Build the model and add weights
        @param input_shape:
        @return:
        """
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        """
        Call - callback function
        @param inputs:
        @return:
        """
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s

        return logits

    def compute_output_shape(self, input_shape):
        """
        Compute Output shape
        @param input_shape:
        @return:
        """
        return (None, self.n_classes)

    def get_config(self):
        """
        Get config
        @return:
        """
        config = super().get_config()
        config.update({
            "s": self.s,
            "m": self.m,
            "n_classes": self.n_classes
        })
        return config

class FaceRecognitionBaselineKerasModelLoader(KerasModelLoader):
  dimensions = 717
  
  def load_model(self):
    """
    Load the Keras Model from model_path
    @return:
    """
    print(os.path.isfile(self.model_path))
    self.model = load_model(self.model_path, compile=False, custom_objects={"ArcFace": ArcFace})
    self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-6].output)
    self.logger.log({
      "keras_model_summary": self.model.summary()
    })


class MNIST2Epochs(KerasModelLoader):
  pass

class MNIST5Epochs(KerasModelLoader):
  pass

class MNIST10Epochs(KerasModelLoader):
  pass

class YuNetModelLoader:
  
  def __init__(self, logger, model_path, conf_threshold, nms_threshold, backend, target, top_k, input_shape=[1, 96, 96, 1]):
    """
    __init__ function of the YuNet Model
    @param logger:
    @param model_path:
    @param conf_threshold:
    @param nms_threshold:
    @param backend:
    @param target:
    @param top_k:
    @param input_shape:
    """
    self.input_shape = input_shape
    (b, input_h, input_w, n_channels) = input_shape
    self.input_w = input_w
    self.input_h = input_h
    self.logger = logger
    self.model_path = model_path
    self.conf_threshold = conf_threshold
    self.nms_threshold = nms_threshold
    self.backend = backend
    self.target = target
    self.top_k = top_k
    self.load_model()
  
  def load_model(self):
    """
    Load the model into memory
    @return:
    """
    self.detector = YuNet(modelPath=self.model_path,
                  inputSize=(self.input_h, self.input_w),
                  confThreshold=self.conf_threshold,
                  nmsThreshold=self.nms_threshold,
                  topK=self.top_k,
                  backendId=self.backend,
                  targetId=self.target)
    
  def resize(self, data):
    """
    Resize the detector input size
    @param data:
    @return:
    """
    return self.detector.setInputSize((self.input_h, self.input_w))
  
  def infer(self, data):
    """
    Inference of the image using YuNet model
    @param data:
    @return:
    """
    return self.detector.infer(data)
  
  