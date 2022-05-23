import sys
import constants
sys.path.append(constants.LATS_REPO)
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import skimage.transform

def preprocess_data_facenet_without_aging(X_train):
  X_train = X_train.astype('float32')

  return X_train

def get_augmented_datasets():
  # Create image augmentation
  augmentation_generator = ImageDataGenerator(horizontal_flip=False, # Randomly flip images
                                    vertical_flip=False, # Randomly flip images
                                    rotation_range = None, 
                                    validation_split=0.0,
                                    brightness_range=None,
                                    preprocessing_function=preprocess_data_facenet_without_aging) #Randomly rotate

  return augmentation_generator

class KerasModelLoader:
  
  def __init__(self, logger, model_path, input_shape=None):
    self.logger = logger
    self.model_path = model_path
    self.type = 'keras'
    (b, input_w, input_h, n_channels) = input_shape
    self.input_w = input_w
    self.input_h = input_h
    self.input_shape = input_shape
  
  """
  Load the Keras Model from model_path
  """
  def load_model(self):
    self.model = load_model(self.model_path)
    self.logger.log({
      "keras_model_summary": self.model.summary()
    })
    
  """
  data: Image
  """
  def infer(self, data):
    return self.model.predict(data)
  
  def resize(self, data):
    return skimage.transform.resize(data, (self.input_w, self.input_h))