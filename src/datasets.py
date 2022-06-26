from dataloaders import DataGenerator
import scipy.io
import numpy as np
import pandas as pd
from pipeline.context import Constants

constants = Constants()

"""
CACD2000 DataGenerator
"""
class CACD2000Dataset(DataGenerator):
  
  def __init__(self, logger, metadata_file, 
               list_IDs, color_mode='grayscale', augmentation_generator=None, data_dir=None, batch_size=64, dim=(72,72), n_channels=1, n_classes=2, shuffle=False, valid=False):
    self.logger = logger
    self.metadata_file = metadata_file
    
    super(CACD2000Dataset, self).__init__(list_IDs, batch_size=batch_size, dim=dim, n_channels=1,
                 n_classes=2, shuffle=shuffle, valid=False)
    
    self.metadata = self.load_dataset(metadata_file)
    self.mapping = self.load_identity_mapping(self.metadata)
    
    self.color_mode = color_mode
    self.batch_size = batch_size
    self.data_dir = data_dir

    if self.augmentation_generator:
      self.augmentation_generator = augmentation_generator

  """
  Loads the metadata of the dataset
  returns: pd.DataFrame
  """
  def load_dataset(self, metadata_file):
    mat = scipy.io.loadmat(metadata_file)
    age, identity, year, feature_1, feature_2, feature_3, feature_4, name = mat['celebrityImageData'][0][0]
    metadata_CACD = pd.DataFrame(np.vstack([age.flatten(), identity.flatten(), year.flatten(), 
                                np.array(list(map(lambda x: x.tolist()[0][0].split("_")[1] + "_" + x.tolist()[0][0].split("_")[2], name))), 
                                np.array(list(map(lambda x: x.tolist()[0][0], name)))]).T, 
                      columns=['age', 'identity', 'year', 'name', 'filename'])
    metadata_CACD['age'] = metadata_CACD['age'].astype(int)
    metadata_CACD['identity'] = metadata_CACD['identity'].astype(int)
    metadata_CACD['year'] = metadata_CACD['year'].astype(int)
    return metadata_CACD
  
  def set_metadata(self, metadata):
    self.metadata = metadata
    self.iterator = self.get_iterator(self.color_mode, self.batch_size, self.data_dir, self.augmentation_generator, x_col='filename', y_col='identity')
  
  """
  Identities
  """
  def load_identity_mapping(self, metadata):
    identity = metadata['identity']
    self.logger.log({constants.INFO: "Identity mapping successfully loaded"})
    return np.unique(identity)
  
"""
AgeDBDataset DataGenerator
Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
class AgeDBDataset(DataGenerator):
  
  def __init__(self, logger, metadata_file, 
               list_IDs, color_mode='grayscale', augmentation_generator=None, data_dir=None, batch_size=64, dim=(72,72), n_channels=1, n_classes=2, shuffle=False, valid=False, 
               filter_func=None):
    self.logger = logger
    self.metadata_file = metadata_file
    
    super(AgeDBDataset, self).__init__(list_IDs, batch_size=batch_size, dim=dim, n_channels=1,
                 n_classes=2, shuffle=shuffle, valid=False)
    
    self.metadata = self.load_dataset(metadata_file, filter_func)
    self.mapping = self.load_name_mapping(self.metadata)
    
    self.color_mode = color_mode
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.augmentation_generator = augmentation_generator

    if self.augmentation_generator:
      self.augmentation_generator = augmentation_generator

  """
  Loads the metadata of the dataset
  returns: pd.DataFrame
  """
  def load_dataset(self, metadata_file, filter_func=None):
    mat = scipy.io.loadmat(metadata_file)
    self.logger.log({constants.INFO: "Dataset/Metadata successfully loaded"})
    fileno = list(map(lambda x: x[0], mat['fileno'][0]))
    filename = list(map(lambda x: x[0], mat['filename'][0]))
    name = list(map(lambda x: x[0], mat['name'][0]))
    age = list(map(lambda x: x[0], mat['age'][0]))
    gender = list(map(lambda x: x[0], mat['gender'][0]))
    metadata_agedb = pd.DataFrame(np.stack([fileno, filename, name, age, gender]).T, 
                                  columns=['fileno', 'filename', 'name', 'age', 'gender'])
    metadata_agedb['age'] = metadata_agedb['age'].astype(np.int)
    metadata_agedb['fileno'] = metadata_agedb['fileno'].astype(np.int)
    metadata_agedb['name'] = metadata_agedb['name'].astype(np.str)
    metadata_agedb['filename'] = metadata_agedb['filename'].astype(np.str)
    if filter_func is not None:
      metadata_agedb = metadata_agedb.apply(filter_func, axis=1)
    return metadata_agedb
  
  def set_metadata(self, metadata):
    self.metadata = metadata
    self.iterator = self.get_iterator(self.color_mode, self.batch_size, self.data_dir, self.augmentation_generator, x_col='filename', y_col='name')
  
  """
  Identities
  """
  def load_name_mapping(self, metadata):
    names = metadata['name']
    self.logger.log({constants.INFO: "Name mapping successfully loaded"})
    return np.unique(names)
  
"""
FGNETDataset DataGenerator
"""
class FGNETDataset(DataGenerator):
  
  def __init__(self, logger, metadata_file, 
               list_IDs, color_mode='grayscale', augmentation_generator=None, data_dir=None, batch_size=64, dim=(72,72), n_channels=1, n_classes=2, shuffle=False, valid=False):
    self.logger = logger
    self.metadata_file = metadata_file
    
    super(FGNETDataset, self).__init__(list_IDs, batch_size=batch_size, dim=dim, n_channels=1,
                 n_classes=2, shuffle=shuffle, valid=False)
    
    self.metadata = self.load_dataset(metadata_file)
    self.mapping = self.load_identity_mapping(self.metadata)
    
    self.color_mode = color_mode
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.augmentation_generator = augmentation_generator
    
    if self.augmentation_generator:
      self.augmentation_generator = augmentation_generator
      self.iterator = self.get_iterator(color_mode, batch_size, data_dir, augmentation_generator, x_col='filename', y_col='fileno')
    
  """
  Loads the metadata of the dataset
  returns: pd.DataFrame
  """
  def load_dataset(self, metadata_file):
    mat = scipy.io.loadmat(metadata_file)
    fileno, filename, age = mat['fileno'], mat['filename'], mat['age']
    metadata_fgnet = pd.DataFrame(
        np.array([
            list(map(lambda x: x[0], fileno[0])), 
            list(map(lambda x: x[0], filename[0])), 
            list(map(lambda x: x[0], age[0]))
        ]).T, 
        columns=['fileno', 'filename', 'age']
    )
    
    return metadata_fgnet
  
  def set_metadata(self, metadata):
    self.metadata = metadata
    self.iterator = self.get_iterator(self.color_mode, self.batch_size, self.data_dir, self.augmentation_generator, x_col='filename', y_col='fileno')
  
  """
  Identities
  """
  def load_identity_mapping(self, metadata):
    fileno = metadata['fileno']
    self.logger.log({constants.INFO: "Identity mapping successfully loaded"})
    return np.unique(fileno)
  