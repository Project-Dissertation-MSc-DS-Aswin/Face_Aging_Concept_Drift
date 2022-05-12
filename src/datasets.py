from dataloaders import DataGenerator
import scipy.io
import numpy as np
import pandas as pd
import constants

"""
CACD2000 DataGenerator
"""
class CACD2000Dataset(DataGenerator):
  
  def __init__(self, iterator, logger, metadata_file, 
               list_IDs, batch_size=64, dim=(72*72), n_channels=1, n_classes=2, shuffle=True, valid=False):
    self.logger = logger
    self.iterator = iterator
    self.metadata_file = metadata_file
    
    self.logger.log(self.opt.__dict__())
    
    super(CACD2000Dataset).__init__(self, list_IDs, batch_size=64, dim=(72*72), n_channels=1,
                 n_classes=2, shuffle=True, valid=False)
    
    self.metadata = self.load_dataset(metadata_file)
    self.mapping = self.load_identity_mapping(self.metadata)
  
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
  
  """
  Identities
  """
  def load_identity_mapping(self, metadata):
    identity = metadata['identity']
    self.logger.log({constants.INFO: "Identity mapping successfully loaded"})
    return np.unique(identity)
  
  def __len__(self):
    return len(self.iterator)
  
  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim))
    y = np.empty((self.batch_size, self.num_classes))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      x1 = self.iterator[ID]
      # Store sample
      identity = self.metadata['identity'].loc[ID]
      y[i] = (self.mapping == identity).astype(int)

    return X, y
  
  def next(self, list_IDs_temp):
    return self.__data_generation(list_IDs_temp)

  def __getitem__(self, i):
    x1 = self.iterator[i]
    # Store sample
    y = self.metadata['identity'].values[i]
    
    return x1, y
    

"""
AgeDBDataset DataGenerator
Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
class AgeDBDataset(DataGenerator):
  
  def __init__(self, iterator, logger, metadata_file, 
               list_IDs, batch_size=64, dim=(72*72), n_channels=1, n_classes=2, shuffle=True, valid=False):
    self.logger = logger
    self.iterator = iterator
    self.metadata_file = metadata_file
    
    self.logger.log(self.opt.__dict__())
    
    super(AgeDBDataset).__init__(self, list_IDs, batch_size=64, dim=(72*72), n_channels=1,
                 n_classes=2, shuffle=True, valid=False)
    
    self.metadata = self.load_dataset(metadata_file)
    self.mapping = self.load_name_mapping(self.metadata)

  """
  Loads the metadata of the dataset
  returns: pd.DataFrame
  """
  def load_dataset(self, metadata_file):
    mat = scipy.io.loadmat(metadata_file)
    self.logger.log({constants.INFO: "Dataset/Metadata successfully loaded"})
    fileno, filename, name, age, gender = mat['fileno'], mat['filename'], mat['name'], mat['age'], mat['gender']
    metadata_array = np.concatenate([fileno, filename, name, age, gender], axis=1)
    metadata = pd.DataFrame(np.array(metadata_array), columns=['fileno', 'filename', 'name', 'age', 'gender'])
    metadata['age'] = metadata['age'].astype(np.int)
    metadata['fileno'] = metadata['fileno'].astype(np.int)
    return metadata
  
  """
  Identities
  """
  def load_name_mapping(self, metadata):
    names = metadata['name']
    self.logger.log({constants.INFO: "Name mapping successfully loaded"})
    return np.unique(names)
  
  def __len__(self):
    return len(self.iterator)
  
  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim))
    y = np.empty((self.batch_size, self.num_classes))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      x1 = self.iterator[ID]
      # Store sample
      name = self.metadata['name'].loc[ID]
      y[i] = (self.mapping == name).astype(int)

    return X, y
  
  def next(self, list_IDs_temp):
    return self.__data_generation(list_IDs_temp)

  def __getitem__(self, i):
    x1 = self.iterator[i]
    # Store sample
    y = self.metadata['name'].values[i]
    
    return x1, y

"""
FGNETDataset DataGenerator
"""
class FGNETDataset(DataGenerator):
  
  def __init__(self, iterator, logger, metadata_file, 
               list_IDs, batch_size=64, dim=(72*72), n_channels=1, n_classes=2, shuffle=True, valid=False):
    self.logger = logger
    self.iterator = iterator
    self.metadata_file = metadata_file
    
    self.logger.log(self.opt.__dict__())
    
    super(FGNETDataset).__init__(self, list_IDs, batch_size=64, dim=(72*72), n_channels=1,
                 n_classes=2, shuffle=True, valid=False)
    
    self.metadata = self.load_dataset(metadata_file)
    self.mapping = self.load_name_mapping(self.metadata)
    
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
  
  """
  Identities
  """
  def load_identity_mapping(self, metadata):
    fileno = metadata['fileno']
    self.logger.log({constants.INFO: "Identity mapping successfully loaded"})
    return np.unique(fileno)
  
  def __len__(self):
    return len(self.iterator)
  
  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim))
    y = np.empty((self.batch_size, self.num_classes))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      x1 = self.iterator[ID]
      # Store sample
      fileno = self.metadata['fileno'].loc[ID]
      y[i] = (self.mapping == fileno).astype(int)

    return X, y
  
  def next(self, list_IDs_temp):
    return self.__data_generation(list_IDs_temp)

  def __getitem__(self, i):
    x1 = self.iterator[i]
    # Store sample
    y = self.metadata['fileno'].values[i]
    
    return x1, y