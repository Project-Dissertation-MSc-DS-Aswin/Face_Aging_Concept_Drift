import pickle
import os
from preprocessing.facenet import l2_normalize, prewhiten
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from evaluation.distance import cosine, euclidean, face_distance
import tensorflow as tf
from tqdm import tqdm

def collect_data(model_loader, train_iterator):
  res_images = []
  # Get input and output tensors
  for ii, (X, y) in tqdm(enumerate(train_iterator)):
    res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    
  return res_images

class FaceNetWithoutAgingExperiment:
  
  def __init__(self, dataset, logger=None, model_loader=None):
    self.dataset = dataset
    self.logger = logger
    self.model_loader = model_loader
    self.batchno = 0

  def set_dataset(self, dataset):
    self.dataset = dataset

  def set_logger(self, logger):
    self.logger = logger

  def set_model_loader(self, model_loader):
    self.model_loader = model_loader
    
  def collect_data(self, data_collection_pkl, data_dir, batch_size):
    if os.path.isfile(data_collection_pkl):
      embeddings = pickle.load(data_collection_pkl)
    else:
      embeddings = collect_data(self.model_loader, self.dataset.iterator)
      
    return embeddings
  
  def calculate_face_distance(self, embeddings):
    m = embeddings.shape[0]
    n = m**2
    x = tf.Variable(tf.constant(list(range(n)), dtype=tf.int32))

    result_distances = []
    for ii, _x in tqdm(enumerate(x)):
      result_distances.append(face_distance(ii, _x, embeddings))
      
    result_distances = tf.concat(result_distances, axis=0)
    
    return result_distances

  @property
  def get_list_IDs(self):
    return self.dataset.list_IDs