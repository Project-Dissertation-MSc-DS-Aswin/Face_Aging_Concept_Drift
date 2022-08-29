import pickle
import os
from preprocessing.facenet import l2_normalize, prewhiten
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from evaluation.distance import cosine, euclidean, face_distance
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import cv2
from collections import OrderedDict
from sklearn.decomposition import PCA

def collect_data_facenet_keras(model_loader, train_iterator, detectorExperiment):
  """
  Collect FaceNet Keras data using face detection model
  @param model_loader:
  @param train_iterator:
  @param detectorExperiment:
  @return:
  """
  res_images = []
  # Get input and output tensors
  for i in tqdm(range(len(train_iterator))):
    X, y = train_iterator[i]
    img = []
    for jj, x in enumerate(X):
      face = detectorExperiment.face_detect(x)
      if face is None:
        continue
      det = face[0][0:4].astype(np.int32)
      if x[det[0]:det[2], det[1]:det[3]].size == 0:
        continue
      img.append(cv2.resize(x[det[0]:det[2], det[1]:det[3]], (model_loader.input_shape[1], model_loader.input_shape[2])))
    images_bw = np.stack(img)
    res_images.append(model_loader.infer(l2_normalize(prewhiten(images_bw))))

  return res_images

def collect_data_face_recognition_baseline(model_loader, train_iterator, detectorExperiment):
  """
  Collect Face Recognition using Baseline CVAE by face detection
  @param model_loader:
  @param train_iterator:
  @param detectorExperiment:
  @return:
  """
  res_images = []
  y_classes = []
  # Get input and output tensors
  print(len(train_iterator))
  for i in tqdm(range(len(train_iterator))):
    X, (y, ) = train_iterator[i]
    y_classes += y.tolist()
  classes_counter = 0
  for i in tqdm(range(len(train_iterator)-1)):
    X, (y, ) = train_iterator[i]
    img = []
    for jj, x in enumerate(X):
      face = detectorExperiment.face_detect(x)
      if face is None:
        continue
      det = face[0][0:4].astype(np.int32)
      if x[det[0]:det[2], det[1]:det[3]].size == 0:
        continue
      img.append(cv2.resize(x[det[0]:det[2], det[1]:det[3]], (model_loader.input_shape[1], model_loader.input_shape[2])))
    images_bw = np.stack(img)
    # res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    classes = y
    unq_classes = np.unique(classes)
    y_valid = np.zeros((len(y), 435))
    for c in unq_classes:
      y_valid[classes==c, classes_counter]
      classes_counter += 1
    res_images.append(model_loader.infer([X/255., y_valid]))

  return res_images

class FaceNetWithoutAgingExperiment:
  
  def __init__(self, dataset, logger=None, model_loader=None):
    """
    __init__ function
    @param dataset:
    @param logger:
    @param model_loader:
    """
    self.dataset = dataset
    self.logger = logger
    self.model_loader = model_loader
    self.batchno = 0

  def set_dataset(self, dataset):
    """
    Set the dataset
    @param dataset:
    @return:
    """
    self.dataset = dataset

  def set_logger(self, logger):
    """
    Set the logger
    @param logger:
    @return:
    """
    self.logger = logger

  def set_model_loader(self, model_loader):
    """
    Set the model loader
    @param model_loader:
    @return:
    """
    self.model_loader = model_loader
    
  def collect_data(self, data_collection_pkl, iterator=None, model=None, detectorExperiment=None):
    """
    Collect data by FaceNetKeras or FaceRecognitionByBaselineCVAE
    @param data_collection_pkl:
    @param iterator:
    @param model:
    @param detectorExperiment:
    @return:
    """
    if os.path.isfile(data_collection_pkl):
      embeddings = pickle.load(data_collection_pkl)
    elif model == 'FaceNetKeras':
      embeddings = collect_data_facenet_keras(self.model_loader, self.dataset.iterator if iterator is None else iterator, detectorExperiment)
    elif model == 'FaceRecognitionBaselineKeras':
      embeddings = collect_data_face_recognition_baseline(self.model_loader, self.dataset.iterator if iterator is None else iterator, detectorExperiment)
      
    return tf.concat(embeddings, axis=0)
  
  def calculate_face_distance(self, embeddings):
    """
    Calculate Face distance
    @param embeddings:
    @return:
    """
    dist = euclidean_distances(embeddings)
    similarity = cosine_similarity(embeddings)
    
    return dist, similarity
  
  def calculate_face_error(self, embeddings):
    """
    Calculate Face Error usign embeddings
    @param embeddings:
    @return:
    """
    embeddings = np.round(embeddings, 1)
    return (DistanceMetric.get_metric('dice').pairwise(embeddings, embeddings))**-1, (DistanceMetric.get_metric('jaccard').pairwise(embeddings, embeddings))**-1
  
  def calculate_face_mahalanonis_distance(self, embeddings):
    """
    Calculate face mahalanobis distances of embeddings
    @param embeddings:
    @return:
    """
    dist = embeddings.dot(embeddings.T)
    pca = PCA(n_components=dist.shape[0])
    pca.fit(dist)
    dist *= np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
    similarity = cosine_similarity(embeddings)
    
    return dist, similarity

  @property
  def get_list_IDs(self):
    """
    get the list_IDs of the dataset
    @return:
    """
    return self.dataset.list_IDs