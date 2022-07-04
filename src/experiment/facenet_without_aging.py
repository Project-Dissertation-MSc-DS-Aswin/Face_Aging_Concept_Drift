import pickle
import os
from preprocessing.facenet import l2_normalize, prewhiten
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from evaluation.distance import cosine, euclidean, face_distance
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
import pandas as pd
from collections import OrderedDict

def collect_data_facenet_keras(model_loader, train_iterator):
  res_images = []
  # Get input and output tensors
  for i in tqdm(range(len(train_iterator))):
    X, y = train_iterator[i]
    res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))

  return res_images

def collect_data_face_recognition_baseline(model_loader, train_iterator):
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
    
  def collect_data(self, data_collection_pkl, iterator=None, model=None):
    if os.path.isfile(data_collection_pkl):
      embeddings = pickle.load(data_collection_pkl)
    elif model == 'FaceNetKeras':
      embeddings = collect_data_facenet_keras(self.model_loader, self.dataset.iterator if iterator is None else iterator)
    elif model == 'FaceRecognitionBaselineKeras':
      embeddings = collect_data_face_recognition_baseline(self.model_loader, self.dataset.iterator if iterator is None else iterator)
      
    return tf.concat(embeddings, axis=0)
  
  def calculate_face_distance(self, embeddings):
    dist = euclidean_distances(embeddings)
    similarity = cosine_similarity(embeddings)
    
    return dist, similarity

  @property
  def get_list_IDs(self):
    return self.dataset.list_IDs