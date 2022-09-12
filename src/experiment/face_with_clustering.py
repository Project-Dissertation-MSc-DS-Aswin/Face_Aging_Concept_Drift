import pickle
import os
from preprocessing.facenet import l2_normalize, prewhiten
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from experiment.face_with_classifier import FaceNetWithClassifierExperiment
from experiment.context import base_estimators_voting_classifier_face_recognition
from evaluation.distance import cosine, euclidean, face_distance
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from preprocessing.facenet import l2_normalize, prewhiten
import numpy as np
import pandas as pd
from collections import OrderedDict
from copy import copy
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from keras.models import load_model, Model
from tqdm import tqdm
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, normalized_mutual_info_score

class FaceNetWithClusteringExperiment(FaceNetWithClassifierExperiment):
  
  def euclidean_distances(self, embeddings):
    """
    Get the euclidean distances
    @param embeddings:
    @return:
    """
    return euclidean_distances(embeddings)
  
  def cluster_embeddings(self, euclidean_embeddings_train, min_samples, eps):
    """
    Get the cluster embeddings
    @param euclidean_embeddings_train:
    @param min_samples:
    @param eps:
    @return:
    """
    self.db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    self.db.fit(euclidean_embeddings_train)
    
    return self.db

  def performance_analysis():
    """
    Conduct performance analysis
    @return:
    """
    model = load_model("/content/facenet_keras.h5")

    batch_size = 128

    def run_algorithm(X):
      """
      Runs the HyperParameter Tuning for Effective Clustering
      @param X:
      @return: tuple()
      """

      idx = 0
      homogeneity_train_dict = {}
      homogeneity_dict = {}
      completeness_train_dict = {}
      completeness_dict = {}
      v_measure_train_dict = {}
      v_measure_dict = {}
      ari_train_dict = {}
      ari_dict = {}
      ami_train_dict = {}
      ami_dict = {}

      for eps in tqdm(range(1, 12, 3)):
        for min_samples in range(1, 20, 5):
          db = DBSCAN(eps=eps, min_samples=min_samples)
          images = []
          for x in X:
            images.append(cv2.resize(x.reshape(50,37), (160,160)))
          images = np.stack(images)
          images = np.concatenate([np.expand_dims(images, 3)]*3, axis=2)
          images = l2_normalize(prewhiten(images.reshape(-1,160,160,3)))
          emb = []
          for ii in tqdm(range(0, len(images), batch_size)):
            emb.append(model.predict(images[ii:ii+batch_size]))
          emb = np.vstack(emb)
          db.fit(euclidean_distances(emb))
          labels = db.labels_

          homogeneity_dict[(eps, min_samples)] = homogeneity_score(y, labels)
          completeness_dict[(eps, min_samples)] = completeness_score(y, labels)
          v_measure_dict[(eps, min_samples)] = v_measure_score(y, labels)
          ari_dict[(eps, min_samples)] = adjusted_rand_score(y, labels)
          ami_dict[(eps, min_samples)] = normalized_mutual_info_score(y, labels)

      return homogeneity_dict, completeness_dict, v_measure_dict, ari_dict, ami_dict

