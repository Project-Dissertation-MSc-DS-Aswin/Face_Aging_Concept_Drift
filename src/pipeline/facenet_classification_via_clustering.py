import sys
from sklearn.ensemble import VotingClassifier
from context import Constants, Args
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA
import whylogs
import mlflow
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.facenet_with_classifier import FaceNetWithClassifierPredictor
from experiment.facenet_with_clustering import FaceNetWithClusteringExperiment
from experiment.model_loader import FaceNetKerasModelLoader, FaceRecognitionBaselineKerasModelLoader
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from sklearn.utils.fixes import loguniform
from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.facenet import l2_normalize, prewhiten
from tqdm import tqdm
from copy import copy
import pickle
import logging
import sys
import cv2
import re
import tensorflow as tf
import imageio

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

args.dataset = os.environ.get('dataset', 'agedb')
args.model = os.environ.get('model', 'FaceNetKeras')
args.model_path = os.environ.get('model_path', 'facenet_keras.h5')
args.data_dir = os.environ.get('data_dir', constants.AGEDB_DATADIR)
args.batch_size = os.environ.get('batch_size', 128)
args.preprocess_prewhiten = os.environ.get('preprocess_prewhiten', 1)
args.data_collection_pkl = os.environ.get('data_collection_pkl', constants.AGEDB_FACENET_INFERENCES)
args.metadata = os.environ.get('metadata', constants.AGEDB_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_with_classifier')
args.no_of_samples = os.environ.get('no_of_samples', 16488)
args.colormode = os.environ.get('colormode', 'rgb')
args.log_images = os.environ.get('log_images', 's3')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.classifier_test_younger = os.environ.get('classifier_test_younger', constants.AGEDB_FACE_CLASSIFIER_TEST_YOUNGER)
args.classifier_train_younger = os.environ.get('classifier_train_younger', constants.AGEDB_FACE_CLASSIFIER_TRAIN_YOUNGER)
args.collect_for = os.environ.get('collect_for', 'age_drifting')
args.drift_evaluate_metrics_test_younger = os.environ.get('drift_evaluate_metrics_test_younger', constants.AGEDB_DRIFT_EVALUATE_METRICS_TEST_YOUNGER)
args.drift_evaluate_metrics_train_younger = os.environ.get('drift_evaluate_metrics_train_younger', constants.AGEDB_DRIFT_EVALUATE_METRICS_TRAIN_YOUNGER)
args.min_samples = os.environ.get('min_samples', 1)
args.eps = os.environ.get('eps', 1.0)
args.experiment_id = os.environ.get("experiment_id", 2)
args.log_file_younger = os.environ.get("log_file_younger", "test_data_predictions_younger.csv")
args.log_file_older = os.environ.get("log_file_older", "test_data_predictions_older.csv")
args.input_shape = os.environ.get('input_shape', (-1,160,160,3))

parameters = list(
    map(lambda s: re.sub('$', '"', s),
        map(
            lambda s: s.replace('=', '="'),
            filter(
                lambda s: s.find('=') > -1 and bool(re.match(r'[A-Za-z0-9_]*=[.\/A-Za-z0-9]*', s)),
                sys.argv
            )
    )))

for parameter in parameters:
    logging.warning('Parameter: ' + parameter)
    exec("args." + parameter)
    
args.batch_size = int(args.batch_size)
args.preprocess_prewhiten = int(args.preprocess_prewhiten)
args.no_of_samples = int(args.no_of_samples)
args.experiment_id = int(args.experiment_id)
args.eps = float(args.eps)
args.min_samples = int(args.min_samples)
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])
    print(args.input_shape)
    
def load_dataset(args, whylogs, no_of_samples, colormode, input_shape=(-1,160,160,3)):
  
    dataset = None
    augmentation_generator = None
    if args.dataset == "agedb":
        augmentation_generator = get_augmented_datasets()
        dataset = AgeDBDataset(whylogs, args.metadata, list_IDs=list(range(no_of_samples)),
                               color_mode=colormode, augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                               dim=(input_shape[1],input_shape[2]), 
                               batch_size=args.batch_size)
    elif args.dataset == "cacd":
        augmentation_generator = get_augmented_datasets()
        dataset = CACD2000Dataset(whylogs, args.metadata, list_IDs=list(range(no_of_samples)),
                                  color_mode=colormode, augmentation_generator=augmentation_generator,
                                  data_dir=args.data_dir, dim=(input_shape[1],input_shape[2]), batch_size=args.batch_size)
    elif args.dataset == "fgnet":
        augmentation_generator = get_augmented_datasets()
        dataset = FGNETDataset(whylogs, args.metadata, list_IDs=None,
                               color_mode=colormode, augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                               dim=(input_shape[1],input_shape[2]), 
                               batch_size=args.batch_size)

    return dataset, augmentation_generator
  
def get_reduced_metadata(args, dataset, seed=1000):
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    np.random.seed(seed)
    names = dataset.metadata.groupby('name').count()
    names = names[names['age'] > 40]
    names = names.index.get_level_values(0)
    idx = [dataset.metadata['name'] == name for name in names]
    result_idx = [False]*len(dataset.metadata)
    for i in idx:
      result_idx = np.logical_or(result_idx, i)
      
    return dataset.metadata.loc[result_idx].reset_index()
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()
  
def collect_data_face_recognition_keras(model_loader, train_iterator):
  res_images = []
  y_classes = []
  files = []
  ages = []
  labels = []
  images = []
  # Get input and output tensors
  classes_counter = 0
  for i in tqdm(range(len(train_iterator)-1)):
    X, (y_age, y_filename, y_label) = train_iterator[i]
    # res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    classes = y_label
    unq_classes = np.unique(classes)
    y_valid = np.zeros((len(y_label), 435))
    for c in unq_classes:
      y_valid[classes==c, classes_counter] = 1
      classes_counter += 1
    images.append(X)
    res_images.append(model_loader.infer([X/255., y_valid]))
    labels += y_label.tolist()
    files += y_filename.tolist()
    ages += y_age.tolist()

  return images, res_images, files, ages, labels

def collect_data_facenet_keras(model_loader, train_iterator):
  res_images = []
  files = []
  ages = []
  labels = []
  images = []
  # Get input and output tensors
  for i in tqdm(range(len(train_iterator))):
    X, (y_age, y_filename, y_label) = train_iterator[i]
    images.append(X)
    res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    labels += y_label.tolist()
    files += y_filename.tolist()
    ages += y_age.tolist()
    
  return images, res_images, files, ages, labels
  
if __name__ == "__main__":
  
    mlflow.set_tracking_uri(args.tracking_uri)
    
    if args.model == 'FaceNetKeras':
      model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    elif args.model == 'FaceRecognitionBaselineKeras':
      model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    
    model_loader.load_model()

    dataset, augmentation_generator = load_dataset(args, whylogs, args.no_of_samples, 'rgb', input_shape=args.input_shape)
    
    dataset.set_metadata(
      get_reduced_metadata(args, dataset)
    )
    
    experiment = FaceNetWithClusteringExperiment(dataset, whylogs, model_loader)
    
    if args.dataset == 'agedb':
      face_classification_iterator = dataset.get_iterator_face_classificaton(
        args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'name']
      )
    elif args.dataset == 'cacd':
      face_classification_iterator = dataset.get_iterator_face_classificaton(
        args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'identity']
      )
    elif args.dataset == 'fgnet':
      face_classification_iterator = dataset.get_iterator_face_classificaton(
        args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'fileno']
      )
    
    if args.model == 'FaceNetKeras':
      images, embeddings, files, ages, labels = collect_data_facenet_keras(model_loader, face_classification_iterator)
    elif args.model == 'FaceRecognitionBaselineKeras':
      images, embeddings, files, ages, labels = collect_data_face_recognition_keras(model_loader, face_classification_iterator)
    
    embeddings = np.vstack(embeddings)
    
    param_grid = {
      "C": loguniform.rvs(0.1, 100, size=5),
      "gamma": loguniform.rvs(1e-4, 1e-1, size=5),
    }
    param_grid2 = {
      "criterion": ['entropy', 'gini'], 
      "min_samples_split": [2,5,10]
    }
    param_grid3 = {
      "learning_rate": [0.001, 0.01, 0.1], 
      "max_iter": [50, 100, 200],
      "max_depth": [3, 5, 10], 
      "min_samples_leaf": [1, 5, 10, 20]
    }
    
    algorithm = FaceNetWithClassifierPredictor(metadata=dataset.metadata, model_loader=model_loader)
    
    algorithm.make_train_test_split(embeddings, files, ages, labels)
    
    pickle.dump(algorithm.embeddings_train, open("embeddings_train.pkl", "wb"))
    pickle.dump(images, open("images.pkl", "wb"))
    
    algorithm.embeddings_train = pickle.load(open("embeddings_train.pkl", "rb"))
    images = pickle.load(open("images.pkl", "rb"))
    
    euclidean_embeddings = experiment.euclidean_distances(algorithm.embeddings_train)
    db = experiment.cluster_embeddings(euclidean_embeddings, args.min_samples, args.eps)
    
    print("Homgeneity Score: ", homogeneity_score(algorithm.labels_train, db.labels_))
    print("Completeness Score: ", completeness_score(algorithm.labels_train, db.labels_))
    
    print("Unique Labels: ")
    print(np.unique(db.labels_))
    
    # algorithm.labels_train = db.labels_
    
    df = pd.DataFrame()
    df['cluster_labels'] = db.labels_
    df['files'] = algorithm.files_train
    df['labels'] = algorithm.labels_train
    
    from collections import Counter
    labels_dict = dict(Counter(db.labels_))
    labels_filtered = [label for label, count in labels_dict.items() if count > 1]
    
    # print(labels_filtered)
    
    labels = np.array([False]*len(df))
    for label in labels_filtered:
      labels = np.logical_or(labels, df['cluster_labels'] == label)
      
    # print(df.loc[labels, ['labels', 'cluster_labels']])
    
    # print(df.loc[labels, ['cluster_labels', 'labels', 'files']].groupby(by=['labels', 'cluster_labels'])['files'].count().to_csv("newfile.csv"))
    
    # df.loc[labels, 'files'].to_csv("files_clustered.csv")
    # df.loc[np.logical_not(labels), 'files'].to_csv("files_for_classification.csv")
    
    established_embeddings = algorithm.embeddings_train[labels]
    pickle.dump(established_embeddings, open("established_embeddings.pkl", "wb"))
    
    with mlflow.start_run(experiment_id=args.experiment_id):
      # print(np.unique(db.labels_), np.unique(db.labels_).shape)
      mds = MDS(n_components=2)
      euc_dimensions = mds.fit_transform(euclidean_embeddings)
      
      fig = plt.figure(figsize=(12,8))
      colors = sns.color_palette("hls", len(np.unique(db.labels_)))
      for ii,label in enumerate(np.unique(db.labels_)):
        plt.scatter(euc_dimensions[db.labels_ == label, 0], euc_dimensions[db.labels_ == label, 1], color=colors[ii])
        
      plt.xlabel("Dimension 1", fontsize=16)
      plt.ylabel("Dimension 2", fontsize=16)
      plt.title("Scatter plot showing MDS dimensionality reduced \nEuclidean Distances clustered by DBSCAN", fontsize=16)
      mlflow.log_figure(fig, "mds/scatter_plot.jpg")
      
      images = np.vstack(images)
      
      cluster_images = {}
      cluster_labels = {}
      for ij, i in enumerate(db.labels_):
        if i not in cluster_images:
          cluster_images[i] = []
          cluster_labels[i] = []
        cluster_images[i].append(images[ij])
        # cluster_labels[i].append(labels[ij])
      
      fig = plt.figure(figsize=(512,64))
      for jj, (i, new_images) in enumerate(cluster_images.items()):
        for ji, image in enumerate(new_images[:50]):
          plt.subplot(2, 50, jj*50 + ji + 1)
          plt.imshow(cv2.resize(image.reshape(160,160,3), (40,40)).astype(int), cmap='gray')
          # plt.title("Name: " + cluster_labels[i][ji], fontsize=4)
      
      mlflow.log_figure(fig, "cluster/cluster_images.jpg")
    
    