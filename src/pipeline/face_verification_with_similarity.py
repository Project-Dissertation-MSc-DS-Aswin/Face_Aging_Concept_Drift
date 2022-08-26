from context import Constants, Args
import tensorflow as tf
from dataloaders import DataGenerator
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.facenet_without_aging import FaceNetWithoutAgingExperiment
from experiment.face_classification_by_images import FaceClassificationByImages
from experiment.model_loader import FaceNetKerasModelLoader, FaceRecognitionBaselineKerasModelLoader
import whylogs
from sklearn.metrics.pairwise import euclidean_distances
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging, YuNetModelLoader
from sklearn.neural_network import MLPClassifier
from preprocessing.facenet import l2_normalize, prewhiten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy
import os
import sys
import re
import pickle
import argparse
from pathlib import Path
import logging
import mlflow
import cv2
from collections import Counter

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

"""
Arguments to the pipeline
"""
args.dataset = os.environ.get('dataset', 'agedb')
args.source_type = os.environ.get('source_type', 'ordered_metadata')
args.model_detect = os.environ.get('model_detect', 'YuNet_onnx')
args.model_path_detect = os.environ.get('model_path_detect', '../models/face_detection_yunet_2022mar.onnx')
args.model = os.environ.get('model', 'FaceNetKeras')
args.model_path = os.environ.get('model_path', '../models/facenet_keras.h5')
args.data_dir = os.environ.get('data_dir', constants.AGEDB_DATADIR)
args.batch_size = int(os.environ.get('batch_size', 128))
args.preprocess_whiten = int(os.environ.get('preprocess_whiten', 1))
args.data_collection_pkl = os.environ.get('data_collection_pkl', constants.AGEDB_FACENET_INFERENCES)
args.drift_source_filename = os.environ.get('drift_source_filename', "../data_collection/agedb_drift_synthesis_metrics.csv")
args.metadata = os.environ.get('metadata', constants.AGEDB_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_without_aging')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.no_of_samples = int(os.environ.get('no_of_samples', 2248))
args.colormode = os.environ.get('colormode', 'rgb')
args.conf_threshold = os.environ.get("conf_threshold", 0.9)
args.nms_threshold = os.environ.get("nms_threshold", 0.9)
args.backend = os.environ.get('backend', cv2.dnn.DNN_BACKEND_OPENCV)
args.target = os.environ.get('target', cv2.dnn.DNN_TARGET_CPU)
args.top_k = os.environ.get('top_k', 5000)
args.unique_name_count = os.environ.get('unique_name_count', 40)
args.experiment_id = os.environ.get('experiment_id', 1)
args.face_id = os.environ.get('face_id', 25)
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
    
args.experiment_id = int(args.experiment_id)
args.target = int(args.target)
args.backend = int(args.backend)
args.nms_threshold = float(args.nms_threshold)
args.top_k = int(args.top_k)
args.conf_threshold = float(args.conf_threshold)
args.batch_size = int(args.batch_size)
args.preprocess_whiten = int(args.preprocess_whiten)
args.no_of_samples = int(args.no_of_samples)
args.unique_name_count = int(args.unique_name_count)
args.face_id = int(args.face_id)
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])
    print(args.input_shape)

def get_reduced_metadata(args, dataset):
  """
  Get the reduced metadata
  @param args:
  @param dataset:
  @return: pd.DataFrame()
  """
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    if args.source_type == 'file':
      np.random.seed(1000)
      filenames = pd.read_csv(args.drift_source_filename)
      idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
      result_idx = [False]*len(dataset.metadata)
      for i in idx:
          result_idx = np.logical_or(result_idx, i)
      
      return dataset.metadata.loc[result_idx].reset_index()
    elif args.source_type == 'not_in_file':
      filenames = pd.read_csv(args.drift_source_filename)
      idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
      result_idx = [False]*len(dataset.metadata)
      for i in idx:
          result_idx = np.logical_or(result_idx, i)
      
      return dataset.metadata.loc[np.logical_not(result_idx)].reset_index()
    elif args.source_type == "ordered_metadata":
        names = dataset.metadata.groupby('name').count()
        names = names[names['age'] > args.unique_name_count]
        names = names.index.get_level_values(0)
        idx = [dataset.metadata['name'] == name for name in names]
        result_idx = [False] * len(dataset.metadata)
        for i in idx:
          result_idx = np.logical_or(result_idx, i)
          
        return dataset.metadata.loc[result_idx].sort_values(by=['name', 'age'])
  elif args.dataset == "cacd":
    np.random.seed(1000)
    if args.source_type == 'file':
      filenames = pd.read_csv(args.drift_source_filename)
      idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
      result_idx = [False]*len(dataset.metadata)
      for i in idx:
          result_idx = np.logical_or(result_idx, i)
    
      return dataset.metadata.loc[result_idx].reset_index()
    elif args.source_type == 'not_in_file':
      filenames = pd.read_csv(args.drift_source_filename)
      idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
      result_idx = [False]*len(dataset.metadata)
      for i in idx:
          result_idx = np.logical_or(result_idx, i)
      
      return dataset.metadata.loc[np.logical_not(result_idx)].reset_index()
    elif args.source_type == "ordered_metadata":
      names = dataset.metadata.groupby('name').count()
      names = names[names['age'] > args.unique_name_count]
      names = names.index.get_level_values(0)
      idx = [dataset.metadata['name'] == name for name in names]
      result_idx = [False] * len(dataset.metadata)
      for i in idx:
        result_idx = np.logical_or(result_idx, 1)
        
      return dataset.metadata.loc[result_idx].sort_values(by=['name', 'age'])

def load_dataset(args, whylogs, input_shape=(-1,160,160,3)):
  """
  Load the dataset
  @param args:
  @param whylogs:
  @param input_shape:
  @return:
  """
  dataset = None
  if args.dataset == "agedb":
    augmentation_generator = get_augmented_datasets()
    dataset = AgeDBDataset(whylogs, args.metadata, list_IDs=list(range(args.no_of_samples)), 
                           color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                           dim=(input_shape[1],input_shape[2]), batch_size=args.batch_size)
  elif args.dataset == "cacd":
    augmentation_generator = get_augmented_datasets()
    dataset = CACD2000Dataset(whylogs, args.metadata, list_IDs=list(range(args.no_of_samples)), 
                              color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                              dim=(input_shape[1],input_shape[2]), batch_size=args.batch_size)
  elif args.dataset == "fgnet":
    augmentation_generator = get_augmented_datasets()
    dataset = FGNETDataset(whylogs, args.metadata, list_IDs=list(range(args.no_of_samples)), 
                           color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                           dim=(input_shape[1],input_shape[2]), batch_size=args.batch_size)
  
  return dataset, augmentation_generator

def collect_data_face_recognition_keras(model_loader, train_iterator):
  """
  Collect the images by face recognition using baseline cvae model
  @param model_loader:
  @param train_iterator:
  @return:
  """
  res_images = []
  y_classes = []
  files = []
  ages = []
  labels = []
  images = []
  # Get input and output tensors
  classes_counter = 0
  for i in tqdm(range(len(train_iterator)-1)):
    X, y_label = train_iterator[i]
    # res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    classes = y_label
    unq_classes = np.unique(classes)
    y_valid = np.zeros((len(y_label), 435))
    for c in unq_classes:
      y_valid[classes==c, classes_counter] = 1
      classes_counter += 1
    images.append(X)
    res_images.append(model_loader.infer([X/255., y_valid]))
    labels += y_label

  return images, res_images, labels

def collect_data_facenet_keras(model_loader, train_iterator):
  """
  Collect the images using FaceNet keras
  @param model_loader:
  @param train_iterator:
  @return:
  """
  res_images = []
  files = []
  ages = []
  labels = []
  images = []
  # Get input and output tensors
  for i in tqdm(range(len(train_iterator))):
    X, y_label = train_iterator[i]
    images.append(X)
    res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    labels += y_label
    
  return images, res_images, labels

if __name__ == "__main__":

  # set tracking URI
  mlflow.set_tracking_uri(args.tracking_uri)
  
  # choose the model
  if args.model == 'FaceNetKeras':
    model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
  elif args.model == 'FaceRecognitionBaselineKeras':
    model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
  
  # load the model
  model_loader.load_model()
  
  # load the dataset and set the metadata
  dataset, augmentation_generator = load_dataset(args, whylogs, input_shape=args.input_shape)
  # copy the full metadata for target set and query set declaration
  full_metadata = copy(dataset.metadata)
  
  args.source_type = 'file'
  # obtain the query metadata
  query_metadata = get_reduced_metadata(args, dataset)
  
  args.source_type = 'not_in_file'
  # obtain the target metadata
  target_metadata = get_reduced_metadata(args, dataset)
  
  args.no_of_samples = len(dataset.metadata)
  
  # get iterator for face classification
  iterator = dataset.get_iterator_face_classificaton(args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['name'])
  
  # get all inference images from pickle file
  all_inference_images = pickle.load(open("../data_collection/agedb_inferences_facenet.pkl", "rb"))
  
  if args.model == 'FaceNetKeras':
    images, res_images, labels = collect_data_facenet_keras(model_loader, iterator)
  elif args.model == 'FaceRecognitionBaselineKeras':
    images, res_images, labels = collect_data_face_recognition_keras(model_loader, iterator)
  
  # images = pickle.load(open("images.pkl", "rb"))
  # res_images = pickle.load(open("res_images.pkl", "rb"))
  # labels = pickle.load(open("labels.pkl", "rb"))
  
  # stack the inference images
  all_inference_images = np.vstack(all_inference_images)
  # res_images = np.vstack(res_images)
  
  idx = [full_metadata['filename'] == filename for filename in target_metadata['filename']]
  result_idx = [False]*len(dataset.metadata)
  for i in idx:
      result_idx = np.logical_or(result_idx, i)
      
  target_metadata = full_metadata.loc[result_idx]
      
  idx = [full_metadata['filename'] == filename for filename in query_metadata['filename']]
  result_idx = [False]*len(dataset.metadata)
  for i in idx:
      result_idx = np.logical_or(result_idx, i)
  
  query_metadata = full_metadata.loc[result_idx]
  target_images = all_inference_images[target_metadata.index] # target metadata
  query_images = all_inference_images[query_metadata.index] # query metadata
  
  # target set and query set, euclidean distances pairwise
  euclidean_distances_pair = euclidean_distances(target_images, query_images)
  
  # pandas dataframe by euclidean distances
  data_table_virtual = pd.DataFrame(euclidean_distances_pair, columns=query_metadata.filename, index=target_metadata.filename)

  print(data_table_virtual.shape)
  
  mean_nmr = {}
  mean_mr = {}
  sem = {}
  count = {}
  std = {}

  # sparse matrix using mean matching and mean non matching
  for filename, age_data in data_table_virtual.iteritems():
    mean_nm = age_data.loc[(target_metadata['name'] != filename.split("_")[1]).values].mean()
    std_nm = age_data.loc[(target_metadata['name'] != filename.split("_")[1]).values].std()
    mean_matching = age_data.loc[(target_metadata['name'] == filename.split("_")[1]).values].mean()
    std_matching = age_data.loc[(target_metadata['name'] == filename.split("_")[1]).values].std()
    std[filename.split("_")[1]] = age_data.loc[(target_metadata['name'] == filename.split("_")[1]).values].std()
    count[filename.split("_")[1]] = age_data.loc[(target_metadata['name'] == filename.split("_")[1]).values].count()
    sem[filename.split("_")[1]] = age_data.loc[(target_metadata['name'] == filename.split("_")[1]).values].sem()
    mean_nmr[filename.split("_")[1]] = ((age_data.loc[(target_metadata['name'] == filename.split("_")[1]).values] - mean_nm) / std_nm).mean()
    mean_mr[filename.split("_")[1]] = ((age_data.loc[(target_metadata['name'] != filename.split("_")[1]).values] - mean_matching) / std_matching).mean()

  # set the std to 0 and sem to 0
  nan_std = [name for name, std_value in list(std.items()) if np.isnan(std_value)]
  for name in nan_std:
      std[name] = 0
      sem[name] = 0
      
  # start the mlflow experiment
  with mlflow.start_run(experiment_id=args.experiment_id):
    fig = plt.figure(figsize=(12,8))
    plt.scatter(list(mean_nmr.values()), list(mean_mr.values()))
    plt.xlabel("matching images with non-matching rate (NMR)", fontsize=16)
    plt.ylabel("non-matching images with matching rate (MR)", fontsize=16)
    plt.title("Scatter plot of matching scores and non-matching scores", fontsize=16)
    mlflow.log_figure(fig, "scatter_plot.png")
    
    fig = plt.figure(figsize=(12,8))
    plt.hist(list(mean_nmr.values()), color='red', label='matching images with non-matching rate', alpha=0.4, bins=10)
    plt.hist(list(mean_mr.values()), color='blue', label='non-matching images with matching rate', alpha=0.4, bins=10)
    plt.xlabel("Non-matching scores and Matching scores", fontsize=16)
    plt.title("Histogram plot of matching scores and non-matching scores", fontsize=16)
    plt.legend()
    mlflow.log_figure(fig, "histogram_plot.png")
    
    # non matching rates and matching rates
    nmr = np.array(list(mean_nmr.values()))
    mr = np.array(list(mean_mr.values()))
    tp = len(mr[mr > 0])
    tn = len(nmr[nmr < 0])
    fn = len(mr[mr <= 0])
    fp = len(nmr[nmr >= 0])
    print((tp + tn) / (tp + tn + fp + fn))
    mlflow.log_metric("accuracy", (tp + tn) / (tp + tn + fp + fn))
    
    import seaborn as sns
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    sns.kdeplot(list(mean_nmr.values()), color='red', label='matching images with non-matching rate', ax=ax, fill=True, alpha=0.5)
    sns.kdeplot(list(mean_mr.values()), color='blue', label='non-matching images with matching rate', ax=ax, fill=True, alpha=0.5)
    ax.set_xlabel("Non-matching scores and Matching scores", fontsize=16)
    ax.set_title("Density plot of matching scores and non-matching scores", fontsize=16)
    ax.legend()
    mlflow.log_figure(fig, "density_plot.png")

    