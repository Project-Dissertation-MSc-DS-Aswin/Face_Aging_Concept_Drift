from context import Constants, Args
import tensorflow as tf
from dataloaders import DataGenerator
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.facenet_without_aging import FaceNetWithoutAgingExperiment
from experiment.model_loader import FaceNetKerasModelLoader, FaceRecognitionBaselineKerasModelLoader
import whylogs
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import re
import pickle
import argparse
from pathlib import Path
import logging
import mlflow
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
args.model = os.environ.get('model', 'FaceNetKeras')
args.model_path = os.environ.get('model_path', 'facenet_keras.h5')
args.data_dir = os.environ.get('data_dir', constants.AGEDB_DATADIR)
args.batch_size = int(os.environ.get('batch_size', 128))
args.preprocess_whiten = int(os.environ.get('preprocess_whiten', 1))
args.data_collection_pkl = os.environ.get('data_collection_pkl', constants.AGEDB_FACENET_INFERENCES)
args.metadata = os.environ.get('metadata', constants.AGEDB_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_without_aging')
args.no_of_samples = int(os.environ.get('no_of_samples', 2248))
args.colormode = os.environ.get('colormode', 'rgb')
args.experiment_id = os.environ.get('experiment_id', 1)
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
args.batch_size = int(args.batch_size)
args.preprocess_whiten = int(args.preprocess_whiten)
args.no_of_samples = int(args.no_of_samples)
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])
    print(args.input_shape)

def get_reduced_metadata(args, dataset):
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    np.random.seed(1000)
    names = np.unique(dataset.metadata['name'])[:435]
    idx = [dataset.metadata['name'] == name for name in names]
    result_idx = [False] * len(dataset.metadata)
    for i in idx:
        result_idx = np.logical_or(result_idx, i)

    return dataset.metadata.loc[result_idx].sort_values(by=['name', 'age'])
  elif args.dataset == "cacd":
    np.random.seed(1000)
    return dataset.metadata.sample(args.no_of_samples).reset_index()

def load_dataset(args, whylogs, input_shape=(-1,160,160,3)):
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
    dataset = FGNETDataset(whylogs, args.metadata, list_IDs=None, 
                           color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                           dim=(input_shape[1],input_shape[2]), batch_size=args.batch_size)
  
  return dataset, augmentation_generator

if __name__ == "__main__":

  if args.model == 'FaceNetKeras':
    model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
  elif args.model == 'FaceRecognitionBaselineKeras':
    model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
  
  model_loader.load_model()
  
  # load the dataset and set the metadata
  dataset, augmentation_generator = load_dataset(args, whylogs, input_shape=args.input_shape)
  dataset.set_metadata(
    get_reduced_metadata(args, dataset)
  )

  iterator = dataset.get_iterator_face_classificaton(args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['name'])
  
  # setup experiment
  experiment = FaceNetWithoutAgingExperiment(dataset, logger=whylogs, model_loader=model_loader)
  
  """
  Collect Data
  """
  embeddings = experiment.collect_data(args.data_collection_pkl, args.data_dir, args.batch_size, iterator=iterator, model=args.model)

  result_euclidean_distances, result_cosine_similarities = experiment.calculate_face_distance(embeddings)

  # if not os.path.isfile(args.data_collection_pkl):
  #   data_collection_pkl = Path(args.data_collection_pkl)
  #
  #   with open(data_collection_pkl, 'wb') as data_collection_file:
  #     pickle.dump(embeddings, data_collection_file)
      
  with mlflow.start_run(experiment_id=args.experiment_id):
    
    if args.datset == 'agedb':
      counter_labels = Counter(dataset.metadata['name'])
    elif args.dataset == 'cacd':
      counter_labels = Counter(dataset.metadata['identity'])
    elif args.dataset == 'fgnet':
      counter_labels = Counter(dataset.metadata['fileno'])
      
    name_first = list(counter_labels.keys())[0]

    array = np.append(result_euclidean_distances[0, :counter_labels[name_first]].flatten(), \
      result_euclidean_distances[0, counter_labels[name_first]:].flatten())
    y_labels = [1]*result_euclidean_distances[0, :counter_labels[name_first]].flatten().shape[0] + \
      [0]*result_euclidean_distances[0, counter_labels[name_first]:].flatten().shape[0]

    for ii, (identity, label) in tqdm(enumerate(counter_labels.items())):
        if ii == 0:
            continue
        array = np.append(array, result_euclidean_distances[ii, :counter_labels[identity]].flatten())
        y_labels += [1]*result_euclidean_distances[ii, :counter_labels[identity]].flatten().shape[0]
        array = np.append(array, result_euclidean_distances[ii, counter_labels[identity]:].flatten())
        y_labels += [0]*result_euclidean_distances[ii, counter_labels[identity]:].flatten().shape[0]
        
    array_cos = np.append(result_cosine_similarities[0, :counter_labels[name_first]].flatten(), \
      result_cosine_similarities[0, counter_labels[name_first]:].flatten())

    for ii, (identity, label) in tqdm(enumerate(counter_labels.items())):
        if ii == 0:
            continue
        array_cos = np.append(array_cos, result_cosine_similarities[ii, :counter_labels[identity]].flatten())
        array_cos = np.append(array_cos, result_cosine_similarities[ii, counter_labels[identity]:].flatten())

    array_full = np.concatenate([array.reshape(-1, 1), array_cos.reshape(-1, 1)], axis=1)

    fig = plt.figure(figsize=(12, 8))
    plt.scatter(array_full[np.array(y_labels) == 0, 0], array_full[np.array(y_labels) == 0, 1], color='blue',
                label='Non-similar Images')
    plt.scatter(array_full[np.array(y_labels) == 1, 0], array_full[np.array(y_labels) == 1, 1], color='orange',
                label='Similar Images')
    plt.xlabel("Euclidean Distance", fontsize=16)
    plt.ylabel("Cosine Similarity", fontsize=16)
    plt.legend()
    plt.savefig("scatter_plot.png")