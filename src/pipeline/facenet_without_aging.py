from context import Constants, Args
import tensorflow as tf
from dataloaders import DataGenerator
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.facenet_without_aging import FaceNetWithoutAgingExperiment
from experiment.model_loader import KerasModelLoader
from logger import logger
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

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

"""
Arguments to the pipeline
"""
args.dataset = os.environ.get('dataset', 'agedb')
args.model = os.environ.get('model', 'facenet_keras.h5')
args.data_dir = os.environ.get('data_dir', constants.AGEDB_DATADIR)
args.batch_size = int(os.environ.get('batch_size', 1))
args.preprocess_whiten = int(os.environ.get('preprocess_whiten', 128))
args.data_collection_pkl = os.environ.get('data_collection_pkl', constants.AGEDB_FACENET_INFERENCES)
args.metadata = os.environ.get('metadata', constants.AGEDB_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_without_aging')
args.no_of_samples = int(os.environ.get('no_of_samples', 2248))
args.colormode = os.environ.get('colormode', 'color')

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

def get_reduced_metadata(args, dataset):
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    np.random.seed(1000)
    return dataset.metadata.sample(args.no_of_samples).reset_index()
  elif args.dataset == "cacd":
    np.random.seed(1000)
    return dataset.metadata.sample(args.no_of_samples).reset_index()

def load_dataset(args, whylogs):
  dataset = None
  if args.dataset == "agedb":
    augmentation_generator = get_augmented_datasets()
    dataset = AgeDBDataset(whylogs, args.metadata, list_IDs=list(range(args.no_of_samples)), 
                           color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                           dim=(160,160), batch_size=args.batch_size)
  elif args.dataset == "cacd":
    augmentation_generator = get_augmented_datasets()
    dataset = CACD2000Dataset(whylogs, args.metadata, list_IDs=list(range(args.no_of_samples)), 
                              color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                              dim=(160,160), batch_size=args.batch_size)
  elif args.dataset == "fgnet":
    augmentation_generator = get_augmented_datasets()
    dataset = FGNETDataset(whylogs, args.metadata, list_IDs=None, 
                           color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                           dim=(160,160), batch_size=args.batch_size)
  
  return dataset

if __name__ == "__main__":
  
  whylogs = logger.setup_logger(args.logger_name)
  model_loader = KerasModelLoader(whylogs, args.model, input_shape=(-1,160,160,3))
  model_loader.load_model()
  
  # load the dataset and set the metadata
  dataset = load_dataset(args, whylogs)
  dataset.set_metadata(
    get_reduced_metadata(args, dataset)
  )
  
  # setup experiment
  experiment = FaceNetWithoutAgingExperiment(dataset, logger=whylogs, model_loader=model_loader)
  
  """
  Collect Data
  """
  embeddings = experiment.collect_data(args.data_collection_pkl, args.data_dir, args.batch_size)
    
  result_euclidean_distances, result_cosine_similarities = experiment.calculate_face_distance(embeddings)

  if not os.path.isfile(args.data_collection_pkl):
    data_collection_pkl = Path(args.data_collection_pkl)
    
    with open(data_collection_pkl, 'wb') as data_collection_file:
      pickle.dump(embeddings, data_collection_file)
      
