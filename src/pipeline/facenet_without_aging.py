from context import constants
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
import pickle
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='agedb',
                      help='The name of the dataset')
  parser.add_argument('--model', type=str, default='facenet_keras',
                      help='The model to load')
  parser.add_argument('--data_dir', type=str, default=constants.AGEDB_DATADIR,
                      help='The images data directory')
  parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                      help='The batch size for inference')
  parser.add_argument('--preprocess_prewhiten', type=int, default=1, 
                      help='Check for preprocess (prewhiten) to be applied')
  parser.add_argument('--data_collection_pkl', type=str, default=constants.AGEDB_FACENET_INFERENCES, 
                      help='Pickle object for data collection')
  parser.add_argument('--metadata', type=str, default=constants.AGEDB_METADATA, 
                      help='Metadata mat file object that represents the metadata of images')
  parser.add_argument('--logger_name', type=str, default='facenet_without_aging', 
                      help='The name of the logger')
  parser.add_argument('--no_of_samples', type=int, default=2248, 
                      help='The number of samples')
  parser.add_argument('--colormode', type=str, default='color', 
                      help='The type of colormode')
  
  return parser.parse_args()

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
                           color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir)
  elif args.dataset == "cacd":
    augmentation_generator = get_augmented_datasets()
    dataset = CACD2000Dataset(whylogs, args.metadata, list_IDs=list(range(args.no_of_samples)), 
                              color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir)
  elif args.dataset == "fgnet":
    augmentation_generator = get_augmented_datasets()
    dataset = FGNETDataset(whylogs, args.metadata, list_IDs=None, 
                           color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir)
  
  return dataset

if __name__ == "__main__":
  
  args = parse_args()
  
  whylogs = logger.setup_logger(args.logger_name)
  model_loader = KerasModelLoader(whylogs, args.model)
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
    
  result_distances = experiment.calculate_face_distance(embeddings)
