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
from experiment.facenet_with_classifier import FaceNetWithClassifierExperiment, FaceNetWithClassifierPredictor
from experiment.model_loader import FaceNetKerasModelLoader, FaceRecognitionBaselineKerasModelLoader
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from preprocessing.facenet import l2_normalize, prewhiten
from sklearn.utils.fixes import loguniform
from tqdm import tqdm
from copy import copy
import scipy
import pickle
import logging
import sys
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
args.no_of_samples = os.environ.get('no_of_samples', 6400)
args.colormode = os.environ.get('colormode', 'rgb')
args.log_images = os.environ.get('log_images', 's3')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.collect_for = os.environ.get('collect_for', 'age_drifting')
args.experiment_id = os.environ.get("experiment_id", 1)
args.drift_synthesis_metrics = os.environ.get('drift_synthesis_metrics', constants.AGEDB_DRIFT_SYNTHESIS_METRICS)
args.drift_model_or_data = os.environ.get('drift_model_or_data', 'model_drift')
args.alpha = os.environ.get('alpha', 0.01)
args.t2_observation_ucl = os.environ.get('t2_observation_ucl', "../data_collection/t2_observation_ucl.csv")
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
args.alpha = float(args.alpha)
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])
    print(args.input_shape)
    
def collect_images(train_iterator):
    images = []
    # Get input and output tensors
    for ii in tqdm(range(len(train_iterator))):
        (X, y) = train_iterator[ii]
        images.append(X)

    return np.vstack(images)

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
    filenames = pd.read_csv(args.drift_synthesis_metrics, index_col=0)
    idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
    result_idx = [False]*len(dataset.metadata)
    for i in idx:
        result_idx = np.logical_or(result_idx, i)
      
    filenames.test_younger = filenames.test_younger.astype(int)
    filenames.train_younger = filenames.train_younger.astype(int)
    filenames.difference = filenames.difference.astype(int)
    filenames.age = filenames.age.astype(int)
    dataset.metadata = dataset.metadata.loc[result_idx].reset_index()
    dataset.metadata = dataset.metadata.set_index('fileno').join(filenames[['filename', 'test_younger', 'train_younger', 'difference']], lsuffix='_left', rsuffix='')
    dataset.metadata['fileno'] = dataset.metadata.index.get_level_values(0)
    return dataset.metadata
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()

def covariance(embeddings):
    return np.cov(embeddings)

def t_squared_distribution(embeddings):
    mean = embeddings.mean(axis=0)
    cov = covariance(embeddings)
    
    return embeddings.shape[0] * (embeddings - mean) * np.linalg.inv(cov) * (embeddings - mean).T

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
    
    images = collect_images(dataset.iterator)
    
    if args.model == 'FaceNetKeras':
        embeddings = model_loader.infer(l2_normalize(prewhiten(images.reshape(-1,args.input_shape[1], args.input_shape[2],3))))
    elif args.model == 'FaceRecognitionBaselineKeras':
        embeddings = model_loader.infer((images.reshape(-1,args.input_shape[1], args.input_shape[2],3))/255.)
    
    mean = embeddings.mean(axis=0)
    cov = covariance(embeddings.T)
    cov_inverse = np.linalg.inv(cov)
    
    # squared mahalanobis distances
    def mahalanobis_squared_observation(embedding):
        return (embedding.reshape(1,-1) - mean).dot(cov_inverse).dot((embedding.reshape(1,-1) - mean).T)
    
    # T2-UCL
    def t_squared_beta(embedding, alpha=args.alpha):
        return (embedding.shape[0] - 1)**2 / embedding.shape[0] * np.quantile(scipy.stats.beta.pdf(embedding, 1/2, (embedding.shape[0] - 1 - 1)/2), q=1-alpha)
    
    df = pd.DataFrame(
        np.concatenate([np.array(list(map(mahalanobis_squared_observation, embeddings))).reshape(-1,1), np.array(list(map(t_squared_beta, embeddings))).reshape(-1,1)], axis=1), 
        columns=['observation', 'ucl_alpha_0.01']
    )
    
    for alpha in tqdm(np.arange(0.02, 1.0, 0.1)):
        df['ucl_alpha_' + str(alpha)] = np.array(list(map(t_squared_beta, embeddings)))
    
    df = df.set_index(dataset.metadata.fileno)
    df['filename'] = dataset.metadata.filename_left
    df['test_younger'] = dataset.metadata.loc[:, 'test_younger']
    df['train_younger'] = dataset.metadata.loc[:, 'train_younger']
    df['difference'] = dataset.metadata.loc[:, 'difference']
    
    df.to_csv(args.t2_observation_ucl)
    