import sys
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import euclidean_distances
from context import Constants, Args
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA
import whylogs
import mlflow
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.facenet_with_classifier import collect_data_face_recognition_keras, collect_data_facenet_keras
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
    filenames = pd.read_csv(args.drift_synthesis_metrics, index_col=0).index.get_level_values(0)
    idx = [dataset.metadata['filename'] == filename for filename in filenames]
    result_idx = [False]*len(dataset.metadata)
    for i in idx:
        result_idx = np.logical_or(result_idx, i)
      
    return dataset.metadata.loc[result_idx].reset_index()
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()

if __name__ == "__main__":
    
    mlflow.set_tracking_uri(args.tracking_uri)

    if args.model == 'FaceNetKeras':
      model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    elif args.model == 'FaceRecognitionBaselineKeras':
      model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    
    model_loader.load_model()

    dataset, augmentation_generator = load_dataset(args, whylogs, args.no_of_samples, 'rgb', input_shape=args.input_shape)
    
    experiment_dataset, augmentation_generator = load_dataset(args, whylogs, args.no_of_samples, 'rgb', input_shape=args.input_shape)
    
    dataset.set_metadata(
        get_reduced_metadata(args, dataset)
    )
    
    experiment_dataset.set_metadata(
        get_reduced_metadata(args, dataset)
    )
    
    images = collect_images(dataset.iterator)
    
    pickle.dump(images, open("images.pkl", "wb"))
    
    # if args.model == 'FaceNetKeras':
    #   embeddings = model_loader.infer(l2_normalize(prewhiten(images.reshape(-1,args.input_shape[1], args.input_shape[2],3))))
    # elif args.model == 'FaceRecognitionBaselineKeras':
    #   embeddings = model_loader.infer((images.reshape(-1,args.input_shape[1], args.input_shape[2],3))/255.)
      
    # if args.dataset == 'agedb':
    #   face_classification_iterator = experiment_dataset.get_iterator_face_classificaton(
    #     args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'name']
    #   )
    # elif args.dataset == 'cacd':
    #   face_classification_iterator = experiment_dataset.get_iterator_face_classificaton(
    #     args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'identity']
    #   )
    # elif args.dataset == 'fgnet':
    #   face_classification_iterator = experiment_dataset.get_iterator_face_classificaton(
    #     args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'fileno']
    #   )
    
    # if args.model == 'FaceNetKeras':
    #   embeddings_all, files, ages, labels = collect_data_facenet_keras(model_loader, face_classification_iterator)
    # elif args.model == 'FaceRecognitionBaselineKeras':
    #   embeddings_all, files, ages, labels = collect_data_face_recognition_keras(model_loader, face_classification_iterator)
      
    # euclidean_embeddings = euclidean_distances(np.vstack(embeddings_all), embeddings)
    
    # df = pd.DataFrame(euclidean_embeddings, index=experiment_dataset.metadata.filename.values, columns=dataset.metadata.filename.values)
    
    # # building sparse matrix
    # for idx, row in df.iterrows():
    #   cols = [i for i in range(dataset.metadata.age.values.shape[0]) if dataset.metadata.age.values[i] != experiment_dataset.metadata.age.values[idx]]
    #   df.loc[idx, cols] = 0
    
    # df.to_csv("file_euclidean.csv")
    
    # df = pd.read_csv("file_euclidean.csv", index_col=0)
    
    # from scipy.sparse import csr_matrix
    # from scipy.sparse.csgraph import minimum_spanning_tree

    # result = minimum_spanning_tree(csr_matrix(df.values)).toarray()
    
    # # count number of trees
    # pd.DataFrame(result, index=experiment_dataset.metadata.filename.values, columns=dataset.metadata.filename.values).to_csv("file2_euclidean.csv")