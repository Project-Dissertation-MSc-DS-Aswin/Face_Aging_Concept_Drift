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
from experiment.face_with_classifier import collect_data_face_recognition_keras, collect_data_facenet_keras
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
import mistree as mist

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
    """
    Collect images
    @param train_iterator:
    @return: np.ndarray
    """
    images = []
    labels = []
    # Get input and output tensors
    for ii in tqdm(range(len(train_iterator))):
        (X, y) = train_iterator[ii]
        images.append(X)
        labels.append(y)

    return np.vstack(images), labels

def load_dataset(args, whylogs, no_of_samples, colormode, input_shape=(-1,160,160,3)):
    """
    Load the dataset
    @param args:
    @param whylogs:
    @param no_of_samples:
    @param colormode:
    @param input_shape:
    @return: tuple()
    """
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
  """
  Get reduced metadata
  @param args:
  @param dataset:
  @param seed:
  @return: pd.DataFrame()
  """
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
    
    # set mlflow tracking URI
    mlflow.set_tracking_uri(args.tracking_uri)

    # choose the model
    if args.model == 'FaceNetKeras':
      model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    elif args.model == 'FaceRecognitionBaselineKeras':
      model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    
    # load the model
    model_loader.load_model()

    # load the experiment dataset
    experiment_dataset, augmentation_generator = load_dataset(args, whylogs, args.no_of_samples, 'rgb', input_shape=args.input_shape)
    
    # set metadata for experiment dataset
    experiment_dataset.set_metadata(
        get_reduced_metadata(args, experiment_dataset)
    )
    
    # iterator for dataset
    if args.dataset == 'agedb':
      face_classification_iterator = experiment_dataset.get_iterator_face_classificaton(
        args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'name']
      )
    elif args.dataset == 'cacd':
      face_classification_iterator = experiment_dataset.get_iterator_face_classificaton(
        args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'identity']
      )
    elif args.dataset == 'fgnet':
      face_classification_iterator = experiment_dataset.get_iterator_face_classificaton(
        args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'fileno']
      )
    
    # collect embeddings from model
    if args.model == 'FaceNetKeras':
      embeddings_all, files, ages, labels = collect_data_facenet_keras(model_loader, face_classification_iterator)
    elif args.model == 'FaceRecognitionBaselineKeras':
      embeddings_all, files, ages, labels = collect_data_face_recognition_keras(model_loader, face_classification_iterator)
      
    # euclidean distances
    euclidean_embeddings = euclidean_distances(np.vstack(embeddings_all))
    
    # dataframe
    euclidean_data = pd.DataFrame(euclidean_embeddings, index=experiment_dataset.metadata.filename.values, columns=experiment_dataset.metadata.filename.values)
    
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    lesser_age_group = experiment_dataset.metadata.groupby('name')['age'].mean().sort_values()[experiment_dataset.metadata.groupby('name')['age'].mean().sort_values() <= 38.50671030080361].index
    
    older_age_group = experiment_dataset.metadata.groupby('name')['age'].mean().sort_values()[experiment_dataset.metadata.groupby('name')['age'].mean().sort_values() > 38.50671030080361].index

    euclidean_data['filename'] = euclidean_data.index.get_level_values(0)
    euclidean_data = euclidean_data.drop(columns=['filename']).groupby('name').mean().T
    euclidean_data['name'] = euclidean_data['filename'].apply(lambda x: "".join(list(map(lambda y: str(y) if (ord(y[0]) >= 65 and ord(y[0]) <= 127) else "", x.split("_")))))
    euclidean_data = euclidean_data.drop(columns=['filename']).groupby('name').mean().T
    
    euclidean_data = euclidean_data.loc[lesser_age_group.values.tolist() + older_age_group.values.tolist(), lesser_age_group.values.tolist() + older_age_group.values.tolist()]

    labels = np.array([0]*len(lesser_age_group) + [1]*len(older_age_group))

    result = minimum_spanning_tree(csr_matrix(euclidean_data)).toarray()
    idx = np.where([(result != 0)])

    # start row-wise
    indices_array = idx[1], idx[2]
    # values truncated
    values_array = result[idx[1], idx[2]]

    import networkx as nx

    G = nx.Graph()
    V = set(list(range(len(euclidean_data))))
    E = list(zip(indices_array[0], indices_array[1], values_array))
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)

    from tqdm import tqdm

    paths = []
    for i in tqdm(range(len(idx[1]))):
      for j in range(i, len(idx[1])):
        # edge distances
        try:
          path = nx.dijkstra_path(G,source=idx[1][i],target=idx[1][j], weight='weight')
          paths.append(path)
        except Exception as e:
          print(e)

    new_paths = [p for p in paths if len(p) > 2]

    new_df = pd.DataFrame(np.zeros((len(new_paths), 3)), columns=['key', 'sum_distance', 'avg_distance'], index=list(range(len(new_paths))))
    for ii, tree in enumerate(new_paths):
      new_df.loc[ii, 'key'] = str(tree)
      new_df.loc[ii, 'sum_distance'] = euclidean_data.iloc[list(tree)].values[euclidean_data.iloc[list(tree)].values!=0].sum()
      new_df.loc[ii, 'avg_distance'] = euclidean_data.iloc[list(tree)].values[euclidean_data.iloc[list(tree)].values!=0].sum() / len(tree)
      new_df.loc[ii, 'age_group'] = str([labels[t] for t in list(tree)])
      new_df.loc[ii, 'higher_age_count'] = sum([labels[t] for t in list(tree)])
      new_df.loc[ii, 'lesser_age_count'] = len([labels[t] for t in list(tree)]) - sum([labels[t] for t in list(tree)])
      
    from scipy.stats import norm

    R = len(new_df)
    # m = drift_file.loc[drift_file['orig_TP'] == 0].shape[0]
    m = new_df['higher_age_count'].sum()
    # n = drift_file.loc[drift_file['orig_TP'] == 1].shape[0]
    n = new_df['lesser_age_count'].sum()
    N = m + n

    mu = 2*m*n / N + 1
    sigma = (2*m*n*(2*m*n - N) / ((N**2)*(N-1)))**0.5

    W = (R - mu) / sigma
    print(W)

    z = (np.abs(R - mu) - 0.5) / sigma
    p_val = norm.sf(z)

    print('Z-score: ', z)
    print('p-value: ', p_val)