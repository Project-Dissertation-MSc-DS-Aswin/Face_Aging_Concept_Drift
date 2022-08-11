import sys
from sklearn.ensemble import VotingClassifier
from context import Constants, Args
import os
# import jax
# import jax.numpy as jnp
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
args.logger_name = os.environ.get('logger_name', 'facenet_cda_fedavg')
args.no_of_samples = os.environ.get('no_of_samples', 6400)
args.colormode = os.environ.get('colormode', 'rgb')
args.log_images = os.environ.get('log_images', 's3')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.collect_for = os.environ.get('collect_for', 'age_drifting')
args.experiment_id = os.environ.get("experiment_id", 1)
args.drift_synthesis_metrics = os.environ.get('drift_synthesis_metrics', constants.AGEDB_DRIFT_SYNTHESIS_METRICS)
args.drift_model_or_data = os.environ.get('drift_model_or_data', 'model_drift')
args.alpha = os.environ.get('alpha', 0.01)
args.delta = os.environ.get('delta', 1)
args.cda_fedavg_observation = os.environ.get('cda_fedavg_observation', "../data_collection/cda_fedavg_observation.csv")
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
      
    return dataset.metadata.loc[result_idx].reset_index().sort_values(by=['age'])
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()
  
def execute_drift_hypothesis(dataset, patches, ii, N=100, delta=25):
    
    l = 1e-15
    
    def drift_detection(lambda_, patches, N=N, delta=delta):
        s_f1 = 0
        diff1 = 0
        T_h = -np.log(lambda_)
        t = (delta, N-delta)
        for k in np.arange(t[0], t[1], 1):
            m_b = patches[0:k].mean()
            m_a = patches[k+1:N].mean()
            # m_c = patches.mean()
            a_or_b = 'a' if m_a < m_b else 'b'
            # here m_a should be less than m_b, if not check whether they are ok to go for if clause

            data = []
            if ((m_a <= (1-lambda_) * m_b)):# and a_or_b == 'a' or ((m_b <= (1-lambda_) * m_a) and a_or_b == 'b'):
                s_k = 0
                alpha_b, c_b, loc_b, scale_b = scipy.stats.beta.fit(patches[0:k].flatten())
                alpha_a, c_a, loc_a, scale_a = scipy.stats.beta.fit(patches[k+1:N].flatten())
                c = range(k+1, N)
                for i in c:
                    if len(patches[i:i+1]) == 0:
                        break
                    a_pdf = scipy.stats.beta.pdf(patches[i:i+1], alpha_a, c_a, loc_a, scale_a)
                    b_pdf = scipy.stats.beta.pdf(patches[i:i+1], alpha_b, c_b, loc_b, scale_b)
                    # if a_or_b == 'a':
                    res = np.log((a_pdf.prod()) / (b_pdf.prod()))
                    diff = (1 / (a_pdf)).sum() - (1 / (b_pdf)).sum()
                    # elif a_or_b == 'b':
                    #     res = np.log((b_pdf.prod()+1e-30) / (a_pdf.prod()+1e-30))
                    #     diff = (1 / (b_pdf)).sum() - (1 / (a_pdf)).sum()
                    if res != np.nan:
                        s_k += np.abs(res)
                    if diff != np.nan:
                        diff1 += diff
                s_f1 = max(s_f1, s_k)
                    
        try:
            data.append([s_f1,diff1,T_h])
        except Exception as e:
            data.append(np.zeros(3).tolist())
            
        return data

    array = []
    try:
      result = drift_detection(l, patches, N=100)
    except Exception as e:
      result = None
      raise e
    res = result if result is not None else np.zeros(3).tolist()
    array.append(res)
    df_orig = pd.DataFrame(np.array(array).reshape(-1,3), columns=['s_f1','diff1','T_h'])
    df_orig['age'] = dataset.metadata['age'].iloc[ii]
    df_orig['filename'] = dataset.metadata['filename'].iloc[ii]
    
    return df_orig

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
        
    # embeddings_cacd = pickle.load(open("../data_collection/embeddings_cacd_age_estimations.pkl"))
    
    
    
    if args.dataset == 'agedb':
      ident = np.unique(dataset.metadata['name'])
    elif args.dataset == 'cacd':
      ident = np.unique(dataset.metadata['identity'])
    elif args.dataset == 'fgnet':
      ident = np.unique(dataset.metadata['fileno'])
    
    print("Unique identities: ", ident.shape)
    print(ident)
    
    total_df = []
    N = 3
    delta = 0
    
    for ii, embedding in tqdm(enumerate(range(len(embeddings)))):
      df_orig = execute_drift_hypothesis(dataset, embeddings[embedding:embedding+N], ii, N, delta)
      total_df.append(df_orig)
      
    total_df = pd.concat(total_df, axis=0)
    total_df.to_csv(args.cda_fedavg_observation)
