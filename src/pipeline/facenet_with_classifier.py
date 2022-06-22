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
from experiment.model_loader import KerasModelLoader
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from sklearn.utils.fixes import loguniform
from tqdm import tqdm
from copy import copy
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
args.model = os.environ.get('model', 'facenet_keras.h5')
args.data_dir = os.environ.get('data_dir', constants.AGEDB_DATADIR)
args.batch_size = os.environ.get('batch_size', 128)
args.preprocess_prewhiten = os.environ.get('preprocess_prewhiten', 1)
args.data_collection_pkl = os.environ.get('data_collection_pkl', constants.AGEDB_FACENET_INFERENCES)
args.metadata = os.environ.get('metadata', constants.AGEDB_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_with_aging')
args.no_of_samples = os.environ.get('no_of_samples', 2248)
args.colormode = os.environ.get('colormode', 'rgb')
args.log_images = os.environ.get('log_images', 's3')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.classifier = os.environ.get('classifier', constants.AGEDB_FACE_CLASSIFIER)

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

def load_dataset(args, whylogs, image_dim, no_of_samples, colormode):
    dataset = None
    augmentation_generator = None
    if args.dataset == "agedb":
        augmentation_generator = get_augmented_datasets()
        dataset = AgeDBDataset(whylogs, args.metadata, list_IDs=list(range(no_of_samples)),
                               color_mode=colormode, augmentation_generator=augmentation_generator, data_dir=args.data_dir, dim=image_dim, 
                               batch_size=args.batch_size)
    elif args.dataset == "cacd":
        augmentation_generator = get_augmented_datasets()
        dataset = CACD2000Dataset(whylogs, args.metadata, list_IDs=list(range(no_of_samples)),
                                  color_mode=colormode, augmentation_generator=augmentation_generator,
                                  data_dir=args.data_dir, dim=image_dim, batch_size=args.batch_size)
    elif args.dataset == "fgnet":
        augmentation_generator = get_augmented_datasets()
        dataset = FGNETDataset(whylogs, args.metadata, list_IDs=None,
                               color_mode=colormode, augmentation_generator=augmentation_generator, data_dir=args.data_dir, dim=image_dim, 
                               batch_size=args.batch_size)

    return dataset, augmentation_generator
  
def get_reduced_metadata(args, dataset, seed=1000):
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()
  
if __name__ == "__main__":
  
    mlflow.set_tracking_uri(args.tracking_uri)
    
    model_loader = KerasModelLoader(whylogs, args.model, input_shape=(-1,160,160,3))
    model_loader.load_model()

    dataset, augmentation_generator = load_dataset(args, whylogs, (49,49), args.no_of_samples, 'grayscale')
    
    dataset.set_metadata(
        get_reduced_metadata(args, dataset)
    )
    
    experiment = FaceNetWithClassifierExperiment(dataset, whylogs, model_loader)
    
    face_classification_iterator = dataset.get_iterator_face_classificaton(
      args.colormode, args.batch_size, args.data_dir, augmentation_generator, x_col='filename', y_cols=['age', 'filename', 'name']
    )
    
    embeddings, files, ages, labels = \
      experiment.collect_data(args.data_collection_pkl, face_classification_iterator)
      
    param_grid = {
      "C": loguniform(0.1, 100),
      "gamma": loguniform(1e-4, 1e-1),
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
    
    algorithm = FaceNetWithClassifierPredictor(metadata=dataset.metadata)
    
    algorithm.make_train_test_split(embeddings, files, ages, labels)
    
    dataframe = algorithm.make_dataframe(algorithm.embeddings_train, algorithm.files_train, algorithm.ages_train, algorithm.labels_train)
    
    if args.collect_for == "classification":
      faces_chunk_array_train, face_classes_array_train = \
        algorithm.make_data(algorithm.labels_train, algorithm.embeddings_train, algorithm.dataframe)
    elif args.colect_for == "age_drifting":
      faces_chunk_array_train, face_classes_array_train = \
        algorithm.make_data_age(algorithm.labels_train, algorithm.embeddings_train, algorithm.dataframe)
      
    score_embedding, face_classes_count, (voting_classifier_array, 
                                                 svm_embedding_array, 
                                                 rf_embedding_array, 
                                                 hist_embedding_array, 
                                                 knn_embeding_array) = algorithm.train_and_evaluate(
      faces_chunk_array_train, face_classes_array_train, 
      param_grid, param_grid2, param_grid3, no_of_classes=3
    )
    
    svm_emb_array = [svm_cv.best_estimator_ for svm_cv in svm_embedding_array]
    rf_emb_array = [rf_cv.best_estimator_ for rf_cv in rf_embedding_array]
    h_emb_array = [hist.best_estimator_ for hist in hist_embedding_array]
    
    c_array = [svm_model.C for svm_model in svm_emb_array]
    split = [rf_model.min_samples_split for rf_model in rf_emb_array]
    depth = [hist_model.max_depth for hist_model in h_emb_array]
    leaf = [hist_model.min_samples_leaf for hist_model in h_emb_array]
    max_iter = [hist_model.max_iter for hist_model in h_emb_array]
    lr = [hist_model.learning_rate for hist_model in h_emb_array]
    criterion = [rf_model.criterion for rf_model in rf_emb_array]
    
    with mlflow.start_run():
      mlflow.log_metric("score_embedding_average", np.mean(score_embedding))
      mlflow.log_metric("score_embedding_weighted_average", score_embedding * np.array(face_classes_count) / np.sum(face_classes_count))
      mlflow.log_metric("standard_error", pd.DataFrame((np.array(score_embedding * np.array(face_classes_count)) / np.sum(face_classes_count))).sem() * np.sqrt(len(score_embedding)))
      mlflow.log_metric("SVM_C_Values", c_array)
      mlflow.log_metric("Hist_Depth", depth)
      mlflow.log_metric("Hist_Max_Iter", max_iter)
      mlflow.log_metric("Hist_Learning_Rate", lr)
      mlflow.log_metric("Hist_Leaf", leaf)
      mlflow.log_metric("RF_Criterion", criterion)
      mlflow.log_metric("RF_Split", split)
