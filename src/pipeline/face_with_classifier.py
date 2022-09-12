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
from experiment.face_with_classifier import FaceNetWithClassifierExperiment, FaceNetWithClassifierPredictor
from experiment.model_loader import FaceNetKerasModelLoader, FaceRecognitionBaselineKerasModelLoader
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from sklearn.utils.fixes import loguniform
from tqdm import tqdm
from copy import copy
import pickle
import logging
import sys
import re
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
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
args.classifier = os.environ.get('classifier', constants.AGEDB_FACE_CLASSIFIER)
args.collect_for = os.environ.get('collect_for', 'age_drifting')
args.drift_evaluate_metrics = os.environ.get('drift_evaluate_metrics', constants.AGEDB_DRIFT_EVALUATE_METRICS)
args.experiment_id = os.environ.get("experiment_id", 1)
args.log_file_younger = os.environ.get("log_file_younger", "test_data_predictions_younger.csv")
args.log_file_older = os.environ.get("log_file_older", "test_data_predictions_older.csv")
args.log_file = os.environ.get("log_file", "test_data_predictions.csv")
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
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])
    print(args.input_shape)

def load_dataset(args, whylogs, no_of_samples, colormode, input_shape=(-1,160,160,3)):
    """
    Load the dataset
    @param args:
    @param whylogs:
    @param image_dim:
    @param no_of_samples:
    @param colormode:
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
  
if __name__ == "__main__":
    # set mlflow experiment
    mlflow.set_tracking_uri(args.tracking_uri)
    
    # choose model
    if args.model == 'FaceNetKeras':
      model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    elif args.model == 'FaceRecognitionBaselineKeras':
      model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    
    # load the model
    model_loader.load_model()

    # load the dataset
    dataset, augmentation_generator = load_dataset(args, whylogs, args.no_of_samples, 'rgb', input_shape=args.input_shape)
    
    # set the metadata for the dataset
    dataset.set_metadata(
        get_reduced_metadata(args, dataset)
    )
    
    # Experiment started with FaceNet Classifier
    experiment = FaceNetWithClassifierExperiment(dataset, whylogs, model_loader)
    
    # choose the iterator based on dataset
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
    
    # collect the data from face classification iterator
    embeddings, files, ages, labels = \
      experiment.collect_data(args.data_collection_pkl, face_classification_iterator, model=args.model)
      
    # param grid
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
    
    # algorithm created with facenet with classifier predictor
    algorithm = FaceNetWithClassifierPredictor(metadata=dataset.metadata, model_loader=model_loader)
    
    # algorithm train_test_split
    algorithm.make_train_test_split(embeddings, files, ages, labels)
    
    # choice of classification or age_drifting
    if args.collect_for == "classification":
      # create data for training testing
      scaler = StandardScaler()
      algorithm.embeddings_train = scaler.fit_transform(algorithm.embeddings_train)
      algorithm.embeddings_test = scaler.transform(algorithm.embeddings_test)
      dataframe = algorithm.make_dataframe(algorithm.embeddings_train, algorithm.labels_train, algorithm.ages_train, algorithm.files_train)
      faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test = \
        algorithm.make_data(algorithm.labels_train, algorithm.embeddings_train, dataframe)
        
      with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
        score_embedding_test, score_embedding_train, face_classes_count_test, face_classes_count_train, (voting_classifier_array, 
                                                    svm_embedding_array, 
                                                    rf_embedding_array, 
                                                    hist_embedding_array, 
                                                    knn_embeding_array) = algorithm.train_and_evaluate(
          faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test, 
          param_grid, param_grid2, param_grid3, no_of_classes=3, original_df=pd.DataFrame(columns=['test_labels', 'test_predictions']), log_file=args.log_file
        )
      
      # create dataframe using test set
      dataframe = algorithm.make_dataframe(algorithm.embeddings_test, algorithm.labels_test, algorithm.ages_test, algorithm.files_test)
      # create data for training testing
      faces_chunk_array_train, face_classes_array_train = \
      algorithm.make_data(algorithm.labels_test, algorithm.embeddings_test, dataframe)
      
    # age_drifting
    elif args.collect_for == "age_drifting":
      scaler = StandardScaler()
      algorithm.embeddings_train = scaler.fit_transform(algorithm.embeddings_train)
      algorithm.embeddings_test = scaler.transform(algorithm.embeddings_test)
      dataframe = algorithm.make_dataframe(algorithm.embeddings_train, algorithm.labels_train, algorithm.ages_train, algorithm.files_train)
      # train younger and test older scenario
      faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test = \
        algorithm.make_data_age_train_younger(algorithm.labels_train, algorithm.embeddings_train, dataframe, age_low=47, age_high=48)
        
      # set mlflow experiment to run
      with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
        score_embedding_test, score_embedding_train, face_classes_count_test, face_classes_count_train, (voting_classifier_array, 
                                                    svm_embedding_array, 
                                                    rf_embedding_array, 
                                                    hist_embedding_array, 
                                                    knn_embeding_array) = algorithm.train_and_evaluate(
          faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test, 
          param_grid, param_grid2, param_grid3, no_of_classes=3
        )
      
      # create dataframe from testing set
      dataframe = algorithm.make_dataframe(algorithm.embeddings_test, algorithm.labels_test, algorithm.ages_test, algorithm.files_test)
      # create chunks of dataset from test set
      faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test = \
        algorithm.make_data_age_test_younger(algorithm.labels_test, algorithm.embeddings_test, dataframe, age_low=47, age_high=48)
    
    # Best estimator
    svm_emb_array = [svm_cv.best_estimator_ for svm_cv in svm_embedding_array]
    rf_emb_array = [rf_cv.best_estimator_ for rf_cv in rf_embedding_array]
    h_emb_array = [hist.best_estimator_ for hist in hist_embedding_array]
    
    # C values
    c_array = [svm_model.C for svm_model in svm_emb_array]
    split = [rf_model.min_samples_split for rf_model in rf_emb_array]
    depth = [hist_model.max_depth for hist_model in h_emb_array]
    leaf = [hist_model.min_samples_leaf for hist_model in h_emb_array]
    max_iter = [hist_model.max_iter for hist_model in h_emb_array]
    lr = [hist_model.learning_rate for hist_model in h_emb_array]
    criterion = [rf_model.criterion for rf_model in rf_emb_array]
    
    # start mlflow experiment
    with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
      mlflow.log_metric("score_embedding_average_test", np.mean(score_embedding_test))
      mlflow.log_metric("score_embedding_weighted_average_test", np.sum(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))
      mlflow.log_metric("standard_error_test", pd.DataFrame((np.array(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))).sem() * np.sqrt(len(score_embedding_test)))
      mlflow.log_metric("score_embedding_average_train", np.mean(score_embedding_train))
      mlflow.log_metric("score_embedding_weighted_average_train", np.sum(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))
      mlflow.log_metric("standard_error_train", pd.DataFrame((np.array(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))).sem() * np.sqrt(len(score_embedding_train)))
      try:
        mlflow.set_tags({"SVM_C_Values": np.round(c_array, 2)})
        mlflow.set_tags({"Hist_Depth": depth})
        mlflow.set_tags({"Hist_Max_Iter": max_iter})
        mlflow.set_tags({"Hist_Learning_Rate": lr})
        mlflow.set_tags({"Hist_Leaf": leaf})
        mlflow.set_tags({"RF_Criterion": criterion})
        mlflow.set_tags({"RF_Split": split})
      except Exception as e:
        print(e.args)
      
    # chunk data from test and train
        faces_chunk_array_test, face_classes_array_test = np.concatenate([faces_chunk_array_train, faces_chunk_array_test], axis=0), np.concatenate([face_classes_array_train, face_classes_array_test], axis=0)
    
    # test and evaluate using existing model
    accuracy, recall = algorithm.test_and_evaluate(voting_classifier_array, faces_chunk_array_test, face_classes_array_test, dataframe,
                                                   algorithm.embeddings_test, collect_for=args.collect_for)
    
    
    # metrics saved
    metrics = pd.DataFrame(dict(zip(list(accuracy.keys()) + list(recall.keys()), list(accuracy.values()) + list(recall.values()))),
                           index=list(accuracy.keys()) + list(recall.keys()))
    # save the metrics
    metrics.T.to_csv(args.drift_evaluate_metrics)
    
    pickle.dump(voting_classifier_array, open(args.classifier, 'wb'))

    # start mlflow experiment
    with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
      mlflow.log_metrics(accuracy)
      mlflow.log_metrics(recall)
  
