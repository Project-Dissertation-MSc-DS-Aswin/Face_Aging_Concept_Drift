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
args.classifier_test_younger = os.environ.get('classifier_test_younger', constants.AGEDB_FACE_CLASSIFIER_TEST_YOUNGER)
args.classifier_train_younger = os.environ.get('classifier_train_younger', constants.AGEDB_FACE_CLASSIFIER_TRAIN_YOUNGER)
args.collect_for = os.environ.get('collect_for', 'age_drifting')
args.drift_evaluate_metrics_test_younger = os.environ.get('drift_evaluate_metrics_test_younger', constants.AGEDB_DRIFT_EVALUATE_METRICS_TEST_YOUNGER)
args.drift_evaluate_metrics_train_younger = os.environ.get('drift_evaluate_metrics_train_younger', constants.AGEDB_DRIFT_EVALUATE_METRICS_TRAIN_YOUNGER)
args.experiment_id = os.environ.get("experiment_id", 1)
args.log_file_younger = os.environ.get("log_file_younger", "test_data_predictions_younger.csv")
args.log_file_older = os.environ.get("log_file_older", "test_data_predictions_older.csv")
args.log_file = os.environ.get("log_file", "test_data_predictions.csv")
args.drift_model_or_data = os.environ.get('drift_model_or_data', 'model_drift')
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
    names = dataset.metadata.groupby('name').count()
    names = names[names['age'] > 60]
    names = names.index.get_level_values(0)
    idx = [dataset.metadata['name'] == name for name in names]
    result_idx = [False]*len(dataset.metadata)
    for i in idx:
      result_idx = np.logical_or(result_idx, i)
      
    return dataset.metadata.loc[result_idx].reset_index()
  elif args.dataset == "cacd":
    np.random.seed(seed)
    names = dataset.metadata.groupby('name').count()
    names = names[names['age'] > 90]
    names = names.index.get_level_values(0)
    idx = [dataset.metadata['name'] == name for name in names]
    result_idx = [False]*len(dataset.metadata)
    for i in idx:
      result_idx = np.logical_or(result_idx, i)
    
    return dataset.metadata.loc[result_idx].reset_index()
  
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
    
    experiment = FaceNetWithClassifierExperiment(dataset, whylogs, model_loader)
    
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
    
    embeddings, files, ages, labels = \
      experiment.collect_data(args.data_collection_pkl, face_classification_iterator, model=args.model)
      
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
    
    algorithm = FaceNetWithClassifierPredictor(metadata=dataset.metadata, model_loader=model_loader)
    
    algorithm.make_train_test_split(embeddings, files, ages, labels)
    
    if args.collect_for == "classification":
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
      
      dataframe = algorithm.make_dataframe(algorithm.embeddings_test, algorithm.labels_test, algorithm.ages_test, algorithm.files_test)
      faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test = \
      algorithm.make_data(algorithm.labels_test, algorithm.embeddings_test, dataframe)
      
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
      
      # with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
      #   mlflow.log_metric("score_embedding_average_test", np.mean(score_embedding_test))
      #   mlflow.log_metric("score_embedding_weighted_average_test", np.sum(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))
      #   mlflow.log_metric("standard_error_test", pd.DataFrame((np.array(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))).sem() * np.sqrt(len(score_embedding_test)))
      #   mlflow.log_metric("score_embedding_average_train", np.mean(score_embedding_train))
      #   mlflow.log_metric("score_embedding_weighted_average_train", np.sum(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))
      #   mlflow.log_metric("standard_error_train", pd.DataFrame((np.array(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))).sem() * np.sqrt(len(score_embedding_train)))
      #   try:
      #     mlflow.set_tags({"SVM_C_Values": np.round(c_array, 2)})
      #     mlflow.set_tags({"Hist_Depth": depth})
      #     mlflow.set_tags({"Hist_Max_Iter": max_iter})
      #     mlflow.set_tags({"Hist_Learning_Rate": lr})
      #     mlflow.set_tags({"Hist_Leaf": leaf})
      #     mlflow.set_tags({"RF_Criterion": criterion})
      #     mlflow.set_tags({"RF_Split": split})
      #   except Exception as e:
      #     print(e.args)
          
      print("score_embedding_average_test", np.mean(score_embedding_test))
      print("score_embedding_weighted_average_test", np.sum(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))
      print("standard_error_test", pd.DataFrame((np.array(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))).sem() * np.sqrt(len(score_embedding_test)))
      print("score_embedding_average_train", np.mean(score_embedding_train))
      print("score_embedding_weighted_average_train", np.sum(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))
      print("standard_error_train", pd.DataFrame((np.array(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))).sem() * np.sqrt(len(score_embedding_train)))
      print({"SVM_C_Values": np.round(c_array, 2)})
      print({"Hist_Depth": depth})
      print({"Hist_Max_Iter": max_iter})
      print({"Hist_Learning_Rate": lr})
      print({"Hist_Leaf": leaf})
      print({"RF_Criterion": criterion})
      print({"RF_Split": split})
        
      faces_chunk_array_test, face_classes_array_test = np.concatenate([faces_chunk_array_train, faces_chunk_array_test]), np.array([face_classes_array_train, face_classes_array_test])
      
      accuracy, recall = algorithm.test_and_evaluate(voting_classifier_array, faces_chunk_array_test, face_classes_array_test, dataframe, 
                                                    algorithm.embeddings_test, collect_for=args.collect_for)
      
      metrics = pd.DataFrame(dict(zip(list(accuracy.keys()) + list(recall.keys()), list(accuracy.values()) + list(recall.values()))), 
                            index=list(accuracy.keys()) + list(recall.keys()))
      metrics.T.to_csv(args.drift_evaluate_metrics)
      
      pickle.dump(voting_classifier_array, open(args.classifier, 'wb'))

      with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
        mlflow.log_metrics(accuracy)
        mlflow.log_metrics(recall)
      
    elif args.collect_for == "age_drifting":
      scaler = StandardScaler()
      algorithm.embeddings_train = scaler.fit_transform(algorithm.embeddings_train)
      algorithm.embeddings_test = scaler.transform(algorithm.embeddings_test)
      dataframe = algorithm.make_dataframe(algorithm.embeddings_train, algorithm.labels_train, algorithm.ages_train, algorithm.files_train)
      faces_chunk_array_train_younger, face_classes_array_train_younger, faces_chunk_array_test_older, face_classes_array_test_older = \
        algorithm.make_data_age_train_younger(algorithm.labels_train, algorithm.embeddings_train, dataframe, age_low=47, age_high=48)
        
      faces_chunk_array_train_older, face_classes_array_train_older, faces_chunk_array_test_younger, face_classes_array_test_younger = \
        algorithm.make_data_age_test_younger(algorithm.labels_train, algorithm.embeddings_train, dataframe, age_low=47, age_high=48)
        
      with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
        score_embedding_test_younger, score_embedding_train_younger, face_classes_count_test_younger, face_classes_count_train_younger, (voting_classifier_array_younger, 
                                                    svm_embedding_array_younger, 
                                                    rf_embedding_array_younger, 
                                                    hist_embedding_array_younger, 
                                                    knn_embeding_array_younger) = algorithm.train_and_evaluate(
          faces_chunk_array_train_younger, face_classes_array_train_younger, faces_chunk_array_test_older, face_classes_array_test_older, 
          param_grid, param_grid2, param_grid3, no_of_classes=3, original_df=pd.DataFrame(columns=['test_labels', 'test_predictions']), log_file=args.log_file_younger
        )
        
        score_embedding_test_older, score_embedding_train_older, face_classes_count_test_older, face_classes_count_train_older, (voting_classifier_array_older, 
                                                    svm_embedding_array_older, 
                                                    rf_embedding_array_older, 
                                                    hist_embedding_array_older, 
                                                    knn_embeding_array_older) = algorithm.train_and_evaluate(
          faces_chunk_array_train_older, face_classes_array_train_older, faces_chunk_array_test_younger, face_classes_array_test_younger, 
          param_grid, param_grid2, param_grid3, no_of_classes=3, original_df=pd.DataFrame(columns=['test_labels', 'test_predictions']), log_file=args.log_file_older
        )
      
      with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
        svm_emb_array = [svm_cv.best_estimator_ for svm_cv in svm_embedding_array_younger]
        rf_emb_array = [rf_cv.best_estimator_ for rf_cv in rf_embedding_array_younger]
        h_emb_array = [hist.best_estimator_ for hist in hist_embedding_array_younger]
        
        c_array = [svm_model.C for svm_model in svm_emb_array]
        split = [rf_model.min_samples_split for rf_model in rf_emb_array]
        depth = [hist_model.max_depth for hist_model in h_emb_array]
        leaf = [hist_model.min_samples_leaf for hist_model in h_emb_array]
        max_iter = [hist_model.max_iter for hist_model in h_emb_array]
        lr = [hist_model.learning_rate for hist_model in h_emb_array]
        criterion = [rf_model.criterion for rf_model in rf_emb_array]
        
        print("score_embedding_test_younger", np.mean(score_embedding_test_younger))
        print("score_embedding_weighted_average_test_younger", np.sum(score_embedding_test_younger * np.array(face_classes_count_test_younger)) / np.sum(face_classes_count_test_younger))
        print("standard_error_test_younger", pd.DataFrame((np.array(score_embedding_test_younger * np.array(face_classes_count_test_younger)) / np.sum(face_classes_count_test_younger))).sem() * np.sqrt(len(score_embedding_test_younger)))
        print("score_embedding_average_train_younger", np.mean(score_embedding_train_younger))
        print("score_embedding_weighted_average_train_younger", np.sum(score_embedding_train_younger * np.array(face_classes_count_train_younger)) / np.sum(face_classes_count_train_younger))
        print("standard_error_train_younger", pd.DataFrame((np.array(score_embedding_train_younger * np.array(face_classes_count_train_younger)) / np.sum(face_classes_count_train_younger))).sem() * np.sqrt(len(score_embedding_train_younger)))
        print({"SVM_C_Values_younger": np.round(c_array, 2)})
        print({"Hist_Depth_younger": depth})
        print({"Hist_Max_Iter_younger": max_iter})
        print({"Hist_Learning_Rate_younger": lr})
        print({"Hist_Leaf_younger": leaf})
        print({"RF_Criterion_younger": criterion})
        print({"RF_Split_younger": split})
        
        svm_emb_array = [svm_cv.best_estimator_ for svm_cv in svm_embedding_array_older]
        rf_emb_array = [rf_cv.best_estimator_ for rf_cv in rf_embedding_array_older]
        h_emb_array = [hist.best_estimator_ for hist in hist_embedding_array_older]
        
        c_array = [svm_model.C for svm_model in svm_emb_array]
        split = [rf_model.min_samples_split for rf_model in rf_emb_array]
        depth = [hist_model.max_depth for hist_model in h_emb_array]
        leaf = [hist_model.min_samples_leaf for hist_model in h_emb_array]
        max_iter = [hist_model.max_iter for hist_model in h_emb_array]
        lr = [hist_model.learning_rate for hist_model in h_emb_array]
        criterion = [rf_model.criterion for rf_model in rf_emb_array]
        
        print("score_embedding_test_older", np.mean(score_embedding_test_older))
        print("score_embedding_weighted_average_test_older", np.sum(score_embedding_test_older * np.array(face_classes_count_test_older)) / np.sum(face_classes_count_test_older))
        print("standard_error_test_older", pd.DataFrame((np.array(score_embedding_test_older * np.array(face_classes_count_test_older)) / np.sum(face_classes_count_test_older))).sem() * np.sqrt(len(score_embedding_test_older)))
        print("score_embedding_average_train_older", np.mean(score_embedding_train_younger))
        print("score_embedding_weighted_average_train_older", np.sum(score_embedding_train_older * np.array(face_classes_count_train_older)) / np.sum(face_classes_count_train_older))
        print("standard_error_train_older", pd.DataFrame((np.array(score_embedding_train_older * np.array(face_classes_count_train_older)) / np.sum(face_classes_count_train_older))).sem() * np.sqrt(len(score_embedding_train_older)))
        print({"SVM_C_Values_older": np.round(c_array, 2)})
        print({"Hist_Depth_older": depth})
        print({"Hist_Max_Iter_older": max_iter})
        print({"Hist_Learning_Rate_older": lr})
        print({"Hist_Leaf_older": leaf})
        print({"RF_Criterion_older": criterion})
        print({"RF_Split_older": split})
          
        # mlflow.log_metric("score_embedding_average_test", np.mean(score_embedding_test))
        # mlflow.log_metric("score_embedding_weighted_average_test", np.sum(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))
        # mlflow.log_metric("standard_error_test", pd.DataFrame((np.array(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))).sem() * np.sqrt(len(score_embedding_test)))
        # mlflow.log_metric("score_embedding_average_train", np.mean(score_embedding_train))
        # mlflow.log_metric("score_embedding_weighted_average_train", np.sum(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))
        # mlflow.log_metric("standard_error_train", pd.DataFrame((np.array(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))).sem() * np.sqrt(len(score_embedding_train)))
        # try:
        #   mlflow.set_tags({"SVM_C_Values": np.round(c_array, 2)})
        #   mlflow.set_tags({"Hist_Depth": depth})
        #   mlflow.set_tags({"Hist_Max_Iter": max_iter})
        #   mlflow.set_tags({"Hist_Learning_Rate": lr})
        #   mlflow.set_tags({"Hist_Leaf": leaf})
        #   mlflow.set_tags({"RF_Criterion": criterion})
        #   mlflow.set_tags({"RF_Split": split})
        # except Exception as e:
        #   print(e.args)
          
        # test younger
        dataframe = algorithm.make_dataframe(algorithm.embeddings_test, algorithm.labels_test, algorithm.ages_test, algorithm.files_test)
        faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test = \
          algorithm.make_data_age_test_younger(algorithm.labels_test, algorithm.embeddings_test, dataframe, age_low=47, age_high=48)
        
        faces_chunk_array_test, face_classes_array_test = np.concatenate([faces_chunk_array_train, faces_chunk_array_test], axis=0), np.concatenate([face_classes_array_train, face_classes_array_test], axis=0)
        
        accuracy, recall = algorithm.test_and_evaluate(voting_classifier_array_younger, faces_chunk_array_test, face_classes_array_test, dataframe, 
                                                      algorithm.embeddings_test, collect_for=args.collect_for)
        
        
        metrics = pd.DataFrame(dict(zip(list(accuracy.keys()) + list(recall.keys()), list(accuracy.values()) + list(recall.values()))), 
                              index=list(accuracy.keys()) + list(recall.keys()))
        metrics.T.to_csv(args.drift_evaluate_metrics_test_younger)
        
        # train younger
        dataframe = algorithm.make_dataframe(algorithm.embeddings_test, algorithm.labels_test, algorithm.ages_test, algorithm.files_test)
        faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test = \
          algorithm.make_data_age_train_younger(algorithm.labels_test, algorithm.embeddings_test, dataframe, age_low=47, age_high=48)
        
        faces_chunk_array_test, face_classes_array_test = np.concatenate([faces_chunk_array_train, faces_chunk_array_test], axis=0), np.concatenate([face_classes_array_train, face_classes_array_test], axis=0)
        
        accuracy, recall = algorithm.test_and_evaluate(voting_classifier_array_older, faces_chunk_array_test, face_classes_array_test, dataframe, 
                                                      algorithm.embeddings_test, collect_for=args.collect_for)
        
        
        metrics = pd.DataFrame(dict(zip(list(accuracy.keys()) + list(recall.keys()), list(accuracy.values()) + list(recall.values()))), 
                              index=list(accuracy.keys()) + list(recall.keys()))
        metrics.T.to_csv(args.drift_evaluate_metrics_train_younger)
        
        pickle.dump(voting_classifier_array_younger, open(args.classifier_test_younger, 'wb'))
        pickle.dump(voting_classifier_array_older, open(args.classifier_train_younger, 'wb'))

        with mlflow.start_run(experiment_id=args.experiment_id, run_name='FaceNet with Classifier'):
          mlflow.log_metrics(accuracy)
          mlflow.log_metrics(recall)
  
