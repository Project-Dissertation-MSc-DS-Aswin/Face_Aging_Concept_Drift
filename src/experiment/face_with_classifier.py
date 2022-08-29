import pickle
import os
from preprocessing.facenet import l2_normalize, prewhiten
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from experiment.context import base_estimators_voting_classifier_face_recognition
from evaluation.distance import cosine, euclidean, face_distance
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
import pandas as pd
from collections import OrderedDict
from copy import copy
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_or_start_run

def collect_data_face_recognition_keras(model_loader, train_iterator):
    """
    Collect Face data for recognition (keras)
    @param model_loader: 
    @param train_iterator: 
    @return: 
    """
    res_images = []
    y_classes = []
    files = []
    ages = []
    labels = []
    # Get input and output tensors
    classes_counter = 0
    for i in tqdm(range(len(train_iterator)-1)):
        X, (y_age, y_filename, y_label) = train_iterator[i]
    # res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
        classes = y_label
        unq_classes = np.unique(classes)
        y_valid = np.zeros((len(y_label), 435))
    for c in unq_classes:
      y_valid[classes==c, np.random.randint(0, 435)] = 1
      classes_counter += 1
    res_images.append(model_loader.infer([X/255., y_valid]))
    labels += y_label.tolist()
    files += y_filename.tolist()
    ages += y_age.tolist()
    
    return res_images, files, ages, labels

def collect_data_facenet_keras(model_loader, train_iterator):
    """
    Collect data for facenet keras
    @param model_loader:
    @param train_iterator:
    @return:
    """
    res_images = []
    files = []
    ages = []
    labels = []
    # Get input and output tensors
    for i in tqdm(range(len(train_iterator))):
        X, (y_age, y_filename, y_label) = train_iterator[i]
        res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    labels += y_label.tolist()
    files += y_filename.tolist()
    ages += y_age.tolist()

    return res_images, files, ages, labels

class FaceNetWithClassifierExperiment:
  
  def __init__(self, dataset, logger=None, model_loader=None):
    """
    __init__ function for facenet with classifier experiment
    @param dataset:
    @param logger:
    @param model_loader:
    """
    self.dataset = dataset
    self.logger = logger
    self.model_loader = model_loader
    self.batchno = 0
    
  def set_dataset(self, dataset):
    """
    Set the dataset
    @param dataset:
    @return:
    """
    self.dataset = dataset

  def set_logger(self, logger):
    """
    Set the logger
    @param logger:
    @return:
    """
    self.logger = logger

  def set_model_loader(self, model_loader):
    """
    Set the model loader
    @param model_loader:
    @return:
    """
    self.model_loader = model_loader
    
  def collect_data(self, data_collection_pkl, face_classification_iterator, model=None):
    """
    Collect the data from FaceNetKeras or FaceRecognitionUsing CVAE
    @param data_collection_pkl:
    @param face_classification_iterator:
    @param model:
    @return:
    """
    if os.path.isfile(data_collection_pkl):
        embeddings = pickle.load(data_collection_pkl)
    elif model == 'FaceNetKeras':
        embeddings, files, ages, labels = collect_data_facenet_keras(self.model_loader, face_classification_iterator)
    elif model == 'FaceRecognitionBaselineKeras':
        embeddings, files, ages, labels = collect_data_face_recognition_keras(self.model_loader, face_classification_iterator)
      
    return tf.concat(embeddings, axis=0), files, ages, labels
  

class FaceNetWithClassifierPredictor:
  
  def __init__(self, metadata, model_loader):
    """
    __init__ function
    @param metadata:
    @param model_loader:
    """
    self.metadata = metadata
    self.model_loader = model_loader

  def train_and_fit(self, faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test, 
                    base_estimators, svm_embedding_array, 
                    rf_embedding_array, hist_embedding_array, knn_embeding_array, 
                    score_embedding_test, score_embedding_train, voting_classifier_array, face_classes_count_test, 
                    face_classes_count_train, iidx, no_of_classes, original_df, log_file=None):
  
    """
    Train and fit function
    @param faces_chunk_array_train:
    @param face_classes_array_train:
    @param faces_chunk_array_test:
    @param face_classes_array_test:
    @param base_estimators:
    @param svm_embedding_array:
    @param rf_embedding_array:
    @param hist_embedding_array:
    @param knn_embeding_array:
    @param score_embedding_test:
    @param score_embedding_train:
    @param voting_classifier_array:
    @param face_classes_count_test:
    @param face_classes_count_train:
    @param iidx:
    @param no_of_classes:
    @param original_df:
    @param log_file:
    @return:
    """
    face_classes_train = np.concatenate(face_classes_array_train[iidx*no_of_classes:iidx*no_of_classes+no_of_classes])
    face_classes_test = np.concatenate(face_classes_array_test[iidx*no_of_classes:iidx*no_of_classes+no_of_classes])
    np.random.seed(100)
    faces_data_train = np.vstack(faces_chunk_array_train[iidx*no_of_classes:iidx*no_of_classes+no_of_classes])
    
    faces_data_test = np.vstack(faces_chunk_array_test[iidx*no_of_classes:iidx*no_of_classes+no_of_classes])
    
    voting_classifier = VotingClassifier(estimators=base_estimators, voting='soft')
    
    voting_classifier.fit(faces_data_train, face_classes_train)
    
    svm_embedding_array.append(voting_classifier.named_estimators_.svm)
    rf_embedding_array.append(voting_classifier.named_estimators_.rf)
    hist_embedding_array.append(voting_classifier.named_estimators_.hist)
    knn_embeding_array.append(voting_classifier.named_estimators_.knn)
    
    test_predictions = voting_classifier.predict(faces_data_test)
    
    df = pd.DataFrame(columns=['test_labels', 'test_predictions'])
    df['test_labels'] = face_classes_test
    df['test_predictions'] = test_predictions
    
    original_df = pd.concat([original_df, df], axis=0)
    original_df.to_csv(os.path.join("../data_collection", log_file))
    
    score_embedding_test.append(accuracy_score(face_classes_test, test_predictions))
    score_embedding_train.append(accuracy_score(face_classes_train, voting_classifier.predict(faces_data_train)))
    voting_classifier_array.append(voting_classifier)
    face_classes_count_test += [len(face_classes_test)]
    face_classes_count_train += [len(face_classes_train)]
    
    return score_embedding_test, score_embedding_train, face_classes_count_test, face_classes_count_train, (voting_classifier_array, 
                                                 svm_embedding_array, 
                                                 rf_embedding_array, 
                                                 hist_embedding_array, 
                                                 knn_embeding_array), original_df
  
  def train_and_evaluate(self, faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test, 
                         param_grid, param_grid2, param_grid3, no_of_classes, original_df, log_file="test_data_predictions.csv"):
    """
    Function to train and evaluate
    @param faces_chunk_array_train:
    @param face_classes_array_train:
    @param faces_chunk_array_test:
    @param face_classes_array_test:
    @param param_grid:
    @param param_grid2:
    @param param_grid3:
    @param no_of_classes:
    @param original_df:
    @param log_file:
    @return:
    """
    score_embedding_test = []
    score_embedding_train = []
    svm_embedding_array = []

    svm_embedding_array = []
    rf_embedding_array = []
    hist_embedding_array = []
    knn_embeding_array = []

    voting_classifier_array = []
    face_classes_count_test = []
    face_classes_count_train = []
    for idx in tqdm(range(len(face_classes_array_train)//no_of_classes)):
        svm_embedding, rf_emb, hist_emb, knn_emb = \
          base_estimators_voting_classifier_face_recognition(param_grid, param_grid2, param_grid3)
        
        base_estimators = (
          ('svm', svm_embedding), 
          ('rf', rf_emb), 
          ('knn', knn_emb), 
          ('hist', hist_emb)
        )
        
        try:
          score_embedding_test, score_embedding_train, face_classes_count_test, face_classes_count_train, (voting_classifier_array, 
                                                  svm_embedding_array, 
                                                  rf_embedding_array, 
                                                  hist_embedding_array, 
                                                  knn_embeding_array), original_df = \
          self.train_and_fit(faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test, 
                      base_estimators, svm_embedding_array, 
                      rf_embedding_array, hist_embedding_array, knn_embeding_array, 
                      score_embedding_test, score_embedding_train, voting_classifier_array, face_classes_count_test, 
                      face_classes_count_train, idx, no_of_classes, original_df, log_file=log_file)
          
          run_id = _get_or_start_run().info.run_id
          MlflowClient().log_metric(run_id, "score_embedding_average_test_" + str(idx), np.mean(score_embedding_test))
          MlflowClient().log_metric(run_id, "score_embedding_weighted_average_test_" + str(idx), np.sum(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))
          MlflowClient().log_metric(run_id, "standard_error_test_" + str(idx), pd.DataFrame((np.array(score_embedding_test * np.array(face_classes_count_test)) / np.sum(face_classes_count_test))).sem() * np.sqrt(len(score_embedding_test)))
          MlflowClient().log_metric(run_id, "score_embedding_average_train_" + str(idx), np.mean(score_embedding_train))
          MlflowClient().log_metric(run_id, "score_embedding_weighted_average_train_" + str(idx), np.sum(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))
          MlflowClient().log_metric(run_id, "standard_error_train_" + str(idx), pd.DataFrame((np.array(score_embedding_train * np.array(face_classes_count_train)) / np.sum(face_classes_count_train))).sem() * np.sqrt(len(score_embedding_train)))
          
        except Exception as e:
          print(e.args)
        
    original_df.to_csv(os.path.join("../data_collection", log_file))
    
    return score_embedding_test, score_embedding_train, face_classes_count_test, face_classes_count_train, (voting_classifier_array, 
                                                 svm_embedding_array, 
                                                 rf_embedding_array, 
                                                 hist_embedding_array, 
                                                 knn_embeding_array)
    
  def test_and_evaluate(self, voting_classifier_array, faces_chunk_array_test, face_classes_array_test, data, embeddings_test, 
                        collect_for='age_drifting', 
                        age_low=47, age_high=48):
    """
    Function to test and evaluate
    @param voting_classifier_array:
    @param faces_chunk_array_test:
    @param face_classes_array_test:
    @param data:
    @param embeddings_test:
    @param collect_for:
    @param age_low:
    @param age_high:
    @return:
    """
    voting_classifier_array_copy = copy(voting_classifier_array)

    accuracy = {}
    recall = {}
    for i in tqdm(range(len(face_classes_array_test))):
        face_classes = np.concatenate(face_classes_array_test[i:i+1])
        faces_data = np.vstack(faces_chunk_array_test[i:i+1])
        
        if collect_for == 'classification':
          df = data[
              data['name'] == face_classes[0]
          ]
        if collect_for == 'age_drifting':
          df1 = data[
              (data['name'] == face_classes[0]) & (data['age'] <= age_low)
          ]
          df2 = data[
              (data['name'] == face_classes[0]) & (data['age'] >= age_high)
          ]
          df = pd.concat([df1, df2], axis=0)
          name = face_classes[0]
        
        if collect_for == 'classification':
          for k in range(len(face_classes)):
              true_positives = 0
              false_negatives = 0
              matches = 0
              for j in range(len(voting_classifier_array_copy)):
                  voting_classifier = voting_classifier_array_copy[j]
                  faces_classes_pred = voting_classifier.predict(faces_data[k].reshape(-1,self.model_loader.dimensions))
                  matches += (faces_classes_pred == face_classes[k]).sum()
              true_positives += 1 if matches == 1 else 0
              false_negatives += 0 if matches == 1 else 1
          accuracy[face_classes[0] + "_accuracy"] = (true_positives) / (false_negatives + true_positives)
          recall[face_classes[0] + "_recall"] = (true_positives) / (false_negatives + true_positives)
          
        elif collect_for == "age_drifting":
          for idx, row in df.iterrows():
            true_positives = 0
            false_negatives = 0
            matches = 0
            for j in range(len(voting_classifier_array_copy)):
                voting_classifier = voting_classifier_array_copy[j]
                faces_classes_pred = voting_classifier.predict(embeddings_test[row['face_id']].reshape(-1,self.model_loader.dimensions))
                matches += (faces_classes_pred == name).sum()
            true_positives += 1 if matches == 1 else 0
            false_negatives += 0 if matches == 1 else 1
            accuracy[row['files'] + "_accuracy"] = (true_positives) / (false_negatives + true_positives)
            recall[row['files'] + "_recall"] = (true_positives) / (false_negatives + true_positives)
        
    return accuracy, recall
  
  def make_train_test_split(self, embeddings, files, ages, labels, seed=1000):
    """
    Create train test split dataframe
    @param embeddings:
    @param files:
    @param ages:
    @param labels:
    @param seed:
    @return:
    """
    np.random.seed(1000)
    df = pd.DataFrame(columns=['files', 'ages', 'labels'])
    df['files'] = files
    df['ages'] = ages
    df['labels'] = labels
    files_train = df.sample(int(0.9*len(embeddings)))['files'].values
    files_test = [f for ii, f in enumerate(files) if f not in files_train.tolist()]
    index_train = [files.index(f) for ii, f in enumerate(files_train)]
    index_test = [files.index(f) for ii, f in enumerate(files_test)]

    embeddings_train = [np.expand_dims(embeddings[ii], 0) for ii in index_train]
    embeddings_test = [np.expand_dims(embeddings[ii], 0) for ii in index_test]

    embeddings_train = tf.concat(embeddings_train, axis=0)
    embeddings_test = tf.concat(embeddings_test, axis=0)

    self.embeddings_train = embeddings_train.numpy()
    self.embeddings_test = embeddings_test.numpy()

    labels_train = [labels[files.index(f)] for ii, f in enumerate(files_train)]
    labels_test = [labels[files.index(f)] for ii, f in enumerate(files_test)]

    ages_train = [ages[files.index(f)] for ii, f in enumerate(files_train)]
    ages_test = [ages[files.index(f)] for ii, f in enumerate(files_test)]

    self.files_train = files_train
    self.files_test = files_test

    self.ages_train = ages_train
    self.ages_test = ages_test

    self.labels_train = labels_train
    self.labels_test = labels_test
    
  # dataframe after splitting the dataset
  def make_dataframe(self, embeddings, labels, ages, files):
    """
    Create a DataFrame from labels, ages, files
    @param embeddings:
    @param labels:
    @param ages:
    @param files:
    @return:
    """
    return pd.DataFrame(dict(face_id=list(range(len(embeddings))), name=labels, age=ages, files=files))
  
  def make_data(self, labels_train, embeddings_train, data):
    """
    Create train and test data as chunks
    @param labels_train:
    @param embeddings_train:
    @param data:
    @return:
    """
    copy_classes = copy(labels_train)
    faces_chunk_train = []
    faces_chunk_array_train = []
    face_classes_train = []
    face_classes_array_train = []
    faces_chunk_test = []
    faces_chunk_array_test = []
    face_classes_test = []
    face_classes_array_test = []
    for name, counter_class in tqdm(dict(Counter(copy_classes)).items()):
        df = data[
            data['name'] == name
        ]
        np.random.seed(1000)
        df1 = df.sample(len(df)).iloc[:int(0.8*len(df))]
        np.random.seed(1000)
        df2 = df.sample(len(df)).iloc[int(0.8*len(df)):]
        
        for idx, row in df1.iterrows():
            faces_chunk_train.append(embeddings_train[row['face_id']])
            face_classes_train.append(name)
        face_classes_array_train.append(face_classes_train)
        faces_chunk_array_train.append(faces_chunk_train)
        faces_chunk_train = []
        face_classes_train = []
        
        for idx, row in df2.iterrows():
            faces_chunk_test.append(embeddings_train[row['face_id']])
            face_classes_test.append(name)
        face_classes_array_test.append(face_classes_train)
        faces_chunk_array_test.append(faces_chunk_train)
        faces_chunk_test = []
        face_classes_test = []
        
    return faces_chunk_array_train, face_classes_array_train, faces_chunk_array_test, face_classes_array_test
  
  def make_data_age_test_younger(self, labels_train, embeddings_train, data, age_low, age_high):
    """
    Create data for test_younger with age_low and age_high
    @param labels_train:
    @param embeddings_train:
    @param data:
    @param age_low:
    @param age_high:
    @return:
    """
    from copy import copy
    from collections import Counter
    
    copy_classes = copy(labels_train)
    faces_chunk_train_age = []
    faces_chunk_array_train_age = []
    face_classes_train_age = []
    face_classes_array_train_age = []
    faces_chunk_test_age = []
    faces_chunk_array_test_age = []
    face_classes_test_age = []
    face_classes_array_test_age = []
    
    for name, counter_class in tqdm(dict(Counter(copy_classes)).items()):
        df1 = data[
            (data['name'] == name) & (data['age'] <= age_low)
        ]
        df2 = data[
            (data['name'] == name) & (data['age'] >= age_high)
        ]
        if len(df1) == 0 or len(df2) == 0:
            continue
        for idx, row in df1.iterrows():
            faces_chunk_test_age.append(embeddings_train[row['face_id']])
            face_classes_test_age.append(name)
        face_classes_array_test_age.append(face_classes_test_age)
        faces_chunk_array_test_age.append(faces_chunk_test_age)
        faces_chunk_test_age = []
        face_classes_test_age = []
        
        for idx, row in df2.iterrows():
            faces_chunk_train_age.append(embeddings_train[row['face_id']])
            face_classes_train_age.append(name)
        face_classes_array_train_age.append(face_classes_train_age)
        faces_chunk_array_train_age.append(faces_chunk_train_age)
        faces_chunk_train_age = []
        face_classes_train_age = []
    
    return faces_chunk_array_train_age, face_classes_array_train_age, faces_chunk_array_test_age, face_classes_array_test_age

  def make_data_age_train_younger(self, labels_train, embeddings_train, data, age_low, age_high):
    """
    Create data for train_younger with age_low and age_high
    @param labels_train:
    @param embeddings_train:
    @param data:
    @param age_low:
    @param age_high:
    @return:
    """
    from copy import copy
    from collections import Counter
    
    copy_classes = copy(labels_train)
    faces_chunk_train_age = []
    faces_chunk_array_train_age = []
    face_classes_train_age = []
    face_classes_array_train_age = []
    faces_chunk_test_age = []
    faces_chunk_array_test_age = []
    face_classes_test_age = []
    face_classes_array_test_age = []
    
    for name, counter_class in tqdm(dict(Counter(copy_classes)).items()):
        df1 = data[
            (data['name'] == name) & (data['age'] <= age_low)
        ]
        df2 = data[
            (data['name'] == name) & (data['age'] >= age_high)
        ]
        if len(df1) == 0 or len(df2) == 0:
            continue
        for idx, row in df2.iterrows():
            faces_chunk_test_age.append(embeddings_train[row['face_id']])
            face_classes_test_age.append(name)
        face_classes_array_test_age.append(face_classes_test_age)
        faces_chunk_array_test_age.append(faces_chunk_test_age)
        faces_chunk_test_age = []
        face_classes_test_age = []
        
        for idx, row in df1.iterrows():
            faces_chunk_train_age.append(embeddings_train[row['face_id']])
            face_classes_train_age.append(name)
        face_classes_array_train_age.append(face_classes_train_age)
        faces_chunk_array_train_age.append(faces_chunk_train_age)
        faces_chunk_train_age = []
        face_classes_train_age = []
    
    return faces_chunk_array_train_age, face_classes_array_train_age, faces_chunk_array_test_age, face_classes_array_test_age