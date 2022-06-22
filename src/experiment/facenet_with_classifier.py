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

def collect_data(model_loader, train_iterator):
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
    self.dataset = dataset
    self.logger = logger
    self.model_loader = model_loader
    self.batchno = 0
    
  def set_dataset(self, dataset):
    self.dataset = dataset

  def set_logger(self, logger):
    self.logger = logger

  def set_model_loader(self, model_loader):
    self.model_loader = model_loader
    
  def collect_data(self, data_collection_pkl, face_classification_iterator):
    if os.path.isfile(data_collection_pkl):
      embeddings = pickle.load(data_collection_pkl)
    else:
      embeddings, files, ages, labels = collect_data(self.model_loader, face_classification_iterator)
      
    return tf.concat(embeddings, axis=0), files, ages, labels
  

class FaceNetWithClassifierPredictor:
  
  def __init__(self, metadata):
    self.metadata = metadata
    
  def train_and_fit(self, faces_chunk_array_train, face_classes_array_train, 
                    base_estimators, svm_embedding_array, 
                    rf_embedding_array, hist_embedding_array, knn_embeding_array, knn_emb, 
                    score_embedding, voting_classifier_array, face_classes_count, idx):
  
    face_classes = np.concatenate(face_classes_array_train[idx*3:idx*3+3])
    np.random.seed(100)
    idx = np.arange(len(face_classes))
    np.random.shuffle(idx)
    face_classes_train = face_classes[idx[:int(0.7*len(face_classes))]]
    faces_data = np.vstack(faces_chunk_array_train[idx*3:idx*3+3])
    faces_data_train = faces_data[idx[:int(0.7*len(face_classes))]]
    
    face_classes_test = face_classes[idx[int(0.7*len(face_classes)):]]
    faces_data_test = faces_data[idx[int(0.7*len(face_classes)):]]
    
    voting_classifier = VotingClassifier(estimators=base_estimators, voting='soft')
    
    voting_classifier.fit(faces_data_train, face_classes_train)
    
    svm_embedding_array.append(voting_classifier.named_estimators_.svm)
    rf_embedding_array.append(voting_classifier.named_estimators_.rf)
    hist_embedding_array.append(voting_classifier.named_estimators_.hist)
    knn_embeding_array.append(voting_classifier.named_estimators_.knn)
    
    score_embedding.append(accuracy_score(face_classes_test, voting_classifier.predict(faces_data_test)))
    voting_classifier_array.append(voting_classifier)
    face_classes_count += len(face_classes_test)
    
    return score_embedding, face_classes_count, (voting_classifier_array, 
                                                 svm_embedding_array, 
                                                 rf_embedding_array, 
                                                 hist_embedding_array, 
                                                 knn_embeding_array)
  
  def train_and_evaluate(self, faces_chunk_array_train, face_classes_array_train, 
                         param_grid, param_grid2, param_grid3, no_of_classes):
    score_embedding = []
    svm_embedding_array = []

    svm_embedding_array = []
    rf_embedding_array = []
    hist_embedding_array = []
    knn_embeding_array = []

    voting_classifier_array = []
    face_classes_count = []
    for idx in tqdm(range(len(face_classes_array_train)//no_of_classes)):
        svm_embedding, rf_emb, hist_emb, knn_emb = \
          base_estimators_voting_classifier_face_recognition(param_grid, param_grid2, param_grid3)
        
        base_estimators = (
          ('svm', svm_embedding), 
          ('rf', rf_emb), 
          ('knn', knn_emb), 
          ('hist', hist_emb)
        )
        
        score_embedding, face_classes_count, (voting_classifier_array, 
                                                 svm_embedding_array, 
                                                 rf_embedding_array, 
                                                 hist_embedding_array, 
                                                 knn_embeding_array) = \
        self.train_and_fit(faces_chunk_array_train, face_classes_array_train, 
                    base_estimators, svm_embedding_array, 
                    rf_embedding_array, hist_embedding_array, knn_embeding_array, knn_emb, 
                    score_embedding, voting_classifier_array, face_classes_count, idx)
        
    return score_embedding, face_classes_count, (voting_classifier_array, 
                                                 svm_embedding_array, 
                                                 rf_embedding_array, 
                                                 hist_embedding_array, 
                                                 knn_embeding_array)
    
  def make_train_test_split(self, embeddings, files, ages, labels, seed=1000):
    np.random.seed(1000)
    files_train = self.metadata.sample(int(0.9*len(embeddings)))['filename'].values
    files_test = [f for ii, f in enumerate(files) if f not in files_train.tolist()]
    index_train = [files.index(f) for ii, f in enumerate(files_train)]
    index_test = [files.index(f) for ii, f in enumerate(files_test)]
    
    embeddings_train = [np.expand_dims(embeddings[ii], 0) for ii in index_train]
    embeddings_test = [np.expand_dims(embeddings[ii], 0) for ii in index_test]
    
    embeddings_train = tf.concat(embeddings_train, axis=0)
    embeddings_test = tf.concat(embeddings_test, axis=0)
    
    self.embeddings_train - embeddings_train
    self.embeddings_test - embeddings_test
    
    labels_train = [labels[files.index(f)] for ii, f in enumerate(files_train)]
    labels_test = [labels[files.index(f)] for ii, f in enumerate(files_test)]
    
    ages_train = [ages[files.index(f)] for ii, f in enumerate(files_train)]
    ages_test = [ages[files.index(f)] for ii, f in enumerate(files_test)]
    
    self.ages_train = ages_train
    self.ages_test = ages_test
    
    self.labels_train - labels_train
    self.labels_test = labels_test
    
  # dataframe after splitting the dataset
  def make_dataframe(self, embeddings, labels, ages, files):
    return pd.DataFrame(dict(face_id=list(range(len(embeddings)), labels=labels, ages=ages, files=files)))
  
  def make_data(self, labels_train, embeddings_train, data):
    copy_classes = copy(labels_train)
    faces_chunk_train = []
    faces_chunk_array_train = []
    face_classes_train = []
    face_classes_array_train = []
    for name, counter_class in tqdm(dict(Counter(copy_classes)).items()):
        df = data[
            data['identity'] == name
        ]
        for idx, row in df.iterrows():
            faces_chunk_train.append(embeddings_train[row['face_id']])
            face_classes_train.append(name)
        face_classes_array_train.append(face_classes_train)
        faces_chunk_array_train.append(faces_chunk_train)
        faces_chunk_train = []
        face_classes_train = []
        
    return faces_chunk_array_train, face_classes_array_train
  
  def make_data_age(self, labels_train, embeddings_train, data, age_low, age_high):
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
            (data['identity'] == name) & (data['age'] >= age_low)
        ]
        df2 = data[
            (data['identity'] == name) & (data['age'] <= age_high)
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
    
    return face_classes_array_train_age, faces_chunk_array_train_age
