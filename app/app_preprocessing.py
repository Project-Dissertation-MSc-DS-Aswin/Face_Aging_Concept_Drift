import context
from datasets import CACD2000Dataset, AgeDBDataset, FGNETDataset
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging, KerasModelLoader, ArcFace, FaceRecognitionBaselineKerasModelLoader, FaceNetKerasModelLoader
import numpy as np
import whylogs
from tqdm import tqdm
from experiment.drift_synthesis_by_eigen_faces import DriftSynthesisByEigenFacesExperiment
from preprocessing.facenet import l2_normalize, prewhiten
from sklearn.decomposition import KernelPCA, PCA
from sklearn import model_selection
import os
import pandas as pd
import tensorflow as tf
import pickle
import drift
from collections import OrderedDict
import math

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

def get_reduced_metadata(args, dataset, filenames, seed=1000):
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    np.random.seed(seed)
    idx = [dataset.metadata['filename'] == filename for filename in filenames]
    result_idx = [False]*len(dataset.metadata)
    for i in idx:
      result_idx = np.logical_or(result_idx, i)
      
    return dataset.metadata.loc[result_idx].reset_index()
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()

def pca_covariates(images_cov, pca_type='PCA', seed=1000):
    pca = KernelPCA(n_components=images_cov.shape[0], kernel='poly') if pca_type == 'KernelPCA' else PCA(n_components=images_cov.shape[0])
    np.random.seed(seed)
    X_pca = pca.fit_transform(images_cov)
    return pca.components_.T if pca_type == 'PCA' else pca.eigenvectors_, pca, X_pca

def images_covariance(images_new, no_of_images):
    images_cov = np.cov(images_new.reshape(no_of_images, -1))
    return images_cov

def demean_images(images_bw, no_of_images):
    images_mean = np.mean(images_bw.reshape(no_of_images, -1), axis=1)
    images_new = (images_bw.reshape(no_of_images, -1) - images_mean.reshape(no_of_images, 1))

    return images_new

def collect_data_face_recognition_keras(model_loader, train_iterator):
  res_images = []
  y_classes = []
  files = []
  ages = []
  labels = []
  images = []
  # Get input and output tensors
  classes_counter = 0
  for i in tqdm(range(len(train_iterator)-1)):
    X, y_label = train_iterator[i]
    # res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    classes = y_label
    unq_classes = np.unique(classes)
    y_valid = np.zeros((len(y_label), 435))
    for c in unq_classes:
      y_valid[classes==c, classes_counter] = 1
      classes_counter += 1
    images.append(X)
    # res_images.append(model_loader.infer([X/255., y_valid]))
    labels += y_label.tolist()
    # files += y_filename.tolist()
    # ages += y_age.tolist()

  return images, labels

def collect_data_facenet_keras(model_loader, train_iterator):
  res_images = []
  files = []
  ages = []
  labels = []
  images = []
  # Get input and output tensors
  for i in tqdm(range(len(train_iterator))):
    X, y_label = train_iterator[i]
    images.append(X)
    # res_images.append(model_loader.infer(l2_normalize(prewhiten(X))))
    labels += y_label.tolist()
    # files += y_filename.tolist()
    # ages += y_age.tolist()
    
  return images, labels

class Preprocessor:
  
  def prepare_data(self, filenames):
    args = context.Args({})
    args.metadata = "../src/dataset_meta/AgeDB_metadata.mat"
    args.model_path = "../src/models/facenet_keras.h5"
    args.batch_size = 128
    args.dataset = "agedb"
    args.data_dir = "../../datasets/AgeDB"
    args.input_shape = (-1,160,160,3)
    args.alt_input_shape = (-1,96,96,3)
    args.model = "FaceNetKeras"
    args.denoise_type = 'opencv_denoising'
    args.drift_type = 'incremental'
    args.function_type = 'morph'
    args.drift_beta = 1
    
    dataset, augmentation_generator = load_dataset(args, whylogs, 16488, 'rgb', input_shape=args.input_shape)
    dataset.set_metadata(
      get_reduced_metadata(args, dataset, filenames)
    )
    self.dataset = dataset
    
    bw_dataset, augmentation_generator = load_dataset(args, whylogs, 16488, 'grayscale', input_shape=args.alt_input_shape)
    bw_dataset.set_metadata(
      get_reduced_metadata(args, bw_dataset, filenames)
    )
    
    self.bw_dataset = bw_dataset
    
    if args.model == 'FaceNetKeras':
      model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    elif args.model == 'FaceRecognitionBaselineKeras':
      model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
    
    model_loader.load_model()
    
    self.model_loader = model_loader
    
    if args.model == 'FaceRecognitionBaselineKeras':
      images, labels = collect_data_face_recognition_keras(model_loader, bw_dataset.iterator)
    elif args.model == 'FaceNetKeras':
      images, labels = collect_data_facenet_keras(model_loader, bw_dataset.iterator)
    
    X = np.vstack(images)
    y = labels
    
    return X, y

  def train_test_split(self, X, y):
    return model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

  def reconstruct_image(self, args, images_bw, voting_classifier_array, filenames):
    images_new = demean_images(images_bw, len(images_bw))
    # images_cov = images_covariance(images_new, len(images_new))
    # P, pca, X_pca = pca_covariates(images_cov, args.pca_type)
    
    # self.pca = pca
    
    full_bw_dataset, augmentation_generator = load_dataset(args, whylogs, 16488, 'grayscale', input_shape=args.alt_input_shape)
    filenames_new = pd.read_csv("../pipeline/Aging/input/facenet_agedb_drift_evaluate_difference.csv", index_col=0).filename.values
    full_bw_dataset.set_metadata(
      get_reduced_metadata(args, full_bw_dataset, filenames_new)
    )
    
    if args.model == 'FaceRecognitionBaselineKeras':
      images_full, labels = collect_data_face_recognition_keras(self.model_loader, full_bw_dataset.iterator)
    elif args.model == 'FaceNetKeras':
      images_full, labels = collect_data_facenet_keras(self.model_loader, full_bw_dataset.iterator)
    
    images_full = np.vstack(images_full)
    
    images_new_full = demean_images(images_full, len(images_full))
    images_cov = images_covariance(images_new_full, len(images_new_full))
    P, pca, X_pca = pca_covariates(images_cov, args.pca_type)
    
    self.pca = pca
    
    experiment = DriftSynthesisByEigenFacesExperiment(args, self.bw_dataset, self.dataset, logger=whylogs, model_loader=self.model_loader, pca=self.pca,
                                                        init_offset=0)
    
    # self.P_pandas = pd.DataFrame(self.pca.components_.T if args.pca_type == 'PCA' else self.pca.eigenvectors_, 
    #                         columns=list(range(self.pca.components_.T.shape[1] if args.pca_type == 'PCA' else self.pca.eigenvectors_.shape[1])))
    # self.index = experiment.dataset.metadata['age'].reset_index()
    
    if args.model == 'FaceRecognitionBaselineKeras':
      images, labels = collect_data_face_recognition_keras(self.model_loader, self.dataset.iterator)
    elif args.model == 'FaceNetKeras':
      images, labels = collect_data_facenet_keras(self.model_loader, self.dataset.iterator)
      
    images = np.vstack(images)
    
    self.P_pandas = pd.DataFrame(self.pca.components_.T if args.pca_type == 'PCA' else self.pca.eigenvectors_, 
                            columns=list(range(self.pca.components_.T.shape[1] if args.pca_type == 'PCA' else self.pca.eigenvectors_.shape[1])))
    self.index = full_bw_dataset.metadata['age'].reset_index()
    
    eigen_vectors = experiment.eigen_vectors()
    b_vector = experiment.eigen_vector_coefficient(eigen_vectors, images_new_full)
    if args.mode == 'image_reconstruction':
      weights_vector, offset, mean_age, std_age, age = experiment.weights_vector(full_bw_dataset.metadata, b_vector, factor=10, learning_rate=0.001)
      
      b_vector_new = tf.constant(np.expand_dims(b_vector, 2), dtype=tf.float64)
      error = tf.reduce_mean(age - offset - (tf.transpose(tf.matmul(tf.transpose(weights_vector, (0,2,1)), b_vector_new), (1, 0, 2)) / tf.norm(weights_vector, ord=2)))
      offset *= std_age
      offset += mean_age
      weights_vector *= std_age
      real_error = tf.reduce_mean(full_bw_dataset.metadata['age'].values - offset - \
          (tf.transpose(tf.matmul(tf.transpose(weights_vector, (0,2,1)), b_vector_new), (1, 0, 2)) / tf.norm(weights_vector, ord=2)) * std_age)
  
      print("""
        Error: {error}, 
        Real Error: {real_error}
        """.format(error=error, real_error=real_error))
      
      print("Taken image_reconstruction choice")
      
    offset_range = np.arange(0.01, -0.01, -0.02)
    
    full_bw_dataset.metadata['identity_grouping_distance'] = 0.0

    distances = experiment.mahalanobis_distance(b_vector)
    
    full_bw_dataset.metadata['identity_grouping_distance'] = distances
    
    if args.grouping_distance_type == 'DISTINCT':
      full_bw_dataset.metadata = experiment.set_hash_sample_by_distinct(full_bw_dataset.metadata['identity_grouping_distance'], metadata=full_bw_dataset.metadata)
    elif args.grouping_distance_type == 'DIST':
      full_bw_dataset.metadata = experiment.set_hash_sample_by_dist(full_bw_dataset.metadata['identity_grouping_distance'])

    hash_samples = []
    for ii, filename in enumerate(full_bw_dataset.metadata.filename.values):
      if filename in filenames:
        hash_samples.append(full_bw_dataset.metadata.loc[full_bw_dataset.metadata.filename == filename, 'hash_sample'].iloc[0])
    
    # self.P_pandas = pd.concat([self.P_pandas.loc[self.index.index.values[full_bw_dataset.metadata['hash_sample'] == hs],
    #       self.index.index.values[full_bw_dataset.metadata['hash_sample'] == hs]] for hs in hash_samples], axis=0)
    
    b_vector = np.concatenate([b_vector[(full_bw_dataset.metadata['hash_sample'] == hs).values] for hs in hash_samples], axis=0)
    weights_vector = np.concatenate([weights_vector[(full_bw_dataset.metadata['hash_sample'] == hs).values] for hs in hash_samples], axis=0)
    offset = np.concatenate([offset[(full_bw_dataset.metadata['hash_sample'] == hs).values] for hs in hash_samples], axis=0)
    idx = []
    for hs in hash_samples:
      idx.append(full_bw_dataset.metadata[(full_bw_dataset.metadata['hash_sample'] == hs)].index.values[0])
    full_bw_dataset.metadata = full_bw_dataset.metadata.loc[idx]
    self.P_pandas = self.P_pandas.loc[idx]
    self.index = self.index.loc[idx]
    experiment.dataset = full_bw_dataset
    predictions_classes_array, reconstructed_images, original_images = experiment.collect_drift_predictions(images, images_new, 
                                        weights_vector, offset, b_vector, offset_range, self.P_pandas, self.index, 
                                        voting_classifier_array, self.model_loader, drift_beta=args.drift_beta, hash_samples=hash_samples)
    
    self.experiment = experiment
    
    return predictions_classes_array, reconstructed_images, original_images

  def prepare_drift_features_deep_learning(self, X, y, ml_model_classification, filenames=None):
    args = context.Args({})
    args.pca_type = 'KernelPCA'
    args.grouping_distance_type = 'DISTINCT'
    args.mode = 'image_reconstruction'
    args.model = 'FaceNetKeras'
    args.drift_type = 'incremental'
    args.dataset = 'agedb'
    args.model_path = "../src/models/facenet_keras.h5"
    args.batch_size = 128
    args.data_dir = "../../datasets/AgeDB"
    args.input_shape = (-1,160,160,3)
    args.function_type = 'morph'
    args.alt_input_shape = (-1,96,96,3)
    args.metadata = "../src/dataset_meta/AgeDB_metadata.mat"
    args.drift_beta = 1
    args.denoise_type = 'opencv_denoising'
    
    predictions_classes_array, reconstructed_images, original_images = self.reconstruct_image(args, X, ml_model_classification, filenames)
    
    predictions_classes = pd.DataFrame(predictions_classes_array, 
                        columns=['hash_sample', 'offset', 'covariates_beta', 'drift_beta', 'true_identity', 'age', 'filename', 
                        'y_pred', 'proba_pred', 'y_drift', 'proba_drift', 'predicted_age', 'euclidean', 'cosine', 'identity_grouping_distance', 
                        'orig_TP', 'orig_FN', 'virtual_TP', 'virtual_FN', 'stat_TP', 'stat_FP', 'stat_undefined'])
    
    predictions_classes.to_csv("predictions_classes.csv")
    
    df, reconstructed_images, original_images = self.reduce(args, reconstructed_images, original_images, predictions_classes)
    
    drift_properties = drift.DriftProperties()
    drift_features = drift_properties.extract_statistical_properties(args, self.experiment, X, reconstructed_images, original_images)
    
    df = pd.concat([df, pd.DataFrame(np.concatenate(drift_features).reshape(4,-1).T, columns=['mse_p', 'mse_t', 'mse_corr', 'psnr'])], axis=1)
    
    return df.loc[:, ['mse_p', 'psnr', 'drift_beta', 'age', 'filename', 'drifted']], df['drifted'].values
  
  def reduce(self, args, reconstructed_images, original_images, predictions_classes):
    beta_identity = OrderedDict({})
    drift_min = OrderedDict({})
    euclidean_max = OrderedDict({})
    cosine_min = OrderedDict({})
    drift_bool = OrderedDict({})
    
    data = predictions_classes.sort_values(by=['hash_sample', 'true_identity', 'drift_beta'])

    for ii, (idx, row) in tqdm(enumerate(data.iterrows())):
      if (row['y_pred'] != row['y_drift']) and (row['true_identity'] + "_" + row['filename'] not in beta_identity):
        beta_identity[row['true_identity'] + "_" + row['filename']] = row['drift_beta']
        drift_min[row['true_identity'] + "_" + row['filename']] = row['proba_pred'] - row['proba_drift']
        euclidean_max[row['true_identity'] + "_" + row['filename']] = row['euclidean']
        cosine_min[row['true_identity'] + "_" + row['filename']] = row['cosine']
        drift_bool[row['true_identity'] + "_" + row['filename']] = int(row['true_identity'] != row['y_pred'])
    
    keys = list(beta_identity.keys())
    keys = list(map(lambda x: [x.split("_")[0], "_".join(x.split("_")[1:])], keys))
    
    df = pd.DataFrame(np.concatenate([np.array(keys), np.array(list(beta_identity.values())).reshape(-1,1), np.array(list(drift_min.values())).reshape(-1,1), 
                                      np.array(list(euclidean_max.values())).reshape(-1,1), 
                                      np.array(list(cosine_min.values())).reshape(-1,1), np.array(list(drift_bool.values())).reshape(-1,1)], axis=1), 
            columns=['identity', 'filename', 'drift_beta', 'drift_difference', 'euclidean', 'cosine', 'drifted'])
    
    df['age'] = df['filename'].apply(lambda x: x.split("_")[2])
    df['gender'] = df['filename'].apply(lambda x: x.split("_")[3])
    
    df['drift_beta'] = df['drift_beta'].astype(float)
    df['drift_difference'] = df['drift_difference'].astype(float)
    df['euclidean'] = df['euclidean'].astype(float)
    df['cosine'] = df['cosine'].astype(float)
    df['age'] = df['age'].astype(float)
    
    df['gender'] = df['gender'].apply(lambda x: x.replace(".jpg", ""))
    df['drifted'] = df['drifted'].astype(int)
    
    new_reconstructed_images = OrderedDict({})
    new_original_images = OrderedDict({})
    for idx, row in df.iterrows():
      key = row['filename'] + "/" + str(row['drift_beta'])
      if (key not in new_reconstructed_images) and (key in reconstructed_images):
        new_reconstructed_images[key] = reconstructed_images[key]
      if (key not in new_original_images) and (key in original_images):
        new_original_images[key] = original_images[key]
        
    return df, new_reconstructed_images, new_original_images

  def prepare_drift_features_statistical(self, X, y, ):
    pass

  def prepare_drift_features_classification(self, X, y, ml_model_classification=None, filenames=None):
    if ml_model_classification is not None:
      return self.prepare_drift_features_deep_learning(X, y, ml_model_classification, filenames)
      
    else:
      return self.prepare_drift_features_statistical(X, y)

