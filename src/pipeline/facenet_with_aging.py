import sys
from context import Constants, Args
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA
import whylogs
import mlflow
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.drift_synthesis_by_eigen_faces import DriftSynthesisByEigenFacesExperiment
from experiment.model_loader import KerasModelLoader
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from tqdm import tqdm
from copy import copy
import pickle
import logging
import sys
import re
import tensorflow as tf

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

args.dataset = os.environ.get('dataset', 'agedb')
args.model = os.environ.get('model', 'facenet_keras.h5')
args.data_dir = os.environ.get('data_dir', constants.AGEDB_DATADIR)
args.grouping_distance_type = os.environ.get('grouping_distance_type', constants.EIGEN_FACES_DISTANCES_GROUPING)
args.grouping_distance_cutoff_range = os.environ.get('grouping_distance_cutoff_range')
args.batch_size = os.environ.get('batch_size', 128)
args.preprocess_prewhiten = os.environ.get('preprocess_prewhiten', 1)
args.data_collection_pkl = os.environ.get('data_collection_pkl', constants.AGEDB_FACENET_INFERENCES)
args.pca_covariates_pkl = os.environ.get('pca_covariates_pkl', constants.AGEDB_PCA_COVARIATES)
args.metadata = os.environ.get('metadata', constants.AGEDB_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_with_aging')
args.no_of_samples = os.environ.get('no_of_samples', 2248)
args.no_of_pca_samples = os.environ.get('no_of_pca_samples', 2248)
args.colormode = os.environ.get('colormode', 'rgb')
args.log_images = os.environ.get('log_images', 's3')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.classifier = os.environ.get('classifier', constants.AGEDB_FACE_CLASSIFIER)
args.drift_synthesis_filename = os.environ.get('drift_synthesis_filename', constants.AGEDB_DRIFT_SYNTHESIS_EDA_CSV_FILENAME)

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
args.no_of_pca_samples = int(args.no_of_pca_samples)

def images_covariance(images_new, no_of_images):
    images_cov = np.cov(images_new.reshape(no_of_images, -1))
    return images_cov

def demean_images(images_bw, no_of_images):
    images_mean = np.mean(images_bw.reshape(no_of_images, -1), axis=1)
    images_new = (images_bw.reshape(no_of_images, -1) - images_mean.reshape(no_of_images, 1))

    return images_new

def collect_images(train_iterator):
    images_bw = []
    # Get input and output tensors
    for ii in tqdm(range(len(train_iterator))):
        (X, y) = train_iterator[ii]
        images_bw.append(X)

    return np.vstack(images_bw)

def pca_covariates(images_cov):
    pca = PCA(n_components=images_cov.shape[0])
    X_pca = pca.fit_transform(images_cov)
    return pca.components_.T, pca, X_pca

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

    # mlflow.set_tracking_uri(args.tracking_uri)

    model_loader = KerasModelLoader(whylogs, args.model, input_shape=(-1,160,160,3))
    model_loader.load_model()

    dataset, augmentation_generator = load_dataset(args, whylogs, (49,49), args.no_of_pca_samples, 'grayscale')
    experiment_dataset, augmentation_generator = load_dataset(args, whylogs, (160,160), args.no_of_samples, 'rgb')
    
    pca_args = copy(args)
    pca_args.no_of_samples = pca_args.no_of_pca_samples
    dataset.set_metadata(
        get_reduced_metadata(pca_args, dataset)
    )
    
    experiment_dataset.set_metadata(
        get_reduced_metadata(args, experiment_dataset)
    )

    images_bw = collect_images(dataset.iterator)
    images_new = demean_images(images_bw, len(dataset))
    if not os.path.isfile(args.pca_covariates_pkl):
        images_cov = images_covariance(images_new, len(images_new))
        P, pca, X_pca = pca_covariates(images_cov)
        
        pickle.dump(pca, open(args.pca_covariates_pkl, "wb"))
    else:
        pca = pickle.load(open(args.pca_covariates_pkl, "rb"))

    print(pca)
    experiment = DriftSynthesisByEigenFacesExperiment(dataset, experiment_dataset, logger=whylogs, model_loader=model_loader, pca=pca,
                                                      init_offset=0)

    P_pandas = pd.DataFrame(pca.components_.T, columns=list(range(pca.components_.T.shape[1])))
    index = experiment.dataset.metadata['age'].reset_index()

    images = collect_images(experiment_dataset.iterator)
    eigen_vectors = experiment.eigen_vectors()
    b_vector = experiment.eigen_vector_coefficient(eigen_vectors, images_new)
    weights_vector, offset, mean_age, std_age, age = experiment.weights_vector(experiment.dataset.metadata, b_vector)
    
    b_vector_new = tf.constant(np.expand_dims(b_vector, 2), dtype=tf.float64)
    error = tf.reduce_mean(age - offset - (tf.transpose(tf.matmul(tf.transpose(weights_vector, (0,2,1)), b_vector_new), (1, 0, 2)) / tf.norm(weights_vector, ord=2)))
    offset *= std_age
    offset += mean_age
    weights_vector *= std_age
    real_error = tf.reduce_mean(experiment.dataset.metadata['age'].values - offset - \
        (tf.transpose(tf.matmul(tf.transpose(weights_vector, (0,2,1)), b_vector_new), (1, 0, 2)) / tf.norm(weights_vector, ord=2)) * std_age)
    
    print("""
          Error: {error}, 
          Real Error: {real_error}
          """.format(error=error, real_error=real_error))
    
    choices_array = None
    offset_range = np.arange(2000, -2000, -500)
    if args.log_images == 's3':

        experiment.dataset.metadata['identity_grouping_distance'] = 0.0

        distances = experiment.mahalanobis_distance(b_vector)
        
        experiment.dataset.metadata['identity_grouping_distance'] = distances
        
        if args.grouping_distance_type == 'DISTINCT':
            experiment.dataset.metadata = experiment.set_hash_sample_by_distinct(experiment.dataset.metadata['identity_grouping_distance'])
        elif args.grouping_distance_type == 'CUTOFF':
            experiment.dataset.metadata = experiment.set_hash_sample_by_cutoff(
                np.unique(np.round(np.unique(experiment.dataset.metadata['identity_grouping_distance']), 1)), 
                                                 experiment.dataset.metadata)
        
        figures, choices_array = experiment.plot_images_with_eigen_faces(
            images, images_new, weights_vector, offset, b_vector, offset_range, P_pandas, index
        )
        
        hash_samples = np.unique(experiment.dataset.metadata['hash_sample'])
        
        experiment_name = "FaceNet with Aging Drift (modified)"
        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = None
        if experiment is not None:
            experiment_id = mlflow_experiment.experiment_id
        
        if experiment_id is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment_id):
            # hash_sample
            for ii, figure in enumerate(figures):
                # offset range
                for jj, fig in enumerate(figure):
                    fig1, fig2, fig3 = tuple(fig)
                    mlflow.log_figure(fig1, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, str(hash_samples[ii]), str(jj), 'aging'))
                    mlflow.log_figure(fig2, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, str(hash_samples[ii]), str(jj), 'actual'))
                    mlflow.log_figure(fig3, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, str(hash_samples[ii]), str(jj), 'predicted'))
                

    predictions_classes_array = experiment.collect_drift_predictions(images, images_new, 
                                        weights_vector, offset, b_vector, offset_range, P_pandas, index, 
                                        args.classifier, model_loader, choices_array=choices_array)
    
    predictions_classes = pd.DataFrame(predictions_classes_array, 
                                columns=['hash_sample', 'offset', 'true_identity', 'age', 'filename', 
                                'y_pred', 'y_drift', 'euclidean', 'cosine', 'identity_grouping_distance'])
    
    for value in predictions_classes_array:
        with mlflow.start_run():
            mlflow.log_param(dict(zip(predictions_classes.columns.values.tolist(), value)))
    
    predictions_classes = experiment.calculate_confusion_matrix_elements(predictions_classes)
    
    recall = predictions_classes['TP'].sum() / (predictions_classes['TP'].sum() + predictions_classes['FN'].sum())
    precision = predictions_classes['TP'].sum() / (predictions_classes['TP'].sum() + predictions_classes['FP'].sum())
    accuracy = (predictions_classes['TP'].sum() + predictions_classes['TN'].sum()) / \
    (predictions_classes['TP'].sum() + predictions_classes['TN'].sum() + predictions_classes['FP'].sum() + predictions_classes['FN'].sum())
    f1 = 2 * recall * precision / (recall + precision)
    print("""
        Recall: {recall}, 
        Precision: {precision}, 
        F1: {f1},  
        Accuracy: {accuracy}
    """.format(
        accuracy=accuracy, 
        f1=f1, 
        precision=precision, 
        recall=recall
    ))
    
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("accuracy", accuracy)
    
    predictions_classes.to_csv(args.drift_synthesis_filename)
        
    with mlflow.start_run():
        figure = experiment.plot_histogram_of_face_distances()
        mlflow.log_figure(figure, "histogram_of_face_distances.png")
        figure = experiment.plot_scatter_of_drift_confusion_matrix()
        mlflow.log_figure(figure, "scatter_plot_of_drift_true_positives_false_negatives.png")
        
    