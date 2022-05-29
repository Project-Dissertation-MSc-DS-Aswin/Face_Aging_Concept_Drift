from context import constants
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA
from logger import logger
import mlflow
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.drift_synthesis_by_eigen_faces import DriftSynthesisByEigenFacesExperiment
from experiment.model_loader import KerasModelLoader
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from tqdm import tqdm
from copy import copy

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='agedb',
                      help='The name of the dataset')
  parser.add_argument('--model', type=str, default='facenet_keras',
                      help='The model to load')
  parser.add_argument('--data_dir', type=str, default=constants.AGEDB_DATADIR,
                      help='The images data directory')
  parser.add_argument('--grouping_distance_type', type=str, default=constants.EIGEN_FACES_DISTANCES_GROUPING,
                      help='The grouping distance type - Allowable values: DISTINCT | CUTOFF_RANGE')
  parser.add_argument('--grouping_distance_cutoff_range', type=list, required=False,
                      help='The grouping distance CUTOFF_RANGE')
  parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                      help='The batch size for inference')
  parser.add_argument('--preprocess_prewhiten', type=int, default=1,
                      help='Check for preprocess (prewhiten) to be applied')
  parser.add_argument('--data_collection_pkl', type=str, default=constants.AGEDB_FACENET_INFERENCES,
                      help='Pickle object for data collection')
  parser.add_argument('--pca_covariates_pkl', type=str, default=constants.AGEDB_PCA_COVARIATES,
                      help='Pickle object for PCA model of eigen faces')
  parser.add_argument('--metadata', type=str, default=constants.AGEDB_METADATA,
                      help='Metadata mat file object that represents the metadata of images')
  parser.add_argument('--logger_name', type=str, default='facenet_with_aging',
                      help='The name of the logger')
  parser.add_argument('--no_of_samples', type=int, default=2248,
                      help='The number of samples')
  parser.add_argument('--colormode', type=str, default='color',
                      help='The type of colormode')
  parser.add_argument('--log_images', type=str, default='s3',
                      help='The source artifact to log images of visualization')
  parser.add_argument('--tracking_uri', type=str, default='http://localhost:5000',
                      help='The uri to track mlflow with s3 or any other artifact root')
  parser.add_argument('--classifier', type=str, default=constants.AGEDB_FACE_CLASSIFIER,
                      help='The uri to track mlflow with s3 or any other artifact root')
  parser.add_argument('--drift_synthesis_filename', type=str, default=constants.AGEDB_DRIFT_SYNTHESIS_EDA_CSV_FILENAME,
                      help='The filename to store with predictions classes')
  
  return parser.parse_args()

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
    for ii, (X, y) in tqdm(enumerate(train_iterator)):
        images_bw.append(X)

    return np.stack(images_bw)

def pca_covariates(images_cov):
    pca = PCA(n_components=images_cov.shape[0])
    X_pca = pca.fit_transform(images_cov)
    return pca.components_.T, pca, X_pca

def load_dataset(args, whylogs):
    dataset = None
    augmentation_generator = None
    if args.dataset == "agedb":
        augmentation_generator = get_augmented_datasets()
        dataset = AgeDBDataset(whylogs, args.metadata, list_IDs=list(range(args.no_of_samples)),
                               color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, dim=(160,160))
    elif args.dataset == "cacd":
        augmentation_generator = get_augmented_datasets()
        dataset = CACD2000Dataset(whylogs, args.metadata, list_IDs=list(range(args.no_of_samples)),
                                  color_mode='rgb', augmentation_generator=augmentation_generator,
                                  data_dir=args.data_dir, dim=(160,160))
    elif args.dataset == "fgnet":
        augmentation_generator = get_augmented_datasets()
        dataset = FGNETDataset(whylogs, args.metadata, list_IDs=None,
                               color_mode='rgb', augmentation_generator=augmentation_generator, data_dir=args.data_dir, dim=(160,160))

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

    args = parse_args()
    
    mlflow.set_tracking_uri(args.tracking_uri)

    whylogs = logger.setup_logger(args.logger_name)
    model_loader = KerasModelLoader(whylogs, args.model, input_shape=(-1,160,160,3))
    model_loader.load_model()

    dataset, augmentation_generator = load_dataset(args, whylogs)
    experiment_dataset = copy(dataset)
    dataset.set_metadata(
        get_reduced_metadata(args, dataset)
    )
    experiment_dataset.set_metadata(
        get_reduced_metadata(args, experiment_dataset, seed=2000)
    )

    if not os.path.isfile(args.pca_covariates_pkl):
        images_bw = collect_images(dataset.iterator)
        images_new = demean_images(images_bw, len(dataset.iterator))
        images_cov = images_covariance(images_new, len(images_new))
        P, pca, X_pca = pca_covariates(images_cov)

    experiment = DriftSynthesisByEigenFacesExperiment(experiment_dataset, logger=logger, model_loader=model_loader, pca=pca,
                                                      init_offset=11)

    P_pandas = pd.DataFrame(P, columns=list(range(P.shape[1])))
    index = experiment.dataset.metadata['year'].reset_index()

    images = collect_images(experiment_dataset.iterator)
    images_demean = demean_images(images, len(experiment_dataset.iterator))
    eigen_vectors = experiment.eigen_vectors()
    b_vector = experiment.eigen_vector_coefficient(eigen_vectors, images_demean)
    weights_vector = experiment.weights_vector(experiment.dataset.metadata, b_vector)

    choices_array = None
    offset_range = np.arange(2000, -2000, 500)
    if args.log_images == 's3':

        figures, choices_array = experiment.plot_images_with_eigen_faces(
            images, images_demean, np.mean(images_bw.reshape(len(experiment_dataset.iterator), -1), axis=1), 
            weights_vector, b_vector, offset_range, P_pandas, index
        )
        
        hash_samples = np.unique(experiment.dataset.metadata['hash_sample'])
        
        # hash_sample
        for ii, figure in enumerate(figures):
            # offset range
            for jj, (fig1, fig2, fig3) in enumerate(figure):
                with mlflow.start_run():
                    mlflow.log_figure(fig1, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, hash_samples[ii], str(jj), 'aging'))
                    mlflow.log_figure(fig2, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, hash_samples[ii], str(jj), 'actual'))
                    mlflow.log_figure(fig3, """{0}/hash_sample_{1}/offset_{2}_{3}.png""".format(args.logger_name, hash_samples[ii], str(jj), 'predicted'))
                
    predictions_classes_array = experiment.collect_drift_predictions(images, images_demean, np.mean(images_bw.reshape(len(experiment_dataset.iterator), -1), axis=1), 
                                        weights_vector, b_vector, offset_range, P_pandas, index, 
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
        
    