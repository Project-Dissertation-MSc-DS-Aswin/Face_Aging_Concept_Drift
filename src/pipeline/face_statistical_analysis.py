import sys
from context import Constants, Args
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA, KernelPCA
import whylogs
import mlflow
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.drift_synthesis_by_eigen_faces import DriftSynthesisByEigenFacesExperiment
from experiment.model_loader import FaceNetKerasModelLoader, FaceRecognitionBaselineKerasModelLoader
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
from tqdm import tqdm
from copy import copy
import pickle
import logging
import sys
import re
import tensorflow as tf
import imageio
from sklearn.metrics import accuracy_score, euclidean_distances, recall_score, roc_auc_score
from sklearn.manifold import Isomap
import scipy.stats

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
args.metadata = os.environ.get('metadata', constants.AGEDB_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_statistical_analysis')
args.no_of_samples = os.environ.get('no_of_samples', 2248)
args.colormode = os.environ.get('colormode', 'rgb')
args.log_images = os.environ.get('log_images', 's3')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.drift_synthesis_metrics = os.environ.get('drift_synthesis_metrics', constants.AGEDB_DRIFT_SYNTHESIS_METRICS)
args.drift_type = os.environ.get('drift_type', 'incremental')
args.classifier = os.environ.get('classifier', constants.AGEDB_FACE_CLASSIFIER)
args.psnr_error = os.environ.get('psnr_error', 0.2)
args.noise_error = os.environ.get('noise_error', 25)
args.drift_beta = os.environ.get('drift_beta', 0)
args.mode = os.environ.get('mode', 'image_reconstruction')
args.inference_images_pkl = os.environ.get('inference_images_pkl', constants.AGEDB_FACENET_INFERENCES)
args.function_type = os.environ.get('function_type', 'beta')
args.input_shape = os.environ.get('input_shape', (-1,160,160,3))
args.denoise_type = os.environ.get('denoise_type', 'gaussian')

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
args.noise_error = float(args.noise_error)
args.psnr_error = float(args.psnr_error)
args.drift_beta = float(args.drift_beta)
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])
    print(args.input_shape)

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

def pca_covariates(images_cov, pca_type='PCA'):
    pca = KernelPCA(n_components=images_cov.shape[0], kernel='poly') if pca_type == 'KernelPCA' else PCA(n_components=images_cov.shape[0])
    X_pca = pca.fit_transform(images_cov)
    return pca.components_.T if pca_type == 'PCA' else pca.eigenvectors_, pca, X_pca

def get_age_ranges(metadata, identity_key='name'):
    # Largest Age Range
    age_ranges = {}
    for identity in np.unique(metadata[identity_key]):
        age_range = np.unique(metadata.loc[metadata[identity_key] == identity, 'age']).astype(int).tolist()
        age_ranges[identity] = age_range
        
    full_age_range = []
    for identity in age_ranges.keys():
        full_age_range += age_ranges[identity]
    
    print(np.unique(full_age_range))
    
def load_dataset(args, whylogs, image_dim, no_of_samples, colormode):
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
  """
  Get reduced metadata
  @param args:
  @param dataset:
  @param seed:
  @return: tuple()
  """
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    np.random.seed(seed)
    if args.mode == 'image_reconstruction':
        filenames = pd.read_csv(args.drift_synthesis_metrics)
        idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
        result_idx = [False]*len(dataset.metadata)
        for i in idx:
            result_idx = np.logical_or(result_idx, i)
        
        return dataset.metadata.loc[result_idx].reset_index()
    elif args.mode == 'image_perturbation':
        np.random.seed(seed)
        return dataset.metadata.sample(args.no_of_samples).reset_index()
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()

def p_value_real_calculate_age_by_age(mean_list, std_list, sem_list, count_list, cols):
    """
    calculate P-value for age-by-age comparison
    @param mean_list:
    @param std_list:
    @param sem_list:
    @param count_list:
    @param cols:
    @return: np.ndarray
    """
    # splitting the findings into 2
    p_value = np.ones((len(cols),len(cols),len(mean_list)))
    for i in tqdm(range(len(cols))):
        for j in range(len(cols)):
            mean1 = np.array([mean_list[ii][cols[i]] for ii in range(len(mean_list))])
            mean2 = np.array([mean_list[ii][cols[j]] for ii in range(len(mean_list))])
            mean_diff = mean1 - mean2
            std1 = np.array([std_list[ii][cols[i]] for ii in range(len(mean_list))])
            std2 = np.array([std_list[ii][cols[j]] for ii in range(len(mean_list))])
            sem1 = np.array([sem_list[ii][cols[i]] for ii in range(len(mean_list))])
            sem2 = np.array([sem_list[ii][cols[j]] for ii in range(len(mean_list))])
            count1 = np.array([count_list[ii][cols[i]] for ii in range(len(mean_list))])
            count2 = np.array([count_list[ii][cols[j]] for ii in range(len(mean_list))])
            # we use general two population means test
            s_pooled = np.sqrt(((count1-1)*std1**2 + (count2-1)*std2**2) / (count1 + count2 - 2))
            se_fit = s_pooled * np.sqrt(1/count1 + 1/count2)
            t_value = pd.Series(mean_diff / se_fit)
            
            # correction to detecting the difference for a single item (consider it as duplicated item)
            # replace nan with 0 such that count1 = 1 and count2 = 1 is reflected
            t_value = t_value.replace(to_replace=np.nan, value=0)
            # replace inf with 0 such that count1 = 1 and count2 = 1 is reflected
            t_value = t_value.replace(to_replace=np.inf, value=0)
            # replace inf with 0 such that count1 = 1 and count2 = 1 is reflected
            t_value = t_value.replace(to_replace=-np.inf, value=0)
            
            # 2-tailed test because mean diff may be positive or negative
            cdf_array = []
            for ii, _t_value in enumerate(t_value):
                if _t_value < 0:
                    cdf1 = scipy.stats.t.cdf(_t_value, df=(count1[ii]+count2[ii]-2))
                    if np.isnan(cdf1):
                        cdf1 = scipy.stats.t.cdf(_t_value, df=1)*2
                else:
                    cdf1 = scipy.stats.t.cdf(-_t_value, df=(count1[ii] + count2[ii] - 2))
                    if np.isnan(cdf1):
                        cdf1 = scipy.stats.t.cdf(-_t_value, df=1)*2
                cdf_array.append(cdf1)
            p_value[i,j,:] = np.array(cdf_array)
            
    return p_value

def power_calculate_age_by_age(p_value, cols, sig=0.05):
    """
    calculate power for age-by-age comparison
    @param p_value:
    @param cols:
    @param sig:
    @return: tuple()
    """
    prob_test_significant = p_value.flatten()[p_value.flatten() < sig].shape[0] / p_value.flatten().shape[0] # assumption that power of test is 0.8
    power = np.zeros((len(cols),len(cols)))
    alpha = np.zeros((len(cols),len(cols)))
    for i in tqdm(range(len(cols))):
        for j in range(len(cols)):
            # let us assume probability(real_difference) = degree of overlap
            try:
                prob_real = p_value[i,j,:].flatten()[(p_value[i,j,:].flatten() < sig)].shape[0] / \
                p_value[i,j,:].flatten().shape[0]
            except Exception as e:
                prob_real = 0
            # 1 - p(real|test_sig)
            power[i,j] = prob_real * (prob_test_significant) / (prob_real * (prob_test_significant) + sig * (1-prob_real))
            alpha[i,j] = (1-prob_real) * (1-prob_test_significant) / ((1-prob_real) * (1-prob_test_significant) + sig * prob_real)
            
    return power, alpha

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

    # load the dataset for PCA experiment
    dataset, augmentation_generator = load_dataset(args, whylogs, (96,96), args.no_of_pca_samples, 'grayscale')
    # load the dataset for experiment
    experiment_dataset, augmentation_generator = load_dataset(args, whylogs, (args.input_shape[1], args.input_shape[2]), args.no_of_samples, 'rgb')
    
    # copy the dataset metadata
    metadata_copy = copy(dataset.metadata)
    
    # PCA args
    pca_args = copy(args)
    pca_args.no_of_samples = pca_args.no_of_pca_samples
    # set the matadata for PCA experiment
    dataset.set_metadata(
        get_reduced_metadata(pca_args, dataset)
    )
    
    # set the metadata for experiment
    experiment_dataset.set_metadata(
        get_reduced_metadata(args, experiment_dataset)
    )
    
    # collect b/w images
    images_bw = collect_images(dataset.iterator)
    if args.noise_error:
        np.random.seed(1000)
        print("Adding error in b/w images of " + str(args.noise_error))
        images_bw += np.random.normal(0, args.noise_error, size=(images_bw.shape))
    # demean the images
    images_new = demean_images(images_bw, len(dataset))
    # apply PCA on images
    if not os.path.isfile(args.pca_covariates_pkl):
        images_cov = images_covariance(images_new, len(images_new))
        P, pca, X_pca = pca_covariates(images_cov, args.pca_type)
        
        # pickle.dump(pca, open(args.pca_covariates_pkl, "wb"))
    else:
        pca = pickle.load(open(args.pca_covariates_pkl, "rb"))

    print(pca)
    # experiment create using image reconstruction
    experiment = DriftSynthesisByEigenFacesExperiment(args, dataset, experiment_dataset, logger=whylogs, model_loader=model_loader, pca=pca,
                                                      init_offset=0)

    # extract the Patch of data for computing the image
    P_pandas = pd.DataFrame(pca.components_.T if args.pca_type == 'PCA' else pca.eigenvectors_,
                            columns=list(range(pca.components_.T.shape[1] if args.pca_type == 'PCA' else pca.eigenvectors_.shape[1])))
    # apply index on the image
    index = experiment.dataset.metadata['age'].reset_index()

    # collect images using experiment dataset
    images = collect_images(experiment_dataset.iterator)
    if args.noise_error:
        np.random.seed(1000)
        print("Adding error of " + str(args.noise_error))
        images += np.random.normal(0, args.noise_error, size=(images.shape))
        
    # get the eigen vectors
    eigen_vectors = experiment.eigen_vectors()
    # extract the eigen vector coefficient
    b_vector = experiment.eigen_vector_coefficient(eigen_vectors, images_new)
    # image recoinstruction method
    if args.mode == 'image_reconstruction':
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
    offset = 2000

    if args.log_images == 's3':
        # set the identity grouping distance
        experiment.dataset.metadata['identity_grouping_distance'] = 0.0

        # extract mahalanobis distances
        distances = experiment.mahalanobis_distance(b_vector)
        
        # set the mahalanobis distances to metadata DataFrame
        experiment.dataset.metadata['identity_grouping_distance'] = distances
        
        # select the DISTINCT or DISTRIBUTION mode for assigning hash_sample
        if args.grouping_distance_type == 'DISTINCT':
            experiment.dataset.metadata = experiment.set_hash_sample_by_distinct(experiment.dataset.metadata['identity_grouping_distance'])
        elif args.grouping_distance_type == 'DIST':
            experiment.dataset.metadata = experiment.set_hash_sample_by_dist(experiment.dataset.metadata['identity_grouping_distance'])
            
    # get the voting classifier array
    voting_classifier_array = pickle.load(open(args.classifier, 'rb'))
    
    # incremental, gradual, sudden or reoccurring drift
    if args.drift_type == 'incremental':
        iter_list = np.arange(-1, args.drift_beta, 0.1 if args.drift_beta > 0 else -0.1)
    elif args.drift_type == 'gradual':
        iter_list = [args.drift_beta]*100
    elif args.drift_type == 'sudden':
        iter_list = [1]*100
    elif args.drift_type == 'recurring':
        iter_list = np.arange(0, args.drift_beta, 0.03)
        
    mean_list_virtual = []
    sem_list_virtual = []
    std_list_virtual = []
    count_list_virtual = []
    
    mean_list_orig = []
    sem_list_orig = []
    std_list_orig = []
    count_list_orig = []
    
    mse_p_array = []
    mse_t_array = []
    mse_corr_array = []
    
    psnr_pca_list = []
    
    metadata_ages = metadata_copy['age']
    metadata_filenames = metadata_copy['filename']
    
    statistical_drift_true_positives_list = [] 
    statistical_drift_true_negatives_list = []
    statistical_drift_undefined_list = []
    predictions_original_list = []
    predictions_virtual_list = []
    
    voting_classifier_array = pickle.load(open(args.classifier, 'rb'))
    
    # for psnr in np.arange(0.01, args.psnr_error, 0.1):
    
    # iterate through beta
    for beta in tqdm(iter_list):
    
        # collect drift statistics
        data = experiment.collect_drift_statistics(images, images_bw, images_new, weights_vector, offset,
                                    b_vector, offset, P_pandas, index, voting_classifier_array, 
                                    model_loader, psnr, args.drift_type, drift_beta=0.5)
        
        # get all inference images
        all_inference_images = pickle.load(open(args.inference_images_pkl, "rb"))
        
        # stack the inference images
        all_inference_images = np.vstack(all_inference_images)
        
        # extract the variables from tuple()
        (inference_images, inference_images_orig, filenames, ages_list, psnr_pca, mse_t_list, mse_p_list, mse_corr_list) = data
        
        # get pairwise euclidean distances for virtual
        euclidean_distances_virtual = euclidean_distances(all_inference_images, inference_images)
        # get pairwise euclidean distances for real
        euclidean_distances_orig = euclidean_distances(all_inference_images, inference_images_orig)

        # set pairwise euclidean distances into dataframe (virtual)
        data_table_virtual = pd.DataFrame(euclidean_distances_virtual, columns=ages_list)
        # set pairwise euclidean distances into dataframe (original)
        data_table_orig = pd.DataFrame(euclidean_distances_orig, columns=ages_list)
        
        mean = {}
        sem = {}
        count = {}
        std = {}

        # create sparse matrix consisting of age
        for age, age_data in data_table_virtual.iteritems():
            mean[int(age)] = age_data.loc[metadata_ages == int(age)].mean()
            std[int(age)] = age_data.loc[metadata_ages == int(age)].std()
            count[int(age)] = age_data.loc[metadata_ages == int(age)].count()
            sem[int(age)] = age_data.loc[metadata_ages == int(age)].sem()
            
        # set 0 for nan_std and nan_sem
        nan_std = [int(age) for age, std_value in list(std.items()) if np.isnan(std_value)]
        for age in nan_std:
            std[int(age)] = 0
            sem[int(age)] = 0
            
        mean_list_virtual += [mean]
        std_list_virtual += [std]
        count_list_virtual += [count]
        sem_list_virtual += [sem]
        
        mean = {}
        sem = {}
        count = {}
        std = {}

        # create sparse matrix for original images
        for age, age_data in data_table_orig.iteritems():
            mean[int(age)] = age_data.loc[metadata_ages == int(age)].mean()
            std[int(age)] = age_data.loc[metadata_ages == int(age)].std()
            count[int(age)] = age_data.loc[metadata_ages == int(age)].count()
            sem[int(age)] = age_data.loc[metadata_ages == int(age)].sem()
            
        # set nan for nan_std and nan_sem
        nan_std = [int(age) for age, std_value in list(std.items()) if np.isnan(std_value)]
        for age in nan_std:
            std[int(age)] = 0
            sem[int(age)] = 0
            
        mean_list_orig += [mean]
        std_list_orig += [std]
        count_list_orig += [count]
        sem_list_orig += [sem]
        
        mse_p_array.append(mse_p_list)
        mse_t_array.append(mse_t_list)
        mse_corr_array.append(mse_corr_list)
        psnr_pca_list.append(psnr_pca)
        
        # statistical_drift_true_positives_list.append(statistical_drift_true_positives)
        # statistical_drift_true_negatives_list.append(statistical_drift_true_negatives)
        # statistical_drift_undefined_list.append(statistical_drift_undefined)
        # predictions_original_list.append(predictions_original)
        # predictions_virtual_list.append(predictions_virtual)
        
    # collect p_value for original and virtual images
    p_value_pca_image_for_real_drift_age_by_age = p_value_real_calculate_age_by_age(mean_list_virtual, std_list_virtual, sem_list_virtual,
                                          count_list_virtual, ages_list)
    p_value_orig_image_for_real_drift_age_by_age = p_value_real_calculate_age_by_age(mean_list_orig, std_list_orig, sem_list_orig, 
                                          count_list_orig, ages_list)
    
    # collect power for virtual and original images
    power_pca_age_by_age, alpha_pca_age_by_age = \
        power_calculate_age_by_age(p_value_pca_image_for_real_drift_age_by_age, ages_list, sig=0.05)
    power_orig_age_by_age, alpha_orig_age_by_age = \
        power_calculate_age_by_age(p_value_orig_image_for_real_drift_age_by_age, ages_list, sig=0.05)
    
    # set the MSE values to dataframe
    mse_p_array_df = pd.DataFrame(mse_p_array, columns=filenames)
    mse_t_array_df = pd.DataFrame(mse_t_array, columns=filenames)
    mse_corr_array_df = pd.DataFrame(mse_corr_array, columns=filenames)
    psnr_pca_df = pd.DataFrame(psnr_pca_list, columns=filenames)

    # set the power to dataframe
    power_pca_df = pd.DataFrame(power_pca_age_by_age, columns=ages_list, index=ages_list)
    power_orig_df = pd.DataFrame(power_orig_age_by_age, columns=ages_list, index=ages_list)
    
    # statistical_drift_true_positives_df = pd.DataFrame(statistical_drift_true_positives_list, columns=filenames)
    # statistical_drift_true_negatives_df = pd.DataFrame(statistical_drift_true_negatives_list, columns=filenames)
    # statistical_drift_undefined_df = pd.DataFrame(statistical_drift_undefined_list, columns=filenames)
    # predictions_original_df = pd.DataFrame(predictions_original_list, columns=filenames)
    # predictions_virtual_df = pd.DataFrame(predictions_virtual_list, columns=filenames)

    # save mse and power to dataframe
    mse_p_array_df.to_csv("../data_collection/morph_facenet_mse_p_array_df_optimized.csv")
    mse_t_array_df.to_csv("../data_collection/morph_facenet_mse_t_array_df_optimized.csv")
    mse_corr_array_df.to_csv("../data_collection/morph_facenet_mse_corr_array_df_optimized.csv")
    psnr_pca_df.to_csv("../data_collection/morph_facenet_psnr_pca_df_optimized.csv")
    power_pca_df.to_csv("../data_collection/power_pca_df.csv")
    power_orig_df.to_csv("../data_collection/power_orig_df.csv")
    
    # statistical_drift_true_positives_df.to_csv("../data_collection/statistical_drift_true_positives_df.csv")
    # statistical_drift_true_negatives_df.to_csv("../data_collection/statistical_drift_true_negatives_df.csv")
    # statistical_drift_undefined_df.to_csv("../data_collection/statistical_drift_undefined_df.csv")
    # predictions_original_df.to_csv("../data_collection/predictions_original_df.csv")
    # predictions_virtual_df.to_csv("../data_collection/predictions_virtual_df.csv")

