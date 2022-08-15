import tensorflow as tf
import pandas as pd
import numpy as np
from context import Constants, Args
from experiment.drift_synthesis_by_eigen_faces import DriftSynthesisByEigenFacesExperiment
import whylogs
import pickle
import os
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

args.pca_covariates_pkl = os.environ.get('pca_covariates_pkl', constants.AGEDB_PCA_COVARIATES)
args.mode = os.environ.get('mode', 'image_reconstruction')
args.pca_type = os.environ.get('pca_type', 'KernelPCA')
args.log_images = os.environ.get('log_images', 's3')
args.grouping_distance_type = os.environ.get('grouping_distance_type', constants.EIGEN_FACES_DISTANCES_GROUPING)
args.classifier = os.environ.get('classifier', constants.AGEDB_FACE_CLASSIFIER)
args.drift_beta = os.environ.get('drift_beta', 1)
args.covariates_beta = os.environ.get('covariates_beta', 1)
args.drift_synthesis_filename = os.environ.get('drift_synthesis_filename', constants.AGEDB_DRIFT_SYNTHESIS_EDA_CSV_FILENAME)

model_loader = pickle.load(open("model_loader.pkl", "rb"))
pca = pickle.load(open(args.pca_covariates_pkl, "rb"))
images_new = pickle.load(open("images_new.pkl", "rb"))
dataset = pickle.load(open("dataset.pkl", "rb"))
experiment_dataset = pickle.load(open("experiment_dataset.pkl", "rb"))
images = pickle.load(open("images.pkl", "rb"))

voting_classifier_array = pickle.load(open(args.classifier, 'rb'))

experiment = DriftSynthesisByEigenFacesExperiment(args, dataset, experiment_dataset, logger=whylogs, model_loader=model_loader, pca=pca,
                                                      init_offset=0)

P_pandas = pd.DataFrame(pca.components_.T if args.pca_type == 'PCA' else pca.eigenvectors_, 
                        columns=list(range(pca.components_.T.shape[1] if args.pca_type == 'PCA' else pca.eigenvectors_.shape[1])))
index = experiment.dataset.metadata['age'].reset_index()

eigen_vectors = experiment.eigen_vectors()
b_vector = experiment.eigen_vector_coefficient(eigen_vectors, images_new)
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

    print("Taken image_reconstruction choice")

elif args.mode == 'image_perturbation':
    # weights vector dimensions
    weights_vector = experiment.weights_vector_perturbation(experiment.dataset.metadata, b_vector, init_offset=0)
    offset = 0

offset_range = np.arange(0.01, -0.01, -0.02)
if args.log_images == 's3':

    experiment.dataset.metadata['identity_grouping_distance'] = 0.0

    distances = experiment.mahalanobis_distance(b_vector)

    experiment.dataset.metadata['identity_grouping_distance'] = distances

    if args.grouping_distance_type == 'DISTINCT':
        experiment.dataset.metadata = experiment.set_hash_sample_by_distinct(experiment.dataset.metadata['identity_grouping_distance'])
    elif args.grouping_distance_type == 'DIST':
        experiment.dataset.metadata = experiment.set_hash_sample_by_dist(experiment.dataset.metadata['identity_grouping_distance'])


P_pandas = pd.DataFrame(pca.components_.T if args.pca_type == 'PCA' else pca.eigenvectors_, 
                            columns=list(range(pca.components_.T.shape[1] if args.pca_type == 'PCA' else pca.eigenvectors_.shape[1])))
index = experiment.dataset.metadata['age'].reset_index()
    
predictions_classes_array, _ = experiment.collect_drift_predictions(images, images_new, 
                                        weights_vector, offset, b_vector, offset_range, P_pandas, index, 
                                        voting_classifier_array, model_loader, drift_beta=args.drift_beta, covariates_beta=args.covariates_beta)

predictions_classes = pd.DataFrame(predictions_classes_array, 
                    columns=['hash_sample', 'offset', 'covariates_beta', 'drift_beta', 'true_identity', 'age', 'filename', 
                    'y_pred', 'proba_pred', 'y_drift', 'proba_drift', 'predicted_age', 'euclidean', 'cosine', 'identity_grouping_distance', 
                    'orig_TP', 'orig_FN', 'virtual_TP', 'virtual_FN', 'stat_TP', 'stat_FP', 'stat_undefined'])

predictions_classes.to_csv(args.drift_synthesis_filename)

# predictions_classes = pd.read_csv(args.drift_synthesis_filename)

recall = predictions_classes['orig_TP'].sum() / (predictions_classes['orig_FN'].sum() + predictions_classes['orig_TP'].sum())
precision = 1.0
accuracy = (predictions_classes['orig_TP'].sum()) / (predictions_classes['orig_TP'].sum() + predictions_classes['orig_FN'].sum())
f1 = 2 * recall * precision / (recall + precision)

recall_virtual = predictions_classes['virtual_TP'].sum() / (predictions_classes['virtual_FN'].sum() + predictions_classes['virtual_TP'].sum())
precision_virtual = 1.0
accuracy_virtual = (predictions_classes['virtual_TP'].sum()) / (predictions_classes['virtual_TP'].sum() + predictions_classes['virtual_FN'].sum())
f1_virtual = 2 * recall_virtual * precision_virtual / (recall_virtual + precision_virtual)

recall_drift = recall_score(predictions_classes['orig_TP'], predictions_classes['stat_FP'])
precision_drift = 1.0
accuracy_drift = accuracy_score(predictions_classes['orig_TP'], predictions_classes['stat_FP'])
f1_drift = 2 * recall_drift * precision_drift / (recall_drift + precision_drift)
roc_drift = roc_auc_score(predictions_classes['orig_TP'], predictions_classes['stat_FP'])

print("""
    Drift Source - Original Image
    -----------------------------
    Recall of prediction: {recall}, 
    Precision of prediction: {precision}, 
    F1 of prediction: {f1},  
    Accuracy of prediction: {accuracy}, 

    Drift Source - Reconstructed Image
    ----------------------------------
    Recall of prediction: {recall_virtual}, 
    Precision of prediction: {precision_virtual}, 
    F1 of prediction: {f1_virtual},  
    Accuracy of prediction: {accuracy_virtual}, 

    Statistical Drift Detected
    --------------------------
    Accuracy of Drift: {accuracy_drift}, 
    Recall of Drift: {recall_drift}, 
    Precision of Drift: {precision_drift}, 
    F1 of Drift: {f1_drift}, 
    ROC of Drift: {roc_drift}, 
""".format(
    accuracy=accuracy, 
    f1=f1, 
    precision=precision, 
    recall=recall, 
    accuracy_virtual=accuracy_virtual, 
    f1_virtual=f1_virtual, 
    precision_virtual=precision_virtual, 
    recall_virtual=recall_virtual, 
    accuracy_drift=accuracy_drift, 
    f1_drift=f1_drift, 
    precision_drift=precision_drift, 
    recall_drift=recall_drift, 
    roc_drift=roc_drift
))

experiment.dataset.metadata.to_csv("experiment_dataset_metadata.csv")