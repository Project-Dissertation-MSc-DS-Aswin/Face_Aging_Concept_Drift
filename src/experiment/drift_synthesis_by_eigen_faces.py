from re import M
import numpy as np
import pandas as pd
import random
import cv2
import tensorflow as tf
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from preprocessing.facenet import l2_normalize, prewhiten
from copy import copy
import pickle
import cvxpy as cvx
from concurrent.futures import ThreadPoolExecutor

# use Agg backend to suppress the plot shown in command line
matplotlib.use('Agg')

class DriftSynthesisByEigenFacesExperiment:

    def __init__(self, args, dataset, experiment_dataset, logger=None, model_loader=None, pca=None, init_offset=0):
        self.dataset = dataset
        self.args = args
        self.experiment_dataset = experiment_dataset
        self.logger = logger
        self.model_loader = model_loader
        self.pca = pca
        self.init_offset = init_offset
        self.no_of_images = self.pca.components_.shape[0] if self.args.pca_type == 'PCA' else self.pca.eigenvectors_.shape[0]
        self.batchno = 0

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_logger(self, logger):
        self.logger = logger

    def set_model_loader(self, model_loader):
        self.model_loader = model_loader

    def eigen_vectors(self):
        return self.pca.components_.T if self.args.pca_type == 'PCA' else self.pca.eigenvectors_

    # b_vector as a matrix, individual rows correspond to a single training example
    def eigen_vector_coefficient(self, eigen_vectors, images_demean):
        return np.matmul(np.linalg.inv(eigen_vectors), images_demean.reshape(len(images_demean), -1))

    def mahalanobis_distance(self, eigen_vector_coefficients):
        
        # each of the training example's parameters are considered as eigen vector coefficient
        # covariance of model parameters from the training examples
        def mahalanobis(bn, bm, C):
            if len(C.shape) == 0:
                return ((bn - bm)).dot((bn - bm).T) * C
            else:
                return (((bn - bm)).dot(np.linalg.inv(C))).dot((bn - bm).T)

        # shape is 1 x model_parameters
        b_vector = eigen_vector_coefficients
        bm = np.expand_dims(b_vector.mean(axis=1), 1)
        # shape is model parameters by model parameters
        C = np.cov(eigen_vector_coefficients.T)
        distance = mahalanobis(b_vector, bm, C)

        return np.diag(distance)

    def set_hash_sample_by_distinct(self, identity_grouping_distance):
        self.dataset.metadata['hash_sample'] = 0
        for ii, distance in enumerate(identity_grouping_distance):
            start_cutoff = distance - 1e-128
            end_cutoff = distance + 1e-128
            self.dataset.metadata.loc[(self.dataset.metadata['identity_grouping_distance'] <= end_cutoff) & \
                                 (self.dataset.metadata[
                                      'identity_grouping_distance'] >= start_cutoff), 'hash_sample'] = ii + 1

        return self.dataset.metadata
    
    def set_hash_sample_by_dist(self, identity_grouping_distance):
        self.dataset.metadata['hash_sample'] = 0
        hist = np.histogram(identity_grouping_distance, bins=self.args.bins)[:-1]
        for ii, distance in enumerate(hist):
            start_cutoff = distance
            end_cutoff = hist[ii+1]
            self.dataset.metadata.loc[(self.dataset.metadata['identity_grouping_distance'] <= end_cutoff) & \
                                 (self.dataset.metadata[
                                      'identity_grouping_distance'] >= start_cutoff), 'hash_sample'] = ii + 1

        return self.dataset.metadata

    # solve by lagrangian
    def weights_vector(self, metadata, b_vector, init_offset=None):

        b_vector = tf.constant(np.expand_dims(b_vector, 2), dtype=tf.float64)
        
        np.random.seed(1000)
        previous_weights = tf.Variable(10 * np.random.normal(size=(len(metadata), 2401, 1)), dtype=tf.float64)
        np.random.seed(1000)
        previous_offset = tf.Variable(np.random.normal(size=(len(metadata),1)), dtype=tf.float64)

        print(tf.norm(previous_weights, ord=1))
        print(tf.norm(previous_offset, ord=1))

        age = metadata['age'].values
        mean_age = np.mean(age)
        std_age = np.std(age)
        age = (age - np.mean(age)) / np.std(age)

        opt = tf.keras.optimizers.SGD(learning_rate=0.001)

        loss = lambda: tf.pow(tf.reduce_sum(tf.abs(age - previous_offset - (tf.transpose(tf.matmul(tf.transpose(previous_weights, (0,2,1)), b_vector), (1, 0, 2)) / tf.norm(previous_weights, ord=2)))), 1) + \
            tf.norm(previous_offset, ord=2)

        opt_op = opt.minimize(loss, var_list=[previous_weights, previous_offset])

        print(tf.norm(previous_weights, ord=1))
        print(tf.norm(previous_offset, ord=1))

        w = previous_weights.value()
        o = previous_offset.value()
        
        return w, o, mean_age, std_age, age
    
    def aging_function(self, weights_vector, b_vector, init_offset=None):
        if init_offset is None:
            init_offset = self.init_offset
        b_vector_new = tf.constant(np.expand_dims(b_vector, 2), dtype=tf.float64)
        result = (tf.transpose(tf.matmul(tf.transpose(weights_vector, (0,2,1)), b_vector_new), (1, 0, 2)) / tf.norm(weights_vector, ord=2)) + init_offset
        return result.numpy().flatten()
    
    def weights_vector_perturbation(self, reduced_metadata, b_vector, init_offset=0):
        if not init_offset:
            init_offset = self.init_offset
        w = (reduced_metadata['age'] - init_offset).dot(np.linalg.inv(b_vector))
        return w

    def aging_function_perturbation(self, weights_vector, b_vector, init_offset=0):
        if not init_offset:
            init_offset = self.init_offset
        return weights_vector.T.dot(b_vector) + init_offset
    
    def plot_images_with_eigen_faces(self, images, images_demean, weights_vector, offset_vector, b_vector, offset_range, P_pandas, index):
        figures = []
        choices_array = []
        for i in tqdm(np.unique(self.dataset.metadata['hash_sample'])):
            figure = []
            choices_list = []
            
            f_now = self.dataset.metadata['identity_grouping_distance'] * (self.aging_function(weights_vector, b_vector, offset_vector))
            f_p_now = f_now[self.dataset.metadata['hash_sample'] == i] / \
                      np.sum(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])
            for offset in offset_range:
                f_new = self.dataset.metadata['identity_grouping_distance'] * (self.aging_function(weights_vector, b_vector, offset))
                f_p_new = f_new[self.dataset.metadata['hash_sample'] == i] / \
                          np.sum(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])

                b_new = b_vector[self.dataset.metadata['hash_sample'] == i] + f_p_new.values.reshape(-1, 1) - f_p_now.values.reshape(-1, 1)

                P_pandas_1 = P_pandas.loc[index.index.values[self.dataset.metadata['hash_sample'] == i],
                                          index.index.values[self.dataset.metadata['hash_sample'] == i]]

                images_syn = images_demean.copy().reshape(len(self.dataset), -1)
                images_syn[self.dataset.metadata['hash_sample'] == i] = P_pandas_1.values.dot(b_new)

                new_images = \
                    (images_syn[self.dataset.metadata['hash_sample'] == i]) + \
                    images_demean.reshape(len(self.dataset), -1)[self.dataset.metadata['hash_sample'] == i]
                    
                choices = np.random.choice(list(range(len(new_images))), size=3)
                
                imgs = []
                ages = []
                fig1 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    img = new_images.reshape(-1, self.dataset.dim[0], self.dataset.dim[1])[choice]
                    plt.subplot(len(choices), 1, ii + 1)
                    plt.imshow(img, cmap='gray')
                    plt.title("Age Function")
                    imgs.append(img)

                fig2 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    plt.subplot(len(choices), 1, ii + 1)
                    plt.imshow(
                        cv2.cvtColor(images[self.dataset.metadata['hash_sample'] == i]\
                                .reshape(-1, self.experiment_dataset.dim[0], self.experiment_dataset.dim[1], 3)[choice], 
                                cv2.COLOR_RGB2GRAY), cmap='gray')
                    plt.title("Actual Age = " + \
                              self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'age'].iloc[choice].astype(str))
                    ages.append(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'filename'].iloc[choice])

                fig3 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    plt.subplot(len(choices), 1, ii + 1)
                    plt.imshow(
                        cv2.cvtColor(images\
                            [self.dataset.metadata['hash_sample'] == i]\
                                .reshape(-1, self.experiment_dataset.dim[0], self.experiment_dataset.dim[1], 3)[choice], 
                                cv2.COLOR_RGB2GRAY) + \
                        cv2.resize(new_images.reshape(-1, self.dataset.dim[0], self.dataset.dim[1])[choice], (160,160)), cmap='gray')
                    plt.title("Predicted Age = " + str(np.round(f_p_new.iloc[choice], 2)))

                figure.append([fig1, fig2, fig3, imgs, ages])
                choices_list.append(choices)

            figures.append(figure)
            choices_array.append(choices_list)
            
        plt.close()

        return figures, choices_array

    def collect_drift_predictions(self, images, images_demean, weights_vector, offset_vector, 
                                  b_vector, offset_range, P_pandas, index, 
                                  voting_classifier_array, model_loader, choices_array=None):
        
        predictions_classes_array = []
        
        def data_predictions(i, data):
            f_now = self.dataset.metadata['identity_grouping_distance'] * (
                self.aging_function(weights_vector, b_vector, offset_vector) if self.args.mode == 'image_reconstruction' else self.aging_function_perturbation(weights_vector, b_vector, 0)
            )
            f_p_now = f_now[self.dataset.metadata['hash_sample'] == i] / \
                    np.sum(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])
            for offset in offset_range:
                f_new = self.dataset.metadata['identity_grouping_distance'] * (
                    self.aging_function(weights_vector, b_vector, offset)  if self.args.mode == 'image_reconstruction' else self.aging_function_perturbation(weights_vector, b_vector, 0)
                )
                f_p_new = f_new[self.dataset.metadata['hash_sample'] == i] / \
                        np.sum(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])

                b_new = b_vector[self.dataset.metadata['hash_sample'] == i] + f_p_new.values.reshape(-1, 1) - f_p_now.values.reshape(-1, 1)

                P_pandas_1 = P_pandas.loc[index.index.values[self.dataset.metadata['hash_sample'] == i],
                                        index.index.values[self.dataset.metadata['hash_sample'] == i]]

                images_syn = images_demean.copy().reshape(len(self.dataset), -1)
                images_syn[self.dataset.metadata['hash_sample'] == i] = P_pandas_1.values.dot(b_new)

                new_images = \
                    (images_syn[self.dataset.metadata['hash_sample'] == i]) + \
                    images_demean.reshape(len(self.dataset), -1)[self.dataset.metadata['hash_sample'] == i]
                    
                choices = list(range(len(new_images)))
                choices = np.unique(choices)
                    
                for ii, choice in enumerate(choices):
                    image = new_images[choice].reshape(self.dataset.dim[0], self.dataset.dim[1])
                    image = np.concatenate([np.expand_dims(image, 2)]*3, 2)
                    orig_image = images[self.dataset.metadata['hash_sample'] == i].reshape(-1, self.experiment_dataset.dim[0], 
                                                                                        self.experiment_dataset.dim[1], 3)[choice]
                    image = model_loader.resize(image)
                    output_image = orig_image - image
                    output_image = np.concatenate([np.expand_dims(output_image, 2)]*3, 2)
                    # output_image = model_loader.resize(output_image)
                    # orig_image = model_loader.resize(orig_image)
                    res1 = model_loader.infer(l2_normalize(prewhiten(output_image)).reshape(*model_loader.input_shape))
                    res2 = model_loader.infer(l2_normalize(prewhiten(orig_image)).reshape(*model_loader.input_shape))
                    
                    if len(voting_classifier_array) > 0:
                        matches = np.zeros(len(voting_classifier_array))
                        virtual_matches = np.zeros(len(voting_classifier_array))
                        pred_original = {}
                        pred_drifted = {}
                        for ij, voting_classifier in enumerate(voting_classifier_array):
                            # reconstructed image
                            pred_virtual = voting_classifier.predict(
                                res1
                            )[0]
                            
                            # original image
                            pred_orig = voting_classifier.predict(
                                res2
                            )[0]
                            
                            pred_original[ij] = pred_orig
                            pred_drifted[ij] = pred_virtual
                            
                            matches[ij] += int(pred_orig == self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'name'].iloc[choice])
                            virtual_matches[ij] += int(pred_virtual == self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'name'].iloc[choice])
                        
                        orig_true_positives = 1 if sum(matches) == 1 else 0
                        orig_false_negatives = 0 if sum(matches) == 1 else 1
                        
                        virtual_true_positives = 1 if sum(virtual_matches) == 1 else 0
                        virtual_false_negatives = 0 if sum(virtual_matches) == 1 else 1
                        
                        statistical_drift_true_positives = 0
                        statistical_drift_true_negatives = 0
                        statistical_drift_undefined = 0
                        
                        idx = np.where(matches == 1)
                        idx = idx[0][0] if len(idx[0]) > 0 else None
                        if (idx is not None) and (pred_original[idx] == pred_drifted[idx]):
                            statistical_drift_true_positives = 1
                            statistical_drift_true_negatives = 0
                        elif idx is not None:
                            statistical_drift_true_positives = 0
                            statistical_drift_true_negatives = 1
                        else:
                            statistical_drift_undefined = 1
                            
                        pred_virtual = pred_drifted[idx] if idx is not None else -1
                        pred_orig = pred_original[idx] if idx is not None else -1
                    
                    else:
                        pred_virtual = -1
                        pred_orig = -1
                    
                    data.append([i, offset, 
                        # identity
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'name'].iloc[choice], 
                        # age
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'age'].iloc[choice], 
                        # filename
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'filename'].iloc[choice], 
                        # predicting original b/w image
                        pred_orig, 
                        # predicting noised image
                        pred_virtual, 
                        np.round(f_p_new.iloc[choice], 2),
                        # euclidean distance
                        tf.norm(res1 - res2, ord=2).numpy(), 
                        # cosine distance
                        (tf.matmul(res1, tf.transpose(res2)) / (tf.norm(res1, ord=2) * tf.norm(res2, ord=2))).numpy()[0][0],
                        # identity grouping distance
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'].iloc[choice], 
                        orig_true_positives, 
                        orig_false_negatives, 
                        virtual_true_positives, 
                        virtual_false_negatives, 
                        statistical_drift_true_positives, 
                        statistical_drift_true_negatives, 
                        statistical_drift_undefined
                    ])
                    
            return data
        
        print(np.unique(self.dataset.metadata['hash_sample']))
        
        data = []
        for ii in tqdm(np.unique(self.dataset.metadata['hash_sample'])):
            data = data_predictions(int(ii), data)
            
        return data
    
    def plot_histogram_of_face_distances(self, predictions_classes):
        
        fig = plt.figure(figsize=(12,8))
        plt.hist(predictions_classes['euclidean'], color='blue')
        plt.hist(predictions_classes['cosine'], color='orange')
        plt.xlabel("Distance (Cosine Similarity / Euclidean)")
        plt.ylabel("Count")
        
        return fig
    
    def plot_scatter_of_drift_confusion_matrix(self, predictions_classes):
        
        fig = plt.figure(figsize=(12,8))
        plt.scatter(predictions_classes.loc[(predictions_classes['TP'] != 1) & \
                                    (predictions_classes['FN'] != 1), 'euclidean'], 
            predictions_classes.loc[(predictions_classes['FN'] != 1) & \
                  (predictions_classes['TP'] != 1), 'cosine'], c='orange', label='Other')
        plt.scatter(predictions_classes.loc[predictions_classes['FN'] == 1, 'euclidean'], predictions_classes.loc[predictions_classes['FN'] == 1, 'cosine'], c='red', label='False Negative')
        plt.scatter(predictions_classes.loc[predictions_classes['TP'] == 1, 'euclidean'], predictions_classes.loc[predictions_classes['TP'] == 1, 'cosine'], c='blue', label='True Positive')
        plt.legend()
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Cosine Similarity")
        
        return fig