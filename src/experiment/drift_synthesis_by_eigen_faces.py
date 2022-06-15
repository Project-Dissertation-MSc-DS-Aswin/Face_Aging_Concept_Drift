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
import cvxpy as cvx

# use Agg backend to suppress the plot shown in command line
matplotlib.use('Agg')

class DriftSynthesisByEigenFacesExperiment:

    def __init__(self, dataset, experiment_dataset, logger=None, model_loader=None, pca=None, init_offset=0):
        self.dataset = dataset
        self.experiment_dataset = experiment_dataset
        self.logger = logger
        self.model_loader = model_loader
        self.pca = pca
        self.init_offset = init_offset
        self.no_of_images = self.pca.components_.shape[0]
        self.batchno = 0

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_logger(self, logger):
        self.logger = logger

    def set_model_loader(self, model_loader):
        self.model_loader = model_loader

    def eigen_vectors(self):
        return self.pca.components_.T

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

    """
    The cutoff_range is viewed first and then a new cutoff range is decided
    
    """
    def set_hash_sample_by_cutoff(self, cutoff_range, reduced_metadata):
        sample_lengths = []

        for ii in range(len(cutoff_range) - 1):
            start_cutoff = cutoff_range[ii]
            end_cutoff = cutoff_range[ii + 1]
            sample_length = self.dataset.metadata.loc[(self.dataset.metadata['identity_grouping_distance'] < end_cutoff) & \
                                                 (self.dataset.metadata[
                                                      'identity_grouping_distance'] >= start_cutoff)].shape[0]
            sample_lengths.append(sample_length)

        sample_lengths.append(
            reduced_metadata.loc[(reduced_metadata['identity_grouping_distance'] >= end_cutoff)].shape[0])

        length = sample_lengths[0]
        l = length
        new_sample_lengths = []
        new_cutoff_range = []
        for ii in range(len(cutoff_range) - 1):
            if l >= length:
                new_sample_lengths.append(l)
                new_cutoff_range.append(cutoff_range[ii])
                new_cutoff_range.append(cutoff_range[ii + 1])
                l = 0
            else:
                l += sample_lengths[ii]
        new_sample_lengths.append(sample_lengths[-1])
        new_cutoff_range.append(cutoff_range[-2])
        new_cutoff_range.append(cutoff_range[-1])

        s = 0
        for ii in range(len(new_cutoff_range) - 1):
            start_cutoff = new_cutoff_range[ii]
            end_cutoff = new_cutoff_range[ii + 1]
            self.dataset.metadata.loc[(self.dataset.metadata['identity_grouping_distance'] < end_cutoff) & \
                                 (self.dataset.metadata[
                                      'identity_grouping_distance'] >= start_cutoff), 'hash_sample'] = ii + 1
        self.dataset.metadata.loc[
            (self.dataset.metadata['identity_grouping_distance'] >= new_cutoff_range[-1]), 'hash_sample'] = len(new_cutoff_range)

        return self.dataset.metadata

    # solve by lagrangian
    def weights_vector(self, metadata, b_vector, init_offset=None):

        b_vector = tf.constant(np.expand_dims(b_vector, 2), dtype=tf.float64)
        
        np.random.seed(1000)
        previous_weights = tf.Variable(np.random.normal(size=(2248, 2401, 1)), dtype=tf.float64)
        np.random.seed(1000)
        previous_offset = tf.Variable(np.random.normal(size=(2248,1)), dtype=tf.float64)

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

    def plot_images_with_eigen_faces(self, images, images_demean, weights_vector, offset_vector, b_vector, offset_range, P_pandas, index):
        figures = []
        choices_array = []
        for i in tqdm(np.unique(self.dataset.metadata['hash_sample'])[:5]):
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
                choices = np.unique(choices)
                
                fig1 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    plt.subplot(1, 3, ii + 1)
                    plt.imshow(new_images.reshape(-1, self.dataset.dim[0], self.dataset.dim[1])[choice], cmap='gray')
                    plt.title("Age Function")

                fig2 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    plt.subplot(1, 3, ii + 1)
                    plt.imshow(
                        cv2.cvtColor(images[self.dataset.metadata['hash_sample'] == i]\
                                .reshape(-1, self.experiment_dataset.dim[0], self.experiment_dataset.dim[1], 3)[choice], 
                                cv2.COLOR_RGB2GRAY), cmap='gray')
                    plt.title("Actual Age = " + \
                              self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'age'].iloc[choice].astype(str))

                fig3 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    plt.subplot(1, 3, ii + 1)
                    plt.imshow(
                        cv2.cvtColor(images\
                            [self.dataset.metadata['hash_sample'] == i]\
                                .reshape(-1, self.experiment_dataset.dim[0], self.experiment_dataset.dim[1], 3)[choice], 
                                cv2.COLOR_RGB2GRAY) + \
                        cv2.resize(new_images.reshape(-1, self.dataset.dim[0], self.dataset.dim[1])[choice], (160,160)), cmap='gray')
                    plt.title("Predicted Age = " + str(np.round(f_p_new.iloc[choice], 2)))

                figure.append([fig1, fig2, fig3])
                choices_list.append(choices)

            figures.append(figure)
            choices_array.append(choices_list)
            
        plt.close()

        return figures, choices_array

    def collect_drift_predictions(self, images, images_demean, weights_vector, offset_vector, 
                                  b_vector, offset_range, P_pandas, index, 
                                  classifier, model_loader, choices_array=None):
        
        predictions_classes_array = []
        
        for i in tqdm(np.unique(self.dataset.metadata['hash_sample'])[:5]):
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
                choices = np.unique(choices)
                    
                for ii, choice in enumerate(choices):
                    image = new_images[choice].reshape(self.dataset.dim[0], self.dataset.dim[1])
                    image = np.concatenate([np.expand_dims(image, 2)]*3, 2)
                    orig_image = images[self.dataset.metadata['hash_sample'] == i].reshape(-1, self.dataset.dim[0], self.dataset.dim[1], 3)[choice]
                    output_image = orig_image + image
                    output_image = np.concatenate([np.expand_dims(output_image, 2)]*3, 2)
                    output_image = model_loader.resize(output_image)
                    orig_image = model_loader.resize(orig_image)
                    image = model_loader.resize(image)
                    res1 = model_loader.infer(l2_normalize(prewhiten(output_image)).reshape(*model_loader.input_shape))
                    res2 = model_loader.infer(l2_normalize(prewhiten(orig_image)).reshape(*model_loader.input_shape))
                    
                    if classifier:
                        pred1 = classifier.predict(
                            res1.numpy()
                        )[0]
                        
                        pred2 = classifier.predict(
                            res2.numpy()
                        )[0]
                    else:
                        pred1 = -1
                        pred2 = -1
                    
                    predictions_classes_array.append(
                        [i, offset, 
                        # identity
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity'].iloc[choice], 
                        # age
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'age'].iloc[choice], 
                        # filename
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'filenames'].iloc[choice], 
                        # predicting noised image
                        pred1, 
                        # predicting original b/w image
                        pred2, 
                        # euclidean distance
                        tf.norm(res1 - res2, ord=2).numpy(), 
                        # cosine distance
                        (tf.matmul(res1, tf.transpose(res2)).numpy() / (tf.norm(res1, ord=2) * tf.norm(res2, ord=2)).numpy())[0][0]],
                        # identity grouping distance
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'].iloc[choice]
                    )
                    
        return predictions_classes_array
    
    def calculate_confusion_matrix_elements(self, predictions_classes):
        predictions_classes['TP'] = 0
        predictions_classes.loc[(predictions_classes['y_pred'] == predictions_classes['true_identity']) & \
                                (predictions_classes['true_identity'] == predictions_classes['y_drift']) & \
                                (predictions_classes['y_pred'] == predictions_classes['y_drift']), 'TP'] = 1
        predictions_classes['TN'] = 0
        predictions_classes.loc[(predictions_classes['y_pred'] == predictions_classes['true_identity']) & \
                                (predictions_classes['y_drift'] != predictions_classes['true_identity']) & \
                                (predictions_classes['y_pred'] == predictions_classes['y_drift']), 'TN'] = 1
        predictions_classes['FP'] = 0
        predictions_classes.loc[(predictions_classes['y_pred'] == predictions_classes['true_identity']) & \
                                (predictions_classes['y_drift'] != predictions_classes['true_identity']) & \
                                (predictions_classes['y_pred'] != predictions_classes['y_drift']), 'FP'] = 1
        predictions_classes['FN'] = 0
        predictions_classes.loc[(predictions_classes['y_pred'] != predictions_classes['y_drift']) & \
                                (predictions_classes['y_drift'] == predictions_classes['true_identity']) & \
                                (predictions_classes['y_pred'] != predictions_classes['y_drift']), 'FN'] = 1
        
        return predictions_classes
    
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