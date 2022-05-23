from re import M
import numpy as np
import pandas as pd
import random
import skimage
import tensorflow as tf
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from preprocessing.facenet import l2_normalize, prewhiten
from copy import copy

# use Agg backend to suppress the plot shown in command line
matplotlib.use('Agg')

class DriftSynthesisByEigenFacesExperiment:

    def __init__(self, dataset, logger=None, model_loader=None, pca=None, init_offset=None):
        self.dataset = dataset
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

    def eigen_vector_coefficient(self, eigen_vectors, images_demean):
        return np.matmul(np.linalg.inv(eigen_vectors), images_demean.reshape(self.no_of_images, -1))

    def mahalanobis_distance(self, eigen_vector_coefficient):
        def mahalanobis(bn, bm, C):
            if len(C.shape) == 0:
                return ((bn - bm)).dot((bn - bm).T) * C
            else:
                return (((bn - bm).T).dot(np.linalg.inv(C))).T.dot((bn - bm).T)

        b_vector = eigen_vector_coefficient
        bm = np.expand_dims(b_vector.mean(axis=1), 1)
        C = np.cov(b_vector)
        distance = mahalanobis(b_vector, bm, C)

        return distance

    def set_hash_sample_by_distinct(self, identity_grouping_distance):
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
            (self.dataset.metadata['identity_grouping_distance'] >= new_cutoff_range[-1]), 'hash_sample'] = len(
            new_cutoff_range)

        return reduced_metadata

    def weights_vector(self, reduced_metadata, b_vector, init_offset=None):
        if not init_offset:
            init_offset = self.init_offset
        w = (reduced_metadata['age'] - init_offset).dot(np.linalg.inv(b_vector))
        return w

    def aging_function(self, weights_vector, b_vector, init_offset=None):
        if not init_offset:
            init_offset = self.init_offset
        return weights_vector.T.dot(b_vector) + init_offset

    def plot_images_with_eigen_faces(self, images, images_demean, images_mean, weights_vector, b_vector, offset_range, P_pandas, index):
        figures = []
        choices_array = []
        for i in tqdm(np.unique(self.dataset.metadata['hash_sample'])):
            figure = []
            choices_list = []
            f_now = self.dataset.metadata['identity_grouping_distance'] * (self.aging_function(weights_vector, b_vector, -11))
            f_p_now = f_now[self.dataset.metadata['hash_sample'] == i] / \
                      np.sum(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])
            for offset in offset_range:
                f_new = self.dataset.metadata['identity_grouping_distance'] * (self.aging_function(weights_vector, b_vector, offset))
                f_p_new = f_new[self.dataset.metadata['hash_sample'] == i] / \
                          np.sum(
                              self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])

                b_new = b_vector[self.dataset.metadata['hash_sample'] == i] + f_p_new.values.reshape(-1,
                                                                                         1) - f_p_now.values.reshape(-1,
                                                                                                                     1)

                P_pandas_1 = P_pandas.loc[index.index.values[self.dataset.metadata['hash_sample'] == i],
                                          index.index.values[self.dataset.metadata['hash_sample'] == i]]

                images_syn = images_demean.copy().reshape(len(self.dataset.iterator), -1)
                images_syn[self.dataset.metadata['hash_sample'] == i] = P_pandas_1.values.dot(b_new)

                new_images = \
                    (images_syn[self.dataset.metadata['hash_sample'] == i]) + \
                    images_mean[self.dataset.metadata['hash_sample'] == i].reshape(-1, 1)

                choices = [random.choice(list(range(len(new_images))))]

                fig1 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    plt.subplot(1, 3, ii + 1)
                    plt.imshow(new_images.reshape(-1, self.dataset.dim[0], self.dataset.dim[1])[choice], cmap='gray')
                    plt.title("Age Function")

                fig2 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    plt.subplot(1, 3, ii + 1)
                    plt.imshow(
                        skimage.color.rgb2gray(images[self.dataset.metadata['hash_sample'] == i].reshape(-1, self.dataset.dim[0], self.dataset.dim[1], 3))[
                            choice],
                        cmap='gray')
                    plt.title("Actual Age = " + \
                              self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'age'].iloc[choice].astype(
                                  str))

                fig3 = plt.figure(figsize=(8, 16))
                for ii, choice in enumerate(choices):
                    plt.subplot(1, 3, ii + 1)
                    plt.imshow(
                        skimage.color.rgb2gray(images[self.dataset.metadata['hash_sample'] == i].reshape(-1, self.dataset.dim[0], self.dataset.dim[1], 3))[
                            choice] + \
                        new_images.reshape(-1, self.dataset.dim[0], self.dataset.dim[1])[choice], cmap='gray')
                    plt.title("Predicted Age = " + str(np.round(f_p_new.iloc[choice], 2)))

                figure.append([fig1, fig2, fig3])
                choices_list.append(choices)

            figures.append(figure)
            choices_array.append(choices_list)
            
        plt.close()

        return figures, choices_array

    def collect_drift_predictions(self, images, images_demean, images_mean, weights_vector, b_vector, offset_range, P_pandas, index, 
                                  classifier, model_loader, choices_array=None):
        
        predictions_classes_array = []
        
        for ii, i in tqdm(enumerate(np.unique(self.dataset.metadata['hash_sample']))):
            f_now = self.dataset.metadata['identity_grouping_distance'] * (self.aging_function(weights_vector, b_vector, -11))
            f_p_now = f_now[self.dataset.metadata['hash_sample'] == i] / \
                      np.sum(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])
            for jj, offset in enumerate(offset_range):
                f_new = self.dataset.metadata['identity_grouping_distance'] * (self.aging_function(weights_vector, b_vector, offset))
                f_p_new = f_new[self.dataset.metadata['hash_sample'] == i] / \
                          np.sum(
                              self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])

                b_new = b_vector[self.dataset.metadata['hash_sample'] == i] + \
                                f_p_new.values.reshape(-1, 1) - f_p_now.values.reshape(-1, 1)

                P_pandas_1 = P_pandas.loc[index.index.values[self.dataset.metadata['hash_sample'] == i],
                                          index.index.values[self.dataset.metadata['hash_sample'] == i]]

                images_syn = images_demean.copy().reshape(len(self.dataset.iterator), -1)
                images_syn[self.dataset.metadata['hash_sample'] == i] = P_pandas_1.values.dot(b_new)

                new_images = \
                    (images_syn[self.dataset.metadata['hash_sample'] == i]) + \
                    images_mean[self.dataset.metadata['hash_sample'] == i].reshape(-1, 1)

                if choices_array is None:
                    choices = [random.choice(list(range(len(new_images))))]
                else:
                    choices = choices_array[ii][jj]
                    
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
                    
                    predictions_classes_array.append(
                        [i, offset, 
                        # identity
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity'].iloc[choice], 
                        # age
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'age'].iloc[choice], 
                        # filename
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'filenames'].iloc[choice], 
                        # predicting noised image
                        classifier.predict(
                            res1.numpy()
                        )[0], 
                        # predicting original b/w image
                        classifier.predict(
                            res2.numpy()
                        )[0], 
                        # euclidean distance
                        tf.norm(res1 - res2, ord=2).numpy(), 
                        # cosine distance
                        (tf.matmul(res1, tf.transpose(res2)).numpy() / (tf.norm(res1, ord=2) * tf.norm(res2, ord=2)).numpy())[0][0]],
                        # identity grouping distance
                        self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'].iloc[choice]
                    )
                    
        return predictions_classes_array