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
from concurrent.futures import ThreadPoolExecutor
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import scipy.stats

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
        previous_weights = tf.Variable(10 * np.random.normal(size=(len(metadata), 9216, 1)), dtype=tf.float64)
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
                    output_image = orig_image - 1e-6 * image
                    # output_image = np.concatenate([np.expand_dims(output_image, 2)]*3, 2)
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
    
    def collect_drift_statistics(self, images, images_bw, images_demean, weights_vector, offset_vector, 
                                  b_vector, offset, P_pandas, index, voting_classifier_array, 
                                  model_loader, args_psnr_error,
                                  drift_type, drift_beta=0):
        
        def data_predictions(i, psnr_error, res1_images, res2_images, psnr_pca, mse_t_list, mse_p_list, mse_corr_list):
            images_copy = images
            images_demean_copy = images_demean
            f_now = self.dataset.metadata['identity_grouping_distance'] * (
                self.aging_function(weights_vector, b_vector, offset_vector) if self.args.mode == 'image_reconstruction' else self.aging_function_perturbation(weights_vector, b_vector, 0)
            )
            f_p_now = f_now[self.dataset.metadata['hash_sample'] == i] / \
                    np.sum(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])
            f_new = self.dataset.metadata['identity_grouping_distance'] * (
                self.aging_function(weights_vector, b_vector, offset)  if self.args.mode == 'image_reconstruction' else self.aging_function_perturbation(weights_vector, b_vector, 0)
            )
            f_p_new = f_new[self.dataset.metadata['hash_sample'] == i] / \
                    np.sum(self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'identity_grouping_distance'])

            b_new = b_vector[self.dataset.metadata['hash_sample'] == i] + f_p_new.values.reshape(-1, 1) - f_p_now.values.reshape(-1, 1)

            P_pandas_1 = P_pandas.loc[index.index.values[self.dataset.metadata['hash_sample'] == i],
                                    index.index.values[self.dataset.metadata['hash_sample'] == i]]

            images_syn = images_demean_copy.copy().reshape(len(self.dataset), -1)
            images_syn[self.dataset.metadata['hash_sample'] == i] = P_pandas_1.values.dot(b_new)

            new_images = \
                (images_syn[self.dataset.metadata['hash_sample'] == i]) + \
                images_demean_copy.reshape(len(self.dataset), -1)[self.dataset.metadata['hash_sample'] == i]
                
            filenames = self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'filename'].values
            ages = self.dataset.metadata.loc[self.dataset.metadata['hash_sample'] == i, 'age'].values
            
            choices = list(range(len(new_images)))
            choices = np.unique(choices)
            for ii, choice in enumerate(choices):
                image = new_images[choice].reshape(self.dataset.dim[0], self.dataset.dim[1])
                # image_error is added for conducting negative test
                orig_image = images_copy[self.dataset.metadata['hash_sample'] == i].reshape(-1, self.experiment_dataset.dim[0], 
                        self.experiment_dataset.dim[1], 3)[choice]
                orig_image_bw = images_bw[self.dataset.metadata['hash_sample'] == i].reshape(-1, self.dataset.dim[0], 
                        self.dataset.dim[1])[choice]
                output_image_bw = self.drift_type_function(drift_type, orig_image_bw, image, beta=drift_beta)
                image = np.concatenate([np.expand_dims(image, 2)]*3, 2)
                image = model_loader.resize(image)
                output_image = self.drift_type_function(drift_type, orig_image, image, beta=drift_beta)
                res1 = l2_normalize(prewhiten(output_image)).reshape(*model_loader.input_shape)
                res2 = l2_normalize(prewhiten(orig_image)).reshape(*model_loader.input_shape)
                
                res1_images.append(res1)
                res2_images.append(res2)
                
                noise_img = random_noise(output_image_bw/255., mode='s&p',amount=psnr_error)
                noise_img = np.array(255*noise_img, dtype = 'uint8')
                
                noise_orig_img = random_noise(orig_image_bw/255., mode='s&p',amount=psnr_error)
                noise_orig_img = np.array(255*noise_orig_img, dtype = 'uint8')
                
                denoised = cv2.fastNlMeansDenoising(noise_img,2,5,7, cv2.NORM_L2)
                denoised_orig = cv2.fastNlMeansDenoising(noise_orig_img,2,5,7, cv2.NORM_L2)
                
                residual_orig = orig_image_bw - denoised_orig
                residual_img = output_image_bw - denoised
                
                # it will be outside the range
                psnr_pca.append(peak_signal_noise_ratio(orig_image/255., output_image/255.))
                
                covariance = np.cov(denoised/255., residual_img/255.)
                covariance_orig = np.cov(denoised_orig/255., residual_orig/255.)
                
                corr = covariance / (np.std(denoised/255.) * np.std(residual_img/255.)) / 500.
                corr_orig = covariance_orig / (np.std(denoised_orig/255.) * np.std(residual_orig/255.)) / 500.
                
                mse_corr = mean_squared_error(corr, corr_orig)
                
                t_value = corr * np.sqrt((corr.shape[0] - 2) / (1 - corr**2))
                t_value_orig = corr_orig * np.sqrt((corr_orig.shape[0] - 2) / (1 - corr_orig**2))
                
                p_value = np.zeros_like(t_value)
                p_value_orig = np.zeros_like(t_value_orig)
                
                mse_t = mean_squared_error(t_value, t_value_orig)
                
                # 2-tailed test because mean diff may be positive or negative
                for ii, _t_value in enumerate(t_value):
                    _t_value = [t if t < 0 else -t for t in _t_value]
                    cdf1 = scipy.stats.t.cdf(_t_value, df=(t_value.shape[0]+t_value.shape[0]-2))*2
                    p_value[ii] = cdf1

                for ii, _t_value in enumerate(t_value_orig):
                    _t_value = [t if t < 0 else -t for t in _t_value]
                    cdf1 = scipy.stats.t.cdf(_t_value, df=(t_value_orig.shape[0]+t_value_orig.shape[0]-2))*2
                    p_value_orig[ii] = cdf1
                
                mse_p = mean_squared_error(p_value, p_value_orig)
                
                mse_p_list.append(mse_p)
                mse_t_list.append(mse_t)
                mse_corr_list.append(mse_corr)
                
            return res1_images, res2_images, filenames, ages, psnr_pca, mse_t_list, mse_p_list, mse_corr_list
            
        print(np.unique(self.dataset.metadata['hash_sample']))
        
        data = []
        res1_images = []
        res2_images = []
        psnr_pca = []
        cols = []
        ages_list = []
        mse_p_list = []
        mse_corr_list = []
        mse_t_list = []
        for ii in np.unique(self.dataset.metadata['hash_sample']):
            res1_images, res2_images, filenames, ages, psnr_pca, mse_t_list, mse_p_list, mse_corr_list = \
                data_predictions(int(ii), args_psnr_error, res1_images, res2_images, psnr_pca, mse_t_list, mse_p_list, mse_corr_list)
                
            cols += filenames.tolist()
            ages_list += ages.tolist()

        all_res1_images = np.vstack(res1_images)
        all_res2_images = np.vstack(res2_images)
        
        inference_images = model_loader.infer(all_res1_images)
        inference_images_orig = model_loader.infer(all_res2_images)
        
        # if len(voting_classifier_array) > 0:
        #     matches = np.zeros((len(voting_classifier_array), len(all_res1_images)))
        #     virtual_matches = np.zeros((len(voting_classifier_array), len(all_res1_images)))
        #     pred_original = {}
        #     pred_drifted = {}
        #     names = self.get_name_from_filename(cols, dataset='agedb')
        #     for ij, voting_classifier in tqdm(enumerate(voting_classifier_array)):
        #         # reconstructed image
        #         pred_virtual_classes = voting_classifier.predict(
        #             inference_images
        #         )
                
        #         # original image
        #         pred_orig_classes = voting_classifier.predict(
        #             inference_images_orig
        #         )
                
        #         pred_original[ij] = pred_orig_classes
        #         pred_drifted[ij] = pred_virtual_classes
                
        #         for jj, pred_orig in enumerate(pred_orig_classes):
        #             matches[ij][jj] += int(pred_orig == names[jj])
                
        #         for jj, pred_virtual in enumerate(pred_virtual_classes):
        #             virtual_matches[ij][jj] += int(pred_virtual == names[jj])
            
        #     orig_true_positives = [1 if match == 1 else 0 for match in np.sum(matches, axis=0)]
        #     orig_false_negatives = [0 if match == 1 else 1 for match in np.sum(matches, axis=0)]
            
        #     virtual_true_positives = [1 if match == 1 else 0 for match in np.sum(virtual_matches, axis=0)]
        #     virtual_false_negatives = [0 if match == 1 else 1 for match in np.sum(virtual_matches, axis=0)]
            
        #     statistical_drift_true_positives = []
        #     statistical_drift_true_negatives = []
        #     statistical_drift_undefined = []
            
        #     predictions_original = []
        #     predictions_virtual = []
            
        #     for jj, match in enumerate(np.sum(matches, axis=0)):
        #         idx = np.where(match == 1)
        #         pred_orig
        #         idx = idx[0][0] if len(idx[0]) > 0 else None
        #         if (idx is not None) and (pred_original[idx][jj] == pred_drifted[idx][jj]):
        #             statistical_drift_true_positives.append(1)
        #             statistical_drift_true_negatives.append(0)
        #             statistical_drift_undefined.append(0)
        #         elif idx is not None:
        #             statistical_drift_true_positives.append(0)
        #             statistical_drift_true_negatives.append(1)
        #             statistical_drift_undefined.append(0)
        #         else:
        #             statistical_drift_undefined.append(1)
        #             statistical_drift_true_positives.append(0)
        #             statistical_drift_true_negatives.append(0)
                    
        #         pred_virtual = pred_drifted[idx][jj] if idx is not None else -1
        #         pred_orig = pred_original[idx][jj] if idx is not None else -1
                
        #         predictions_original.append(pred_orig)
        #         predictions_virtual.append(pred_virtual)
                
        # else:
        #     pred_virtual = -1
        #     pred_orig = -1
        
        data = (inference_images, inference_images_orig, cols, ages_list, psnr_pca, mse_t_list, mse_p_list, mse_corr_list)
        
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
    
    def drift_type_function(self, drift_type, original_image, virtual_image, beta=0, seed=1000):
        if drift_type == 'incremental':
            return original_image - beta * virtual_image
        elif drift_type == 'sudden':
            return original_image - beta * virtual_image
        elif drift_type == 'gradual':
            np.random.seed(seed)
            return original_image - np.random.normal(0, beta) * virtual_image
        elif drift_type == 'recurring':
            return original_image - np.sin(beta) * virtual_image

    def get_name_from_filename(self, filenames, dataset='agedb'):
        if dataset == 'agedb':
            return [filename.split("_")[1] for filename in filenames]
        elif dataset == 'cacd2000':
            return ['' for filename in filenames]
        elif dataset == 'fgnet':
            return ['' for filename in filenames]