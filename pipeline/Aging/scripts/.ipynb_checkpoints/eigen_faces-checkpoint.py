from context import Constants, Args
import pickle
from sklearn.manifold import Isomap
from tqdm import tqdm
import os
from sklearn.decomposition import PCA, KernelPCA
import numpy as np

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

args.covariates_beta = os.environ.get('covariates_beta', 0)
args.pca_type = os.environ.get('pca_type', 'KernelPCA')
args.pca_covariates_pkl = os.environ.get('pca_covariates_pkl', constants.AGEDB_PCA_COVARIATES)

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

def pca_covariates(images_cov, pca_type='PCA', covariates_beta=1, seed=1000):
    pca = KernelPCA(n_components=images_cov.shape[0], kernel='poly') if pca_type == 'KernelPCA' else PCA(n_components=images_cov.shape[0])
    np.random.seed(seed)
    X_pca = pca.fit_transform(images_cov * np.random.normal(0, covariates_beta, size=images_cov.shape) if covariates_beta else images_cov)
    return pca.components_.T if pca_type == 'PCA' else pca.eigenvectors_, pca, X_pca

def isomap_images(images_bw):
    isomap = Isomap(n_components=images_bw.shape[0])
    X_transform = isomap.fit(images_bw)
    return isomap.embedding_vectors_, isomap, X_transform

images_bw  = pickle.load(open("images_bw.pkl", "rb"))

np.random.seed(1000)
images_new = demean_images(images_bw, len(images_bw))
images_cov = images_covariance(images_new, len(images_new))
P, pca, X_pca = pca_covariates(images_cov, args.pca_type, args.covariates_beta)

pickle.dump(pca, open(args.pca_covariates_pkl, "wb"))
pickle.dump(P, open("P.pkl", "wb"))
pickle.dump(images_new, open("images_new.pkl", "wb"))

print(pca)

