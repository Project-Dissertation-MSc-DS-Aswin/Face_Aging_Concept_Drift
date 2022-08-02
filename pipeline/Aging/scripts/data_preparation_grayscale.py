from context import Constants, Args
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging
import pickle
import whylogs
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

args.dataset = os.environ.get('dataset', 'agedb')
args.logger_name = os.environ.get('logger_name', 'facenet_with_aging')
args.no_of_samples = os.environ.get('no_of_samples', 2598)
args.no_of_pca_samples = os.environ.get('no_of_pca_samples', 2598)
args.grayscale_input_shape = (-1,96,96,1)
args.drift_source_filename = os.environ.get('drift_source_filename', constants.AGEDB_DRIFT_SOURCE_FILENAME)

model_loader = pickle.load(open("model_loader.pkl", "rb"))

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
    if args.mode == 'image_reconstruction':
        filenames = pd.read_csv(args.drift_source_filename)
        idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
        result_idx = [False]*len(dataset.metadata)
        for i in idx:
            result_idx = np.logical_or(result_idx, i)

        return dataset.metadata.loc[result_idx].reset_index()
    elif args.mode == 'image_perturbation':
        filenames = pd.read_csv(args.drift_source_filename)
        idx = [dataset.metadata['filename'] == filename for filename in filenames['filename']]
        result_idx = [False]*len(dataset.metadata)
        for i in idx:
            result_idx = np.logical_or(result_idx, i)

        return dataset.metadata.loc[result_idx].reset_index()
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()

def collect_images(train_iterator):
    images_bw = []
    # Get input and output tensors
    for ii in tqdm(range(len(train_iterator))):
        (X, y) = train_iterator[ii]
        images_bw.append(X)

    return np.vstack(images_bw)

pca_args = copy(args)
pca_args.no_of_samples = pca_args.no_of_pca_samples

dataset, augmentation_generator = load_dataset(args, whylogs, (args.grayscale_input_shape[0], args.grayscale_input_shape[1]),
                                               args.no_of_pca_samples, 'grayscale')
dataset.set_metadata(
    get_reduced_metadata(pca_args, dataset)
)

images_bw = collect_images(dataset.iterator)

pickle.dump(images_bw, open("images_bw.pkl", "wb"))
pickle.dump(dataset, open("dataset.pkl", "wb"))
