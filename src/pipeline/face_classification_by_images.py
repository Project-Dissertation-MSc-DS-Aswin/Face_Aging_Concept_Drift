"""Reference:

https://github.com/opencv/opencv/blob/master/samples/dnn/face_detect.py

https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/yunet.py

"""

import cv2 as cv
import numpy as np
import pandas as pd
from context import Constants, Args
from experiment.face_classification_by_images import FaceClassificationByImages
import os
import sys
from tqdm import tqdm
from copy import copy
from sklearn.decomposition import PCA
import whylogs
import mlflow
from datasets import CACD2000Dataset, FGNETDataset, AgeDBDataset
from experiment.model_loader import get_augmented_datasets, preprocess_data_facenet_without_aging, YuNetModelLoader
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import logging

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

args.dataset = os.environ.get('dataset', 'agedb')
args.model = os.environ.get('model', 'YuNet_onnx')
args.model_path = os.environ.get('model_path', 'face_detection_yunet_2022mar.onnx')
args.data_dir = os.environ.get('data_dir', constants.AGEDB_DATADIR)
args.batch_size = os.environ.get('batch_size', 128)
args.preprocess_prewhiten = os.environ.get('preprocess_prewhiten', 1)
args.metadata = os.environ.get('metadata', constants.AGEDB_METADATA)
args.logger_name = os.environ.get('logger_name', 'facenet_with_classifier')
args.no_of_samples = os.environ.get('no_of_samples', 1288)
args.colormode = os.environ.get('colormode', 'rgb')
args.log_images = os.environ.get('log_images', 's3')
args.tracking_uri = os.environ.get('tracking_uri', 'http://localhost:5000')
args.experiment_id = os.environ.get("experiment_id", 2)
args.conf_threshold = os.environ.get("conf_threshold", 0.9)
args.nms_threshold = os.environ.get("nms_threshold", 0.9)
args.backend = os.environ.get('backend', cv.dnn.DNN_BACKEND_OPENCV)
args.target = os.environ.get('target', cv.dnn.DNN_TARGET_CPU)
args.top_k = os.environ.get('top_k', 5000)
args.input_shape = os.environ.get('input_shape', (-1,96,96,1))

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
args.experiment_id = int(args.experiment_id)
args.target = int(args.target)
args.backend = int(args.backend)
args.nms_threshold = float(args.nms_threshold)
args.top_k = int(args.top_k)
args.conf_threshold = float(args.conf_threshold)
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    """
    Visualize the output of the network
    @param image:
    @param results:
    @param box_color:
    @param text_color:
    @param fps:
    @return:
    """
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output
  
def load_dataset(args, whylogs, no_of_samples, colormode, input_shape=(-1,160,160,3)):
    """
    Load the dataset
    @param args:
    @param whylogs:
    @param no_of_samples:
    @param colormode:
    @param input_shape:
    @return:
    """
    dataset = None
    augmentation_generator = None
    if args.dataset == "agedb":
        augmentation_generator = get_augmented_datasets()
        dataset = AgeDBDataset(whylogs, args.metadata, list_IDs=list(range(no_of_samples)),
                               color_mode=colormode, augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                               dim=(input_shape[1],input_shape[2]), 
                               batch_size=args.batch_size)
    elif args.dataset == "cacd":
        augmentation_generator = get_augmented_datasets()
        dataset = CACD2000Dataset(whylogs, args.metadata, list_IDs=list(range(no_of_samples)),
                                  color_mode=colormode, augmentation_generator=augmentation_generator,
                                  data_dir=args.data_dir, dim=(input_shape[1],input_shape[2]), batch_size=args.batch_size)
    elif args.dataset == "fgnet":
        augmentation_generator = get_augmented_datasets()
        dataset = FGNETDataset(whylogs, args.metadata, list_IDs=None,
                               color_mode=colormode, augmentation_generator=augmentation_generator, data_dir=args.data_dir, 
                               dim=(input_shape[1],input_shape[2]), 
                               batch_size=args.batch_size)

    return dataset, augmentation_generator
  
def get_reduced_metadata(args, dataset, seed=1000):
  """
  Get reduced metadata, reduce the metadata by applying pandas filters
  @param args:
  @param dataset:
  @param seed:
  @return:
  """
  if args.dataset == "fgnet":
    return dataset.metadata
  elif args.dataset == "agedb":
    np.random.seed(seed)
    names = dataset.metadata.groupby('name').count()
    names = names[names['age'] > 42]
    names = names.index.get_level_values(0)
    idx = [dataset.metadata['name'] == name for name in names]
    result_idx = [False]*len(dataset.metadata)
    for i in idx:
      result_idx = np.logical_or(result_idx, i)
      
    return dataset.metadata.loc[result_idx].reset_index().sort_values(by=['name'])
  elif args.dataset == "cacd":
    np.random.seed(seed)
    return dataset.metadata.sample(args.no_of_samples).reset_index()
  
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """
    Helper function to plot a gallery of portraits
    @param images:
    @param titles:
    @param h:
    @param w:
    @param n_row:
    @param n_col:
    @return:
    """
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], fontsize=9)
        plt.xticks(())
        plt.yticks(())
        
    return fig

if __name__ == "__main__":
  
  # set the tracking URI
  mlflow.set_tracking_uri(args.tracking_uri)
  
  # check if the model is YuNet_onnx
  if args.model == 'YuNet_onnx':
    model_loader = YuNetModelLoader(whylogs, args.model_path, input_shape=[1,96,96,3], conf_threshold=args.conf_threshold, nms_threshold=args.nms_threshold, backend=args.backend, target=args.target, top_k=args.top_k)
  
  # load the dataset
  dataset, augmentation_generator = load_dataset(args, whylogs, args.no_of_samples, 'rgb', input_shape=args.input_shape)

  # set the metadata
  dataset.set_metadata(
      get_reduced_metadata(args, dataset), class_mode='raw'
  )
  
  # extract the experiment
  experiment = FaceClassificationByImages(dataset, whylogs, model_loader)
  
  # collect data from the experiment
  images_bw, classes = experiment.collect_data(dataset.iterator, output_size=(args.input_shape[1], args.input_shape[2]))
  
  print("No. of classes: ", np.unique(classes))
  print("Classes: ", classes)
  
  # preprocess the data and split the data
  X_train, X_test, y_train, y_test = experiment.preprocess_and_split(images_bw.reshape(len(images_bw), -1), classes)
  
  X_train_pca, X_test_pca = [], []
  for i in range(len(X_train)):
    _X_train_pca, _X_test_pca, _eigenfaces, _pca = experiment.pca_transform(X_train[i], X_test[i])
    X_train_pca.append(_X_train_pca)
    X_test_pca.append(_X_test_pca)
  
  # train the classifiers
  clf = []
  for i in range(len(X_train_pca)):
    try:
      clf.append(experiment.train(X_train_pca[i], y_train[i]))
    except Exception as e:
      continue
    
  # score the classifiers
  y_pred = []
  for i in range(len(X_test_pca)):
    try:
      y_pred.append(experiment.score(clf[i], X_test_pca[i], y_test[i]))
    except Exception as e:
      continue
  
  # start mlflow experiment
  with mlflow.start_run(experiment_id=args.experiment_id):
    
    def title(y_pred, y_test, i):
      pred_name = y_pred[i].rsplit(" ", 1)[-1]
      true_name = y_test[i].rsplit(" ", 1)[-1]
      print("predicted: %s\ntrue:      %s" % (pred_name, true_name))
      return "predicted: %s\ntrue:      %s" % (pred_name, true_name) 

    for i in range(len(y_pred)):
      prediction_titles = [
        title(y_pred[i], y_test[i], j) for j in range(y_pred[i].shape[0])
      ]

      fig = plot_gallery(X_test[i], prediction_titles, args.input_shape[1], args.input_shape[2])
      # log the figure of mlflow for face classification by images
      mlflow.log_figure(fig, "images/fig_" + str(i) + ".jpg")