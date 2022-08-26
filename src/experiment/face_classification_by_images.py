from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.metrics import accuracy_score, recall_score, f1_score
import cv2
from time import time

class FaceClassificationByImages:
  
  def __init__(self, dataset, logger, model_loader, detector=None):
    """
    __init__ function
    @param dataset:
    @param logger:
    @param model_loader:
    @param detector:
    """
    self.dataset = dataset
    self.logger = logger
    self.model_loader = model_loader
    self.batchno = 0
    self.detector = detector
    
  def set_dataset(self, dataset):
    """
    Set the dataset
    @param dataset:
    @return:
    """
    self.dataset = dataset

  def set_logger(self, logger):
    """
    Set the logger
    @param logger:
    @return:
    """
    self.logger = logger

  def set_model_loader(self, model_loader):
    """
    Set the model loader
    @param model_loader:
    @return:
    """
    self.model_loader = model_loader
    
  def collect_data(self, train_iterator, output_size):
    """
    Collect the data
    @param train_iterator:
    @param output_size:
    @return:
    """
    images_bw = []
    classes = []
    # Get input and output tensors
    for ii in tqdm(range(len(train_iterator))):
        (X, y) = train_iterator[ii]
        img = []
        for jj, x in enumerate(X):
          face = self.face_detect(x)
          if face is None:
            continue
          det = face[0][0:4].astype(np.int32)
          img.append(cv2.cvtColor(cv2.resize(x[det[0]:det[2], det[1]:det[3]], output_size), cv2.COLOR_RGB2GRAY))
          classes.append(y[jj])
        images_bw.append(np.stack(img))

    return np.vstack(images_bw), classes
  
  def face_detect(self, image):
    """
    Detect Faces
    @param image:
    @return:
    """
    model_loader = self.model_loader
    model_loader.resize(image)
    results = model_loader.infer(image)
    
    return results
  
  def preprocess_and_split(self, images_bw, classes):
    """
    Preprocess and split the images
    @param images_bw:
    @param classes:
    @return:
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(classes)//117):
      _X_train, _X_test, _y_train, _y_test = train_test_split(
        images_bw[i*117:i*117+117], classes[i*117:i*117+117], test_size=0.25, random_state=42
      )

      scaler = StandardScaler()
      _X_train = scaler.fit_transform(_X_train)
      _X_test = scaler.transform(_X_test)
      
      X_train.append(_X_train)
      X_test.append(_X_test)
      y_train.append(_y_train)
      y_test.append(_y_test)
    
    return X_train, X_test, y_train, y_test
  
  def pca_transform(self, X_train, X_test):
    """
    Apply PCA transform to training and testing set
    @param X_train:
    @param X_test:
    @return:
    """
    h, w = self.dataset.dim
    
    n_components = 5

    print(
        "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
    )
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    
    return X_train_pca, X_test_pca, eigenfaces, pca
  
  def train(self, X_train_pca, y_train):
    """
    Train the classifier model
    @param X_train_pca:
    @param y_train:
    @return:
    """
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {
        "C": loguniform(1e-2, 1e4),
        "gamma": loguniform(1e-4, 1e-1),
    }
    clf = RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=20, cv=5
    )
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_, clf.classes_)
    
    return clf
    
  def score(self, clf, X_test_pca, y_test):
    """
    Score the classifier model
    @param clf:
    @param X_test_pca:
    @param y_test:
    @return:
    """
    y_pred = clf.predict(X_test_pca)
    
    print("""Score:
      Accuracy: {accuracy}, 
      F1 Score: {f1_score}, 
      Recall: {recall}
    """.format(accuracy=accuracy_score(y_test, y_pred), f1_score=f1_score(y_test, y_pred, average='weighted'), recall=recall_score(y_test, y_pred, average='weighted')))
    
    return y_pred