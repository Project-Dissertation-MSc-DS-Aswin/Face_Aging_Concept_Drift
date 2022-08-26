import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC

def base_estimators_voting_classifier_face_recognition(param_grid, param_grid2, param_grid3):
  """
  Base estimators showing a voting classifier for Face Recognition
  @param param_grid: dict
  @param param_grid2: dict
  @param param_grid3: dict
  @return:
  """

  # random_state used to stabilize hyperparameter drift
  svm_embeding = RandomizedSearchCV(
      SVC(kernel="linear", probability=True), param_grid, cv=2, random_state=42
  )
  rf_emb = RandomizedSearchCV(RandomForestClassifier(), param_grid2, cv=2, random_state=42)
  hist_emb = RandomizedSearchCV(HistGradientBoostingClassifier(), param_grid3, cv=2, random_state=42)
  knn_emb = KNeighborsClassifier(n_neighbors=4)
  
  return (
    svm_embeding, 
    rf_emb, 
    hist_emb, 
    knn_emb
  )