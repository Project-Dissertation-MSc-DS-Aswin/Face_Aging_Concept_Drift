import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

def base_estimators_voting_classifier_face_recognition(param_grid, param_grid2, param_grid3):
  svm_embeding = RandomizedSearchCV(
      SVC(kernel="linear", probability=True), param_grid, n_iter=10, cv=2
  )
  rf_emb = RandomizedSearchCV(RandomForestClassifier(), param_grid2, n_iter=10, cv=2)
  hist_emb = RandomizedSearchCV(HistGradientBoostingClassifier(), param_grid3, n_iter=10, cv=2)
  knn_emb = KNeighborsClassifier(n_neighbors=4)
  
  return (
    svm_embeding, 
    rf_emb, 
    hist_emb, 
    knn_emb
  )