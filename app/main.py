"""
Adapted from Source: 
https://testdriven.io/blog/fastapi-streamlit/
"""

# Import Necessary Libraries

import uuid
import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI, Header
from fastapi import UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import pandas as pd
import config
import inference
import drift
import app_preprocessing
import context
import os
from time import time
import pickle
from config import *
from time import time
from typing import Union, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import socketio
import config

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')

@sio.event
def message(data):
    print('message received with ', data)

@sio.event
def disconnect():
    print('disconnected from server')
    
sio.connect('http://localhost:8000/', wait_timeout = 10)

app = FastAPI()

origins = [
    "http://localhost:3000", 
    "http://localhost:8041",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("../src/data_collection/agedb_drift_beta_optimized.csv")

def map_pipeline(pipeline):
  if pipeline == "face_aging":
    return "Aging"
  return ""

"""
REST API:
GET /
    - Displays the message Welcome from the API
"""
@app.get("/")
def read_root():
  return {"message": "Welcome from the API"}


"""
REST API:
POST /model/classification
    - Uploaded File
POST /model/anomaly_detection
    - Uploaded File
"""
# @app.post("/model/{style}/")
# def get_image(style: str, file: UploadFile = File(...)):
#   image = np.array(Image.open(file.file))
#   if style == "anomaly_detection":
#     image = cv2.resize(image, (224,224))
#     output, viz_output, image_label, inference_time = inference.(best_model_anomaly_detection, anomaly_detection_models, image)
#   elif style == "classification":
#     image = cv2.resize(image, (224,224))
#     output, viz_output, image_label, inference_time = inference.(best_model_classification, classification_models, image)

#   return {"output": output, "viz_output": viz_output, "label": image_label, "inference_time": inference_time}

@app.get("/images/temp/{images_path}")
async def get_image(images_path: str):
  def iterfile():
    with open('images/temp/' + images_path, mode="rb") as file_like:
      yield from file_like

  return StreamingResponse(iterfile(), media_type="image/jpg")

@app.get("/backend/images/{num}/")
def get_image_url(num: int):
  paths = []
  np.random.seed(1000)
  drifted_df = df.loc[df['drifted'] == 0].sample(num//2)
  np.random.seed(1000)
  non_drifted_df = df.loc[df['drifted'] == 1].sample(num//2)
  dataframe = pd.concat([drifted_df, non_drifted_df], axis=0)
  image_paths = dataframe.filename
  drifted = dataframe.drifted.values.tolist()
  
  for image_path in image_paths.values:
    image_path = image_path.replace(".jpg", "-resized.jpg")
    paths.append("""images/temp/{image}""".format(image=image_path))
  
  return {"paths": paths, "drifted": drifted}

class Item(BaseModel):
  filenames: List[str] = []
  
@app.post("/backend/drift/{seed}/{num}/{pipeline}/")
def get_drift_predictions(seed: int, num: int, pipeline: str, item: Item):
  
  filenames = item.filenames
  
  args = context.Args({})
  args.metadata = "../src/dataset_meta/AgeDB_metadata.mat"
  args.classifier = "../src/models/agedb_voting_classifier_age.pkl"
  args.batch_size = 128
  args.dataset = "agedb"
  args.data_dir = "../../datasets/AgeDB"
  args.input_shape = (-1,160,160,3)
  args.alt_input_shape = (-1,96,96,3)
  args.model = "FaceNetKeras"
  
  sio.emit('status', {'percentage': 0, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  ml_model_classification = pickle.load(open(args.classifier, 'rb'))
  
  sio.emit('status', {'percentage': 1, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  preprocessor = app_preprocessing.Preprocessor()
  
  X, y = preprocessor.prepare_data(filenames)
  
  sio.emit('status', {'percentage': 15, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  X, y = preprocessor.prepare_drift_features_classification(X, y, ml_model_classification, filenames)
  
  sio.emit('status', {'percentage': 35, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  pipeline_folder = "Aging"
  
  X.to_csv(os.path.join("../pipeline", pipeline_folder, "output", "df_" + str(time()) + "_data.csv"))
  
  dataframe = pd.read_csv(os.path.join("../pipeline", pipeline_folder, "input", "agedb_drift_beta_optimized_features.csv"))
  
  data_X = dataframe[['drift_beta', 'psnr', 'mse_p', 'age']]
  X_train, X_test, y_train, y_test = preprocessor.train_test_split(data_X, dataframe['drifted'])
  
  sio.emit('status', {'percentage': 45, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  rf = drift.random_forest(X_train, y_train)
  
  sio.emit('status', {'percentage': 65, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  score_validation = drift.score_random_forest(rf, X_test, y_test)
  
  sio.emit('status', {'percentage': 75, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  score_test = drift.score_random_forest(rf, X.drop(columns=['filename', 'drifted']), y)
  
  sio.emit('status', {'percentage': 85, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  predictions_df = drift.generate_predictions(rf, X.drop(columns=['filename', 'drifted']), filenames, X['drifted'])
  
  sio.emit('status', {'percentage': 95, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  predictions_df.to_csv(os.path.join("../pipeline", pipeline_folder, "output", "predictions_df_" + str(time()) + ".csv"))
  
  sio.emit('status', {'percentage': 100, 'display': '<em class="progressbar-display">&nbsp;</em>'})
  
  return {"predictions": predictions_df.to_dict(), "score_validation": score_validation, "score_test": score_test}
  
if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8080)