import streamlit as st
import cv2
import requests
from PIL import Image
import config
import numpy as np
import io

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.radio(
            'Go To',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()
        
app = MultiApp()

PAGE_CONFIG = {"page_title": "StColab.io", "page_icon": ":smiley:", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    test_X = cv2.resize(gray, (64, 64))

    return test_X

def app_home():
    st.title("Home")
    submit_files = []
    if st.button('Apply'):
      col1, col2 = st.columns(2)
      res = requests.get(config.API_HOST + f"backend/images/10/")
      img_path = res.json()
      filenames = np.array(img_path.get("paths"))
      drifted = np.array(img_path.get("drifted"))
      
      for ii in range(len(filenames)):
        checkbox = st.checkbox("Drifted1: " + str(filenames[ii].split("/")[-1]) + " - " + str(drifted[ii]), value=filenames[ii])
        if checkbox:
          submit_files.append(filenames[ii])
      
      res = requests.post(config.API_HOST + f"backend/drift/1000/10/face_aging/", params={"filenames": submit_files})
      output = res.json()
      results = np.array(output.get("predictions"))
      score_test = np.array(output.get("score_test"))
      score_validation = np.array(output.get("score_validation"))
      for ii, result in enumerate(filenames):
        col1.checkbox("Drifted: " + str(filenames[ii].split("/")[-1]) + " - " + str(drifted[ii]), value=filenames[ii])
      for result in filenames:
        r = requests.get(result, stream=True)
        col2.image(np.asarray(Image.open(io.BytesIO(r.content))), caption='Output Image from network')
      st.write(str(score_test))
      st.write(str(score_validation))
      for ii, pred in enumerate(results):
        st.write(str(pred))

    else:
      col1, col2 = st.columns(2)
      res = requests.get(config.API_HOST + f"backend/images/10", params={'num': 10})
      img_path = res.json()
      filenames = np.array(img_path.get("paths"))
      drifted = np.array(img_path.get("drifted"))
      
      for ii, result in enumerate(filenames):
        col1.checkbox("Drifted: " + str(result.split("/")[-1]) + " - " + str(drifted[ii]), value=result)
      for result in filenames:
        r = requests.get(result, stream=True)
        col2.image(np.asarray(Image.open(io.BytesIO(r.content))), caption='Output Image from network')

# Add all your application here
app.add_app("Home", app_home)

app.run()
