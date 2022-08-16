from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import numpy as np
import cv2
from tqdm import tqdm

def random_forest(X_train, y_train):
  rf = RandomForestClassifier(random_state=42, n_estimators=120)
  rf.fit(X_train, y_train)
  return rf

def score_random_forest(rf, X_test, y_test):
  return accuracy_score(y_test, rf.predict(X_test))

def generate_predictions(rf, X_test, filenames, y_test):
  y_pred_test = rf.predict(X_test)
  return pd.DataFrame(dict(y_pred_test=y_pred_test, y_test=y_test, filenames=filenames)).sort_values(by=['y_pred_test'])

class DriftProperties:
  
  def extract_statistical_properties(self, args, experiment, images, reconstructed_images, orig_images, psnr_error=0.05):
    mse_p_list = []
    mse_t_list = []
    mse_corr_list = []
    psnr_pca = []
    
    for ii, (filename_beta, pca_image) in tqdm(enumerate(reconstructed_images.items())):
      pca_image_bw = cv2.cvtColor(cv2.resize(pca_image[0], (96,96)), cv2.COLOR_RGB2GRAY)
      orig_image_bw = images[ii][:,:,0]
      orig_image = orig_images[filename_beta][0]
      output_image = experiment.drift_type_function(args.drift_type, orig_image, pca_image[0], beta=args.drift_beta, function_type=args.function_type)
      output_image_bw = experiment.drift_type_function(args.drift_type, orig_image_bw, pca_image_bw, beta=args.drift_beta, function_type=args.function_type)
      
      noise_img = random_noise(output_image_bw/255., mode='s&p',amount=psnr_error)
      noise_img = np.array(255*noise_img, dtype = 'uint8')
      
      noise_orig_img = random_noise(orig_image_bw/255., mode='s&p',amount=psnr_error)
      noise_orig_img = np.array(255*noise_orig_img, dtype = 'uint8')
      
      if args.denoise_type == 'gaussian':
          denoised = cv2.GaussianBlur(noise_img,(3,3),3,3,cv2.BORDER_DEFAULT)
          denoised_orig = cv2.GaussianBlur(noise_orig_img,(3,3),3,3,cv2.BORDER_DEFAULT)
      elif args.denoise_type == 'opencv_denoising':
          denoised = cv2.fastNlMeansDenoising(noise_img)
          denoised_orig = cv2.fastNlMeansDenoising(noise_orig_img)
          
      residual_orig = orig_image_bw - denoised_orig
      residual_img = output_image_bw - denoised
      
      # it will be outside the range
      psnr_pca.append(peak_signal_noise_ratio(orig_image/255., output_image/255.))
      
      covariance = np.cov(denoised/255., residual_img/255.)
      covariance_orig = np.cov(denoised_orig/255., residual_orig/255.)
      
      corr = covariance[experiment.dataset.dim[0]:, :experiment.dataset.dim[1]] / (np.std(denoised/255.) * np.std(residual_img/255.))
      corr_orig = covariance_orig[experiment.dataset.dim[0]:, :experiment.dataset.dim[1]] / (np.std(denoised_orig/255.) * np.std(residual_orig/255.))
      
      corr = MinMaxScaler(feature_range=(-0.99,0.99)).fit_transform(corr)
      corr_orig = MinMaxScaler(feature_range=(-0.99,0.99)).fit_transform(corr_orig)
      
      mse_corr = mean_squared_error(corr, corr_orig)
      
      t_value = corr * np.sqrt((corr.shape[0] - 2) / (1 - corr**2))
      t_value_orig = corr_orig * np.sqrt((corr_orig.shape[0] - 2) / (1 - corr_orig**2))
      
      p_value = np.zeros_like(t_value)
      p_value_orig = np.zeros_like(t_value_orig)
      
      mse_t = mean_squared_error(t_value, t_value_orig)
      
      # 2-tailed test because mean diff may be positive or negative
      for ii, _t_value in enumerate(t_value):
          _t_value = [t if t < 0 else -t for t in _t_value]
          cdf1 = scipy.stats.t.cdf(_t_value, df=(t_value.shape[0]+t_value.shape[0]-2))
          p_value[ii] = cdf1

      for ii, _t_value in enumerate(t_value_orig):
          _t_value = [t if t < 0 else -t for t in _t_value]
          cdf1 = scipy.stats.t.cdf(_t_value, df=(t_value_orig.shape[0]+t_value_orig.shape[0]-2))
          p_value_orig[ii] = cdf1
      
      mse_p = mean_squared_error(p_value, p_value_orig)
      
      mse_p_list.append(mse_p)
      mse_t_list.append(mse_t)
      mse_corr_list.append(mse_corr)
      
    return mse_p_list, mse_t_list, mse_corr_list, psnr_pca