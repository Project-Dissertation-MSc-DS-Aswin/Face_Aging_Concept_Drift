# Face Aging Concept Drift

## **Requirements**

- torch
- torchvision
- numpy
- Pillow
- whylogs
- scipy
- pandas
- tqdm
- keras
- tensorflow
- scikit-learn
- opencv-python
- scikit-image
- mlflow

## **Environment**

- Python: 3.7
- Redis: 7.0.4

## **IMPORT MODELS**

Import model for Face Recognition by FaceNet

```
wget https://project-dissertation.s3.eu-west-2.amazonaws.com/facenet_keras.h5 -P src/models
```

Import model for Face Recognition by CVAE (Convolutional Variational Autoencoder)

```
wget https://project-dissertation.s3.eu-west-2.amazonaws.com/vit_face_recognition_model.h5 -P src/models
```

```
unzip -P ByrMAP@15 src/models/all_models.zip src/models
cp -Rf src/models/all_models/*.pkl src/models
```

```
unzip -P ByrMAP@15 src/models/16.07.2022_two_classifiers.zip src/models
cp -Rf src/models/16.07.2022_two_classifiers/*.pkl src/models
```

## **Using Docker**

Execute these commands in series
--------------------------------

```

git clone https://github.com/Project-Dissertation-MSc-DS-Aswin/Face_Aging_Concept_Drift/

docker-compose up -d

docker-compose exec face_aging_concept_drift unzip /home/project/src/models/all_ml_models.zip -d /home/project/src/models
docker-compose exec face_aging_concept_drift unzip /home/project/src/models/16.07.2022_two_classifiers.zip -d /home/project/src/models

docker-compose exec face_aging_concept_drift wget https://project-dissertation.s3.eu-west-2.amazonaws.com/facenet_keras.h5 -P /home/project/src/models
docker-compose exec face_aging_concept_drift wget https://project-dissertation.s3.eu-west-2.amazonaws.com/vit_face_recognition_model.h5 -P /home/project/src/models

```

## **RESULTS**

### **Anomaly Detection using GaussianMixtureModel**

![./images/gaussianmixturemodel-anomaly.png](./images/gaussianmixturemodel-anomaly.png)

### **Real Drift Classifier Approach using GaussianProcessClassifier**

![./images/gaussianprocessclassifier-real.png](./images/gaussianprocessclassifier-real.png)

### **Virtual Drift Classifier Approach using GaussianProcessClassifier**

![./images/Gaussianprocessclassifier-virtual.png](./images/Gaussianprocessclassifier-virtual.png)

## **RESULTS Comparison**

!["./images/comparison-of-accuracy-drift.png"](./images/comparison-of-accuracy-drift.png)

### **FOLDER STRUCTURE**

```
|   LICENSE
|   README.md
|
+---images
|       decision_model.png
|       drift_metric.png
|
\---src
    |_📂 src
       |_📁 aws
              |_📄 boto3_api.py
              |_📄 credentials.rar
       |_📁 dataset_meta
              |_📄 AgeDB_metadata.mat
              |_📄 celebrity2000_meta.mat
              |_📄 FGNET_metadata.mat
       |_📁 data_collection
              |_📁 16.07.2022_two_classifiers_baseline
              |_📁 16.07.2022_two_classifiers_facenet
              |_📄 agedb_drift_beta_optimized.csv
              |_📄 agedb_drift_beta_optimized_10_samples.csv
              |_📄 agedb_drift_source_table.csv
              |_📄 agedb_drift_synthesis_metrics.csv
              |_📄 agedb_inferences_facenet.pkl
              |_📄 agedb_two_classifiers_dissimilarity_measurement_model_drift.csv
              |_📄 age_predictions.csv
              |_📄 baseline_agedb_drift_synthesis_filename-0-2.csv
              |_📄 baseline_agedb_drift_synthesis_filename-0-5-shape.csv
              |_📄 baseline_agedb_drift_synthesis_filename-0-5.csv
              |_📄 baseline_agedb_drift_synthesis_filename-0.2-shape.csv
              |_📄 baseline_agedb_drift_synthesis_filename-0.csv
              |_📄 baseline_agedb_drift_synthesis_filename-1-0.csv
              |_📄 baseline_agedb_drift_synthesis_filename-minus-1.0-shape.csv
              |_📄 baseline_agedb_drift_synthesis_filename.csv
              |_📄 baseline_agedb_drift_synthesis_filename_minus-0-5.csv
              |_📄 baseline_agedb_drift_synthesis_metrics.csv
              |_📄 beta_morph_facenet_mse_corr_array_df_optimized.csv
              |_📄 beta_morph_facenet_mse_p_array_df_optimized.csv
              |_📄 beta_morph_facenet_mse_t_array_df_optimized.csv
              |_📄 beta_morph_facenet_psnr_pca_df_optimized.csv
              |_📄 cda_fedavg_observation.csv
              |_📄 embeddings_cacd_age_estimations.pkl
              |_📄 facenet_agedb_age_distribution_two_classifiers.csv
              |_📄 facenet_agedb_drift_evaluate_metrics.csv
              |_📄 facenet_agedb_drift_synthesis_filename-0-2.csv
              |_📄 facenet_agedb_drift_synthesis_filename-0-5.csv
              |_📄 facenet_agedb_drift_synthesis_filename-1-0.csv
              |_📄 facenet_agedb_drift_synthesis_filename-range-of-beta-10-samples.csv
              |_📄 facenet_agedb_drift_synthesis_morph_filename-range-of-beta.csv
              |_📄 facenet_agedb_drift_synthesis_morph_filename-range-of-beta_copy.csv
              |_📄 facenet_agedb_drift_synthesis_morph_filename-range-of-beta_copy.zip
              |_📄 morph_baseline_mse_corr_array_df.csv
              |_📄 morph_baseline_mse_p_array_df.csv
              |_📄 morph_baseline_mse_t_array_df.csv
              |_📄 morph_baseline_psnr_pca_df.csv
              |_📄 morph_facenet_mse_corr_array_df.csv
              |_📄 morph_facenet_mse_p_array_df.csv
              |_📄 morph_facenet_mse_t_array_df.csv
              |_📄 morph_facenet_psnr_pca_df.csv
              |_📄 t2_observation_ucl.csv
       |_📁 evaluation
              |_📁 __pycache__
              |_📄 distance.py
       |_📁 experiment
              |_📄 context.py
              |_📄 drift_synthesis_by_eigen_faces.py
              |_📄 face_classification_by_images.py
              |_📄 face_without_aging.py
              |_📄 face_with_classifier.py
              |_📄 face_with_clustering.py
              |_📄 model_loader.py
              |_📄 yunet.py
       |_📁 models
              |_📄 16.07.2022_two_classifiers.zip
              |_📄 all_ml_models.zip
              |_📄 CACD_MAE_4.59.pth
              |_📄 cvae_face_recognition_model.zip
              |_📄 facenet_keras.h5
              |_📄 mnist_epoch10.hdf5
              |_📄 mnist_epoch2.hdf5
              |_📄 mnist_epoch5.hdf5
              |_📄 vit_face_recognition_model.zip
       |_📁 notebooks
              |_📄 AgeDB_of image_classification_with_vision_transformer_encoder_decoder_loading_by_age.ipynb
              |_📄 Analysis.ipynb
              |_📄 Baseline_Model_image_classification_with_vision_transformer_encoder_decoder_loading_by_age.ipynb
              |_📄 classification_clustering_drift_agedb.ipynb
              |_📄 concept-drift-hypothesis-tests-Copy6.ipynb
              |_📄 Copy_of_DBSCAN_Performance_Analysis.ipynb
              |_📄 Copy_of_DBSCAN_Performance_Analysis_CACD_vs_AGEDB.ipynb
              |_📄 drift_synthesis_eda.ipynb
              |_📄 formative_viva_analysis.ipynb
              |_📄 image_classification_with_vision_transformer_encoder_decoder_loading_by_age.ipynb
              |_📄 image_classification_with_vision_transformer_encoder_decoder_loading_by_age_finalised.ipynb
              |_📄 plot_face_recognition.ipynb
              |_📄 Results_and_Tables_Dissertation.ipynb
       |_📁 pipeline
              |_📄 context.py
              |_📄 drift_cda_fedavg.py
              |_📄 face_classification_by_images.py
              |_📄 face_clustering.py
              |_📄 face_statistical_analysis.py
              |_📄 face_verification.py
              |_📄 face_verification_with_similarity.py
              |_📄 face_without_aging.py
              |_📄 face_with_aging.py
              |_📄 face_with_aging_cacd.py
              |_📄 face_with_classifier.py
              |_📄 face_with_T2_Mahalanobis.py
              |_📄 face_with_two_classifiers.py
              |_📄 face_with_ww_mst.py
              |_📄 scatter_plot_agedb_baseline.png
              |_📄 scatter_plot_agedb_facenet.png
              |_📄 __init__.py
       |_📁 preprocessing
              |_📄 facenet.py
       |_📄 constants.yml
       |_📄 dataloaders.py
       |_📄 datasets.py
       |_📄 __init__.py
```

```
|_📂 src
       |_📁 dataset_meta
              |_📄 AgeDB_metadata.mat
```
| fileno | filename |                      name |         age | gender |   |
|-------:|---------:|--------------------------:|------------:|-------:|---|
|      0 |        0 |    0_MariaCallas_35_f.jpg | MariaCallas |     35 | f |
|      1 |    10000 | 10000_GlennClose_62_f.jpg |  GlennClose |     62 | f |
|      2 |    10001 | 10001_GoldieHawn_23_f.jpg |  GoldieHawn |     23 | f |
|      3 |    10002 | 10002_GoldieHawn_24_f.jpg |  GoldieHawn |     24 | f |
|      4 |    10003 | 10003_GoldieHawn_24_f.jpg |  GoldieHawn |     24 | f |
```
|_📂 src
       |_📁 dataset_meta
              |_📄 celebrity2000_meta.mat
```
| age | identity | year | name |       filename |                            |
|----:|---------:|-----:|-----:|---------------:|----------------------------|
|   0 |       53 |    1 | 2004 | Robin_Williams | 53_Robin_Williams_0001.jpg |
|   1 |       53 |    1 | 2004 | Robin_Williams | 53_Robin_Williams_0002.jpg |
|   2 |       53 |    1 | 2004 | Robin_Williams | 53_Robin_Williams_0003.jpg |
|   3 |       53 |    1 | 2004 | Robin_Williams | 53_Robin_Williams_0004.jpg |
|   4 |       53 |    1 | 2004 | Robin_Williams | 53_Robin_Williams_0005.jpg |
```
|_📂 src
       |_📁 dataset_meta
              |_📄 FGNET_metadata.mat
```
| fileno | filename |        age |    |
|-------:|---------:|-----------:|---:|
|      0 |        1 | 001A02.JPG |  2 |
|      1 |        1 | 001A05.JPG |  5 |
|      2 |        1 | 001A08.JPG |  8 |
|      3 |        1 | 001A10.JPG | 10 |
|      4 |        1 | 001A14.JPG | 14 |
```
|_📂 src
       |_📁 models
              |_📄 16.07.2022_two_classifiers.zip
              |_📄 all_ml_models.zip
              |_📄 CACD_MAE_4.59.pth
              |_📄 cvae_face_recognition_model.zip
              |_📄 facenet_keras.h5
              |_📄 mnist_epoch10.hdf5
              |_📄 mnist_epoch2.hdf5
              |_📄 mnist_epoch5.hdf5
              |_📄 vit_face_recognition_model.zip

|_📄 16.07.2022_two_classifiers.zip
       - This zip file contains two Machine Learning models derived from `FaceNetKeras` and `FaceRecognitionBaselineKeras` using the two_classifiers method

|_📄 all_ml_models.zip
       - This zip file contains Machine Learning models derived in the `Age Drifting Scenario` with one model usign randomised ages and another trained with younger faces

|_📄 cvae_face_recognition_model.zip
       - This is a model trained by adding target labels in latent space instead of latent distribution to demonstrate the improvement of accuracy from 60 - 68% to 94%

|_📄 vit_face_recognition_model.zip
       - This is a model trained by adding target labels in latent distribution that shows an accuracy of 94% in ordered target vectors but reduces to 91% in randomised target vectors

|_📄 facenet_keras.h5
       - This is a pre-trained model obtained from FaceNetKeras (original model from FaceNet converted to keras format)

|_📄 mnist_epoch10.hdf5
|_📄 mnist_epoch2.hdf5
|_📄 mnist_epoch5.hdf5
       - These are MNIST based models trained with spochs 2, 5 and 10

|_📂 src
       |_📁 data_collection
              |_📁 16.07.2022_two_classifiers_baseline
              |_📁 16.07.2022_two_classifiers_facenet
```
**facenet_agedb_drift_evaluate_difference.csv**

| fileno | name        | age | gender | filename                   | type     | train_younger | test_younger | difference |
|--------|-------------|-----|--------|----------------------------|----------|---------------|--------------|------------|
| 99     | PaulAnka    | 46  | m.jpg  | 99_PaulAnka_46_m.jpg       | accuracy | 0             | 0            | 0          |
| 102    | PaulAnka    | 60  | m.jpg  | 102_PaulAnka_60_m.jpg      | accuracy | 1             | 1            | 1          |
| 85     | PaulAnka    | 65  | m.jpg  | 85_PaulAnka_65_m.jpg       | accuracy | 1             | 1            | 1          |
| 88     | PaulAnka    | 67  | m.jpg  | 88_PaulAnka_67_m.jpg       | accuracy | 1             | 1            | 1          |
| 90     | PaulAnka    | 70  | m.jpg  | 90_PaulAnka_70_m.jpg       | accuracy | 1             | 0            | 0          |
| 10638  | LuiseRainer | 26  | f.jpg  | 10638_LuiseRainer_26_f.jpg | accuracy | 0             | 1            | 0          |
| 10645  | LuiseRainer | 27  | f.jpg  | 10645_LuiseRainer_27_f.jpg | accuracy | 1             | 1            | 1          |
| 10666  | LuiseRainer | 28  | f.jpg  | 10666_LuiseRainer_28_f.jpg | accuracy | 1             | 1            | 1          |
```
|_📂 src
       |_📁 data_collection
              |_📄 agedb_drift_beta_optimized.csv
```
| filename                   | drift_beta | identity    | drift_difference | euclidean1  | cosine1     | drifted | age1 | gender | hash_sample | offset | covariates_beta | true_identity | y_pred               | proba_pred  | y_drift              | proba_drift | predicted_age | identity_grouping_distance | orig_TP | orig_FN | virtual_TP | virtual_FN | stat_TP | stat_FP | stat_undefined |
|----------------------------|------------|-------------|------------------|-------------|-------------|---------|------|--------|-------------|--------|-----------------|---------------|----------------------|-------------|----------------------|-------------|---------------|----------------------------|---------|---------|------------|------------|---------|---------|----------------|
| 102_PaulAnka_60_m.jpg      | 0.64       | PaulAnka    | 0.040642423      | 12.38562393 | 0.321228981 | 0       | 60   | m      | 1           | 0.01   | 0               | PaulAnka      | PaulAnka             | 0.967190819 | EvaMarieSaint        | 0.926548396 | 6.47          | -42634.44051               | 1       | 0       | 0          | 1          | 0       | 1       | 0              |
| 10638_LuiseRainer_26_f.jpg | 0.46       | LuiseRainer | -0.016787292     | 6.320907116 | 0.769984603 | 1       | 26   | f      | 2           | 0.01   | 0               | LuiseRainer   | AngelaLansbury       | 0.897008418 | ArnoldSchwarzenegger | 0.91379571  | -10.24        | 5942.371164                | 0       | 1       | 0          | 1          | 0       | 0       | 1              |
| 10645_LuiseRainer_27_f.jpg | 0.37       | LuiseRainer | -0.065294556     | 4.653621197 | 0.732229829 | 1       | 27   | f      | 3           | 0.01   | 0               | LuiseRainer   | EdwardGRobinson      | 0.751063295 | ArnoldSchwarzenegger | 0.816357851 | -7.04         | -3210.578365               | 0       | 1       | 1          | 0          | 0       | 0       | 1              |
| 10666_LuiseRainer_28_f.jpg | 0.46       | LuiseRainer | 0.001110673      | 7.824175358 | 0.68356359  | 1       | 28   | f      | 4           | 0.01   | 0               | LuiseRainer   | AngelaLansbury       | 0.868224948 | ArnoldSchwarzenegger | 0.867114275 | -11.09        | -12399.74146               | 0       | 1       | 0          | 1          | 0       | 0       | 1              |
| 10671_LuiseRainer_74_f.jpg | 0.19       | LuiseRainer | 0.005811695      | 1.727426291 | 0.97987169  | 1       | 74   | f      | 5           | 0.01   | 0               | LuiseRainer   | MichaelYork          | 0.792833961 | EdwardGRobinson      | 0.787022266 | 25.08         | 16118.46821                | 0       | 1       | 0          | 1          | 0       | 0       | 1              |
| 10674_LuiseRainer_27_f.jpg | 0.37       | LuiseRainer | -0.011325149     | 4.859529972 | 0.86461252  | 1       | 27   | f      | 6           | 0.01   | 0               | LuiseRainer   | ArnoldSchwarzenegger | 0.865948173 | EdwardFox            | 0.877273322 | -3.79         | -27218.47951               | 0       | 1       | 0          | 1          | 0       | 0       | 1              |
| 10686_LuiseRainer_46_f.jpg | 0.55       | LuiseRainer | 0.049523246      | 9.628829002 | 0.304223418 | 0       | 46   | f      | 7           | 0.01   | 0               | LuiseRainer   | LuiseRainer          | 0.910266346 | FrancesDee           | 0.8607431   | -1.56         | 6679.276062                | 1       | 0       | 0          | 1          | 0       | 1       | 0              |
```
|_📂 src
       |_📁 data_collection
              |_📄 facenet_agedb_drift_synthesis_filename-range-of-beta-10-samples.csv
```
| hash_sample | offset | covariates_beta | drift_beta | true_identity  | age | filename                      | y_pred         | proba_pred | y_drift        | proba_drift | predicted_age | euclidean   | cosine       | identity_grouping_distance | orig_TP | orig_FN | virtual_TP | virtual_FN | stat_TP | stat_FP | stat_undefined |
|-------------|--------|-----------------|------------|----------------|-----|-------------------------------|----------------|------------|----------------|-------------|---------------|-------------|--------------|----------------------------|---------|---------|------------|------------|---------|---------|----------------|
| 1           | 0.01   | 0               | 0.1        | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | AudreyHepburn  | 0.86675855  | 42.91         | 0.878396928 | 0.997328758  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 1           | 0.01   | 0               | 0.19       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | AudreyHepburn  | 0.85362003  | 42.91         | 1.818750858 | 0.988538682  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 1           | 0.01   | 0               | 0.28       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | AudreyHepburn  | 0.85018653  | 42.91         | 2.989679337 | 0.968946576  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 1           | 0.01   | 0               | 0.37       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | AudreyHepburn  | 0.86426846  | 42.91         | 4.985581875 | 0.913693309  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 1           | 0.01   | 0               | 0.46       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | AudreyHepburn  | 0.74805764  | 42.91         | 9.080112457 | 0.675958514  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 1           | 0.01   | 0               | 0.55       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | TomJones       | 0.34733817  | 42.91         | 14.76459122 | -0.133362398 | -2.69E+16                  | 1       | 0       | 0          | 1          | 0       | 1       | 0              |
| 1           | 0.01   | 0               | 0.64       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | AudreyHepburn  | 0.42980481  | 42.91         | 14.58305931 | -0.024875406 | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 1           | 0.01   | 0               | 0.73       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | AudreyHepburn  | 0.43618995  | 42.91         | 14.47703171 | 0.012410657  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 1           | 0.01   | 0               | 0.82       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | TomJones       | 0.30371242  | 42.91         | 14.69083881 | -0.04430135  | -2.69E+16                  | 1       | 0       | 0          | 1          | 0       | 1       | 0              |
| 1           | 0.01   | 0               | 0.91       | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | TomJones       | 0.34865512  | 42.91         | 14.34119606 | -0.090376064 | -2.69E+16                  | 1       | 0       | 0          | 1          | 0       | 1       | 0              |
| 1           | 0.01   | 0               | 1          | AudreyHepburn  | 36  | 14551_AudreyHepburn_36_f.jpg  | AudreyHepburn  | 0.87009716 | TomJones       | 0.21772063  | 42.91         | 13.32721901 | -0.041950963 | -2.69E+16                  | 1       | 0       | 0          | 1          | 0       | 1       | 0              |
| 2           | 0.01   | 0               | 0.1        | AngelaLansbury | 60  | 11971_AngelaLansbury_60_f.jpg | AngelaLansbury | 0.73361039 | AngelaLansbury | 0.73374195  | -17.01        | 0.802545965 | 0.997955918  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 2           | 0.01   | 0               | 0.19       | AngelaLansbury | 60  | 11971_AngelaLansbury_60_f.jpg | AngelaLansbury | 0.73361039 | AngelaLansbury | 0.73584245  | -17.01        | 1.850758791 | 0.989085674  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 2           | 0.01   | 0               | 0.28       | AngelaLansbury | 60  | 11971_AngelaLansbury_60_f.jpg | AngelaLansbury | 0.73361039 | AngelaLansbury | 0.72853002  | -17.01        | 3.292001247 | 0.96531266   | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 2           | 0.01   | 0               | 0.37       | AngelaLansbury | 60  | 11971_AngelaLansbury_60_f.jpg | AngelaLansbury | 0.73361039 | AngelaLansbury | 0.73521995  | -17.01        | 5.371869087 | 0.906560481  | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
| 2           | 0.01   | 0               | 0.46       | AngelaLansbury | 60  | 11971_AngelaLansbury_60_f.jpg | AngelaLansbury | 0.73361039 | AngelaLansbury | 0.45034714  | -17.01        | 13.3840065  | 0.28426224   | -2.69E+16                  | 1       | 0       | 1          | 0          | 1       | 0       | 0              |
```
|_📂 src
       |_📁 data_collection
              |_📄 cda_fedavg_observation.csv
```
| s_f1        | diff1        | T_h         | age | filename |              |    |       | filename                   |
|-------------|--------------|-------------|-----|----------|--------------|----|-------|----------------------------|
| inf         | #NAME?       | 34.53877639 | 8   | 7401     | MickeyRooney | 8  | m.jpg | 7401_MickeyRooney_8_m.jpg  |
| 0           | 0            | 34.53877639 | 10  | 7402     | MickeyRooney | 10 | m.jpg | 7402_MickeyRooney_10_m.jpg |
| inf         | #NAME?       | 34.53877639 | 10  | 6472     | JohnnyCash   | 10 | m.jpg | 6472_JohnnyCash_10_m.jpg   |
| 0           | 0            | 34.53877639 | 13  | 3556     | MichaelYork  | 13 | m.jpg | 3556_MichaelYork_13_m.jpg  |
| 0           | 0            | 34.53877639 | 14  | 15369    | JanePowell   | 14 | f.jpg | 15369_JanePowell_14_f.jpg  |
| 0           | 0            | 34.53877639 | 15  | 15001    | AnnMiller    | 15 | f.jpg | 15001_AnnMiller_15_f.jpg   |
| 0           | 0            | 34.53877639 | 15  | 7098     | GladysCooper | 15 | m.jpg | 7098_GladysCooper_15_m.jpg |
| 16.65070512 | -112.7365628 | 34.53877639 | 15  | 10752    | OrnellaMuti  | 15 | f.jpg | 10752_OrnellaMuti_15_f.jpg |
| 12.72861731 | -465.6143899 | 34.53877639 | 17  | 15003    | AnnMiller    | 17 | f.jpg | 15003_AnnMiller_17_f.jpg   |
```
|_📂 src
       |_📁 data_collection
              |_📄 t2_observation_ucl.csv
```
| fileno | observation | ucl_alpha_0.01 | ucl_alpha_0.02 | ucl_alpha_0.12000000000000001 | ucl_alpha_0.22 | ucl_alpha_0.32000000000000006 | ucl_alpha_0.42000000000000004 | ucl_alpha_0.52 | ucl_alpha_0.6200000000000001 | ucl_alpha_0.7200000000000001 | ucl_alpha_0.8200000000000001 | ucl_alpha_0.92 | filename             | test_younger | train_younger | difference |
|--------|-------------|----------------|----------------|-------------------------------|----------------|-------------------------------|-------------------------------|----------------|------------------------------|------------------------------|------------------------------|----------------|----------------------|--------------|---------------|------------|
| 85     | 105.3619    | 328.8276       | 328.8276       | 328.8276                      | 328.8276       | 328.8276                      | 328.8276                      | 328.8276       | 328.8276                     | 328.8276                     | 328.8276                     | 328.8276       | 85_PaulAnka_65_m.jpg | 1            | 1             | 1          |
| 85     | 105.3619    | 328.8276       | 328.8276       | 328.8276                      | 328.8276       | 328.8276                      | 328.8276                      | 328.8276       | 328.8276                     | 328.8276                     | 328.8276                     | 328.8276       | 85_PaulAnka_65_m.jpg | 1            | 1             | 1          |
| 88     | 112.0468    | 350.9686       | 350.9686       | 350.9686                      | 350.9686       | 350.9686                      | 350.9686                      | 350.9686       | 350.9686                     | 350.9686                     | 350.9686                     | 350.9686       | 88_PaulAnka_67_m.jpg | 1            | 1             | 1          |
| 88     | 112.0468    | 350.9686       | 350.9686       | 350.9686                      | 350.9686       | 350.9686                      | 350.9686                      | 350.9686       | 350.9686                     | 350.9686                     | 350.9686                     | 350.9686       | 88_PaulAnka_67_m.jpg | 1            | 1             | 1          |
| 90     | 131.3238    | 787.0542       | 787.0542       | 787.0542                      | 787.0542       | 787.0542                      | 787.0542                      | 787.0542       | 787.0542                     | 787.0542                     | 787.0542                     | 787.0542       | 90_PaulAnka_70_m.jpg | 0            | 1             | 0          |
| 90     | 131.3238    | 787.0542       | 787.0542       | 787.0542                      | 787.0542       | 787.0542                      | 787.0542                      | 787.0542       | 787.0542                     | 787.0542                     | 787.0542                     | 787.0542       | 90_PaulAnka_70_m.jpg | 0            | 1             | 0          |
| 99     | 86.35297    | 2922.163       | 2922.163       | 2922.163                      | 2922.163       | 2922.163                      | 2922.163                      | 2922.163       | 2922.163                     | 2922.163                     | 2922.163                     | 2922.163       | 99_PaulAnka_46_m.jpg | 0            | 0             | 0          |

# **Introduction**

Concept Drift is a phenomenon through which models decay over time and show ambiguous results on Machine Learning inference. The models may decay because they have used a restricted dataset which may not contain all the necessary feature representations and encodings. Concept Drift is observed in the target labels of the data and occurs due to a change in the underlying data distribution, change in data over time and changes in the predicted output due to a change of methods of data collection. 

# **Motivation**

A paper by (Fernando E. Casado, 2022) et al. describes an algorithm called CDA-FedAvg, a version of the FedAvg algorithm applying the Concept Drift aware algorithm, implemented for activity recognition using smartphones. This is done by simulating the target variables and measuring the real target values from the activity. An evaluation of actual and simulated variables reveals an appropriate metric.

# **Metric of Choice**

![./images/drift_metric.png](./images/drift_metric.png)

_Recall/Accuracy_ is the metric of choice, because Drift detection algorithm can detect False Negatives. 

# **Decision Model**

![./images/decision_model.png](./images/decision_model.png)

# **Existing Datasets**

1. CACD2000

CACD2000 is a large dataset containing face images of celebrities of actors, scientists, etc. It consists of 2000 identities and the age ranges from 14 to 62:

```
Age Unique Values: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
       31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
       48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
```

2. FGNET

FGNET is a small dataset and created in 2007-2008. It consists of 1002 images and 63 identities / subjects and the age ranges from 0 to 69.

```
Age Unique Values: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 58, 60, 61, 62, 63, 67, 69]
```

3. AgeDB

AgeDB is a comparatively larger dataset. It consists of 568 subjects and 16,488 images. The Age ranges from 1 to 101. 

```
Age Unique Values: [  1,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
        15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
        41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
        54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
        67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
        93,  94,  95,  96,  97,  98,  99, 100, 101]
```

# **Project Contents**

The project consists of experiments and pipelines

1. Experiments: Experiments help collect primary data from a model or service

2. Pipelines: Pipelines consist of 1 or more experiments with a logger name prefix and they are executed in a command line environment, with a frontend or programatically


# **Project Implementation**

**Always, cd to `src/pipeline`**

## **How to extract the CDA-FedAvg results**

```
python3.7 .\drift_cda_fedavg.py dataset=agedb model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/AgeDB_metadata.mat no_of_samples=245 tracking_uri="mlruns/" logger_name=drift_cda_fedavg experiment_id=0 model=FaceNetKeras input_shape=-1,160,160,3 data_dir="../../../datasets/AgeDB" drift_synthesis_metrics=../data_collection/16.07.2022_two_classifiers_facenet/facenet_agedb_drift_evaluate_difference.csv 
```

## **How to classify Faces by images**

```
python3.7 face_classification_by_images.py model=YuNet_onnx model_path=../models/face_detection_yunet_2022mar.onnx no_of_samples=1288 logger_name=face_classification_by_images tracking_uri="mlruns/" data_dir=../../../datasets/AgeDB experiment_id=3
```

## **How to cluster face images and send the results to classification pipeline**

```
python3.7 .\face_clustering.py dataset=agedb model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/AgeDB_metadata.mat no_of_samples=2598 tracking_uri="mlruns/" logger_name=facenet_model_clustering collect_for=classification experiment_id=0 classifier_test_younger=../models/facenet_agedb_voting_classifier_age_test_younger.pkl model=FaceNetKeras input_shape=-1,160,160,3 data_dir="../../../datasets/AgeDB" drift_evaluate_metrics_test_younger=../data_collection/facenet_agedb_drift_evaluate_metrics_test_younger.csv drift_evaluate_metrics_train_younger=../data_collection/facenet_agedb_drift_evaluate_metrics_train_younger.csv classifier_train_younger=../models/facenet_agedb_voting_classifier_age_train_younger.pkl classifier_test_younger=../models/facenet_agedb_voting_classifier_age_test_younger.pkl drift_evaluate_metrics=../data_collection/facenet_agedb_drift_evaluate_metrics_clustering.csv eps=11.0 min_samples=1 experiment_id=0 min_samples=16 eps=2.5
```

## **How to conduct statistical analysis and PSNR / MSE simulation over drift beta**

```
python3.7 .\face_statistical_analysis.py dataset=agedb model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/AgeDB_metadata.mat no_of_samples=188 no_of_pca_samples=188 pca_covariates_pkl=../data_collection/agedb_pca_covariates.pkl grouping_distance_type=DISTINCT tracking_uri="mlruns/" logger_name=facenet_statistical_analysis classifier=../models/agedb_voting_classifier_age.pkl experiment_id=1 pca_type=KernelPCA noise_error=0 image_error=0 drift_type=incremental drift_beta=2 inference_images_pkl=../data_collection/agedb_inferences_facenet.pkl drift_synthesis_metrics=../data_collection/agedb_drift_beta_optimized.csv data_dir=../../../datasets/AgeDB input_shape=-1,160,160,3 model=FaceNetKeras function_type=morph denoise_type=opencv_denoising
```

## **How to perform face verification with similarity**

```
python3.7 .\face_verification_with_similarity.py dataset=agedb model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/AgeDB_metadata.mat unique_name_count=30 no_of_samples=14157 tracking_uri="mlruns/" logger_name=facenet_without_aging_keras input_shape=-1,160,160,3 data_dir="../../../datasets/AgeDB" model=FaceNetKeras source_type=file experiment_id=0 data_dir=../../../datasets/AgeDB
```

## **How to perform face verification using perceptrons**

```
python3.7 .\face_verification.py dataset=agedb model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/AgeDB_metadata.mat unique_name_count=30 no_of_samples=14157 tracking_uri="mlruns/" logger_name=facenet_without_aging_keras experiment_id=2 input_shape=-1,160,160,3 data_dir="../../../datasets/AgeDB" data_collection_pkl=../data_collection/detection_agedb_inferences_baseline_cvae_9k.pkl model=FaceNetKeras source_type=ordered_metadata experiment_id=0 data_dir=../../../datasets/AgeDB face_id=25
```

## **How to create Face Aging concept drift `Drift table` using Statistical Face Model by PCA**

```
python3.7 .\facenet_with_aging_cacd.py dataset=cacd model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/celebrity2000_meta.mat no_of_samples=320 no_of_pca_samples=320 grouping_distance_type=DISTINCT tracking_uri="mlruns/" logger_name=facenet_with_aging classifier=../models/facenet_agedb_voting_classifier_age_train_younger_latest_3.pkl experiment_id=0 pca_type=KernelPCA noise_error=0 mode=image_reconstruction drift_beta=1.0 covariates_beta=0 data_dir=../../../datasets/CACD2000_processed drift_synthesis_filename=../data_collection/facenet_cacd_drift_synthesis_filename-range-of-beta-latest-1.csv drift_source_filename=../data_collection/facenet_cacd_drift_evaluate_metrics_difference.csv model=FaceNetKeras input_shape=-1,160,160,3 function_type=morph drift_type=incremental
```

## **How to create Voting Classifier Models using `Age Drifting` scenario and `Non Age Drifting Scenario` (Classification)**

```
python3.7 .\facenet_with_two_classifiers.py dataset=cacd model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/celebrity2000_meta.mat tracking_uri="mlruns/" logger_name=facenet_model_with_two_classifiers_latest_3 collect_for=age_drifting experiment_id=0 classifier_test_younger=../models/facenet_cacd_voting_classifier_age_test_younger_latest_3.pkl model=FaceNetKeras input_shape=-1,160,160,3 data_dir="../../../datasets/CACD2000_processed" drift_evaluate_metrics_test_younger=../data_collection/facenet_cacd_drift_evaluate_metrics_test_younger_latest_3.csv drift_evaluate_metrics_train_younger=../data_collection/facenet_cacd_drift_evaluate_metrics_train_younger_latest_3.csv classifier_train_younger=../models/facenet_cacd_voting_classifier_age_train_younger_latest_3.pkl no_of_samples=15000 
```

## **How to conduct Mahalanobis T2 Statistical Analysis**

```
python3.7 .\facenet_with_T2_Mahalanobis.py dataset=agedb model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/AgeDB_metadata.mat no_of_samples=239 tracking_uri="mlruns/" logger_name=facenet_model_with_T2_Mahalanobis_latest collect_for=age_drifting experiment_id=0 model=FaceNetKeras input_shape=-1,160,160,3 data_dir="../../../datasets/AgeDB" drift_synthesis_metrics=../data_collection/16.07.2022_two_classifiers_facenet/facenet_agedb_drift_evaluate_difference.csv t2_observation_ucl=../data_collection/t2_observation_ucl.csv
```

## **How to conduct Runs Test with face images using MST (Minimum Spanning Tree)**

```
python3.7 .\facenet_with_ww_mst.py dataset=agedb model_path=../models/facenet_keras.h5 batch_size=128 metadata=../dataset_meta/AgeDB_metadata.mat no_of_samples=245 tracking_uri="mlruns/" logger_name=facenet_model_with_ww_mst_latest collect_for=age_drifting experiment_id=2 model=FaceNetKeras input_shape=-1,160,160,3 data_dir="../../../datasets/AgeDB" drift_synthesis_metrics=../data_collection/16.07.2022_two_classifiers_facenet/facenet_agedb_drift_evaluate_metrics_train_younger_late_Copy.csv 
```

