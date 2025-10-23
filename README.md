# New-York-City-Taxi-Trip-Duration

---

## overview
This repository is based on the **NYC Trip Duration Prediction Kaggle project**.
The main objective of this project is to **predict the trip duration using machine learning techniques**.

I used **Ridge Regression (alpha = 1.0)** as the main model and performed extensive feature engineering to create new meaningful features from the raw data.
A **log transformation** was applied to both the **trip_duration (target variable) and the distance-related features**, which had a **significant positive impact** on model performance. This transformation greatly improved the model’s generalization and resulted in an **R² score of 0.6040** on the validation dataset.

I also experimented with **polynomial feature transformations** to explore potential nonlinear relationships between features. However, these transformations **did not lead to any improvement in performance**.

---

## About data(raw features)
<img width="1117" height="627" alt="Screenshot 2025-10-23 212040" src="https://github.com/user-attachments/assets/678b94bf-e70b-45d1-8d66-6d63a79627b1" />

---
## Install Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt

```
---

## Repo structure 

```bash

│   .gitignore
│   Feature_engineering.py
│   NYC_Trip_duration_EDA.ipynb
│   NYC_Trip_duration_prediction_report.pdf
│   Predict.py
│   README.md
│   requirements.txt
│   Trip_duration_data_utils.py
│   Trip_duration_train.py
│
├───.best_model
│       feature_columns.json
│       ridge_model.joblib
│       scaler.joblib
│
├───.data
│       test.zip
│       train.zip
│       val.zip
│
└───.polynomail_transformation_code_version
        Feature_engineering.py
        predict.py
        Trip_duration_data_utils.py
        Trip_duration_train.py


```
```.gitignore```: Git configuration file for line endings and file attribute settings.

```Feature_engineering.py```: Contains feature engineering functions such as distance features, time-based feature extraction, and transformations.

```NYC_Trip_duration_EDA.ipynb```: Exploratory Data Analysis  notebook — includes data visualization, correlation analysis, and insights about trip duration patterns..

```NYC_Trip_duration_prediction_report.pdf```: Final project report summarizing methodology, feature engineering, model, results, and conclusions.

```predict.py```: for running inference on new trip data using the trained model and scaler.

```README.md```: Documentation file (this file).

```requirements.txt```: Lists all Python package dependencies required to reproduce the project environment..

```Trip_duration_data_utils.py```: Contains data preprocessing utilities (scaling, saving, splitting, loading).

```Trip_duration_train.py```: Main training script — trains Ridge regression models, saves best models, and evaluates performance.

```.best_model/```
- **feature_columns.json** : JSON file storing feature names used during training for consistent preprocessing in inference.
- **ridge_model.joblib**: Serialized Ridge Regression model saved using joblib.
- **scaler.joblib	Scaler** : Scaler used to normalize numerical features before model training and inference.


```.data/```
Folder containing the dataset in zipped format.

- **train.csv**: Training set
- **val.csv**: Validation set
- **test.csv**: Test set


```.polynomail_transformation_code_version/```
Folder containing the dataset in zipped format.

- **Feature_engineering.py**: Alternate version of the feature engineering module including polynomial feature transformations.
- **predic.py**: Inference script adapted for polynomial-transformed features.
- **Trip_duration_data_utils.py**: TData utilities adjusted for the polynomial feature workflow.
- **Trip_duration_train.py**: Training script for the polynomial feature version of the model, used to compare performance improvements.


---
## How to run 
```bash
python Trip_duration_train.py \
  --train_path path/to/train.csv \
  --val_path path/to/val.csv \
  --alpha 1.0 \
  --scaler MinMaxscaler \
  --target_col "log_trip_duration" \
```

```--train_path```: path to the training dataset(.csv).

```--val_path```:  path to the validation dataset(.csv).

```--scaler```: Choose a scaling method: standard or minmax.

```---alpha```: Regularization strength for Ridge Regression (best = 1.0).

```--target_col```: Name of the target column (default: log_trip_duration).

**2- Run predictions with a saved model (inference phase)**

```bash
python Predict.py \
  --data_path "path/to/test.csv" \
  --model_path ".best_model/ridge_model.joblib" \
  --scaler_path ".best_model/scaler.joblib" \
  --columns_path ".best_model/feature_columns.json" \
  --output_path "outputs/predictions.csv"

  
  

```
```--data_path```: Path to the test dataset (.csv).

```--model_path```: Path to the trained model file (.joblib).

```--scaler_path```: Path to the saved scaler used during training.

```---columns_path```: JSON file containing feature column names used during training.

```--target_col```: (Optional) Target column name for evaluation if available.

```--no_feature_engineering```: Use this flag to skip feature engineering if test data is already processed.

```--output_path```: (Optional) Path to save predictions as a .csv file.

---

## final model features

**1- Temporal Feature Extraction**

- **start_hour**: hour of the day (0–23)

- **day_of_week**: day index (0 = Monday, …, 6 = Sunday)

-**weekend_day**: binary flag (1 if Saturday/Sunday, else 0)

- **month**: calendar month (1–12)

- Created a categorical feature **which_part_of_day** 

**2- Distance feature**: haversine_distance

**3- Log transformation**: applied to trip_duration and haversine_distance

**4 - Outliers**: no clipping or removal applied to any features

**5- Polynomial transformation**: not applied

---

## Conclusion (learning lessons)
1- **Clipping or removing outliers** in this project is **useless**, as it has no positive impact and, in some cases, even harms performance. 

2- I made a big **mistake** by trying to add the **speed feature directly using the trip_duration column**. As we know, **speed = distance/time**.
This approach improved the model’s performance, reaching an R² score of 0.95. However, it introduces **data leakage**, since trip_duration is the target variable being predicted. Moreover, it doesn’t make sense — during inference, the trip_duration value is unknown, so the model wouldn’t have access to it.

3 - I also tried to add speed through a speed model and historical speed using other features like which_part_of_day and start_hour, along with many other attempts. All of them were **useless**. There was no impact on the performance.

4- Conducting **EDA** is essential to understand the distribution and behavior of the features.

5- **Exploring other models in future work could provide further improvements.**

6- Applying **polynomial regression (transformation)** in this project **did not have a significant impact** as expected theoretically, though it may be more beneficial in other contexts.

---

## result 

<img width="721" height="332" alt="image" src="https://github.com/user-attachments/assets/e70486dd-bd61-45c1-961c-4e96f76864a6" />

---

## correlation matrix

<img width="1097" height="736" alt="image" src="https://github.com/user-attachments/assets/07d4df7f-27cf-40c8-93bf-476b6a16f0e1" />


<img width="783" height="492" alt="image" src="https://github.com/user-attachments/assets/3fc494a6-3e77-4c05-9e7c-914a23241d99" />
