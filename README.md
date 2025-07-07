# All_Tasks

# Summer Analytics 2025 Capstone Project: Dynamic Pricing and Data Analysis

This repository contains five Python code files developed for the Summer Analytics 2025 Capstone Project, hosted by the Consulting & Analytics Club × Pathway. The project encompasses data analysis, machine learning, and dynamic pricing for urban parking lots, as well as classification tasks using various datasets. The codes demonstrate exploratory data analysis, deep learning, logistic regression, health data classification, and real-time dynamic pricing with streaming data. Below is a comprehensive description of each code file, their objectives, functionalities, and instructions for execution.

## Prerequisites

To run the codes, ensure the following requirements are met:

- **Environment**: Google Colab is recommended due to its support for visualization and deep learning dependencies. Alternatively, a local Python environment with Python 3.8+ can be used.
- **Dependencies**: Install the required libraries using the following command:
  ```bash
  !pip install pandas numpy matplotlib seaborn scikit-learn tensorflow pathway bokeh panel xgboost imblearn scipy pillow --quiet
  ```
- **Datasets**: Ensure the required datasets (Cars.csv, hacktrain.csv, hacktest.csv, Train_Data.csv, Test_Data.csv, dataset.csv) are available in the working directory.

## Code Descriptions

### Code 1: Data Analysis and Visualization (Cars.csv)
**Objective**: Perform exploratory data analysis (EDA) and visualization on the Cars.csv dataset to derive insights about car attributes such as MPG, horsepower, weight, and origin.

**Key Features**: Data Preprocessing: Loads Cars.csv, sets indices (name, horsepower_per_weight), and computes a new feature (horsepower_per_weight).
 Analysis Tasks:
- Identifies the car with the highest horsepower.
- Counts cars with MPG ≥ 35.
- Finds the most common origin for cars with horsepower > 100 and weight < 3000.
- Calculates the mean acceleration for Japanese cars.
- Determines the model year with the highest average MPG.
- Finds the car with the best horsepower-to-weight ratio among cars with above-median MPG.
  
**Visualizations**:
- Line plot of average MPG by model year and origin.
- Scatter plot of horsepower vs. weight, colored by origin and sized by MPG.
- Analysis of car models appearing in multiple years with consistent MPG (std < 1.0), sorted by appearances and MPG.

**Libraries**: pandas, numpy, matplotlib, seaborn.

### Code 2: Deep Learning Classification (`hacktrain.csv`, `hacktest.csv`)

**File**: `code2_deep_learning.py`

**Objective**: Build and evaluate deep learning models (BiGRU and CNN) for multi-class classification on the `hacktrain.csv` dataset, generating predictions for `hacktest.csv`.

**Key Features**:
- **Data Preprocessing**:
  - Loads `hacktrain.csv` and imputes missing values using column means.
  - Drops the `ID` column to focus on relevant features.
  - Encodes the target `class` column using `LabelEncoder` and converts to one-hot encoding with `to_categorical`.
  - Scales features using `StandardScaler` and reshapes data into 3D format (samples, timesteps, features) for compatibility with 1D input requirements of GRU and CNN models.
- **Models**:
  - **BiGRU**: Consists of two bidirectional GRU layers (64 and 32 units) with dropout (0.3) to prevent overfitting, followed by a dense layer (64 units, ReLU activation) and a softmax output layer for multi-class classification. Variants include enhanced models with additional GRU layers (128, 64 units), batch normalization, layer normalization, and L2 regularization.
  - **CNN**: Includes two convolutional layers (64 and 128 filters, kernel size 3, ReLU activation) with max pooling (pool size 2), followed by a flatten layer, a dense layer (128 units, ReLU, dropout 0.5), and a softmax output layer.
- **Training**:
  - Employs 5-fold stratified cross-validation to ensure balanced class distribution across folds.
  - Uses early stopping (patience=5 or 15) to halt training when validation loss stops improving, restoring the best weights.
  - Some variants include learning rate reduction (factor=0.5, min_lr=1e-6) for better convergence.
  - Models are compiled with the Adam optimizer (learning rate=0.0002 to 0.001) and categorical cross-entropy loss.
- **Evaluation**:
  - Generates classification reports for validation folds, detailing precision, recall, and F1-score for each class.
  - Predicts class labels for `hacktest.csv` using the trained model.
- **Outputs**:
  - Saves test predictions as `bigru.csv`, `cnn.csv`, `bigru_1.csv`, `improved_bigru.csv`, and `submission_bigru.csv`.
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `scipy`.


### Code 3: Logistic Regression with Feature Engineering (`hacktrain.csv`, `hacktest.csv`)

**File**: `code3_logistic_regression.py`

**Objective**: Perform multi-class classification using logistic regression with advanced feature engineering on the `hacktrain.csv` dataset, predicting classes for `hacktest.csv`.

**Key Features**:
- **Data Preprocessing**:
  - Loads `hacktrain.csv` and applies Savitzky-Golay smoothing (window_length=5, polyorder=2) to numeric columns for noise reduction.
  - Imputes missing values using `KNNImputer` (n_neighbors=3 or 5).
  - Drops the `ID` column to focus on relevant features.
- **Feature Engineering**:
  - Identifies NDVI columns (containing `_N`) and performs temporal interpolation using `interpolate` (limit_direction='both').
  - Applies Savitzky-Golay smoothing to NDVI columns.
  - Extracts features: `mean_ndvi` (mean of NDVI columns), `std_ndvi` (standard deviation), `ndvi_amp` (max - min), and seasonal means (spring: first 7 columns, summer: next 7, fall: next 7, winter: remaining).
  - Computes `trend` using linear regression slope on numeric columns.
- **Model**:
  - Uses logistic regression with L2 regularization (C=0.1 or 0.5) in a pipeline with `KNNImputer`, `StandardScaler`, and `SelectKBest` (k=20 features, f_classif scoring).
- **Training**:
  - Employs 5-fold stratified cross-validation to ensure balanced class distribution across folds.
  - Prints classification reports for each fold, detailing precision, recall, and F1-score.
- **Outputs**:
  - Saves test predictions as `lr_submission.csv` and `submission_lr.csv`.
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `scipy`.

**How to Run**:
1. Place `hacktrain.csv` and `hacktest.csv` in the working directory.
2. Run the script in a Python environment or Google Colab.
3. Outputs include classification reports for validation folds and prediction CSV files.

### Code 4: Health Data Classification (`Train_Data.csv`, `Test_Data.csv`)

**File**: `code4_health_classification.py`

**Objective**: Analyze and classify age groups (`Adult`, `Senior`) in `Train_Data.csv` using logistic regression, XGBoost, and Random Forest, predicting for `Test_Data.csv`.

**Key Features**:
- **EDA**:
  - Visualizes distributions, correlations, and relationships for features like `BMXBMI`, `LBXGLU`, `LBXIN`, and `age_group` using histograms, box plots, heatmaps, count plots, violin plots, scatter plots, and KDE plots.
- **Feature Engineering**:
  - Creates features: `BMI_cat` (categorized as Underweight, Normal, Overweight, Obese), `Glucose_Insulin_Ratio` (LBXGLU / (LBXIN + 1e-5)), `High_Glucose` (LBXGLU > 100), and `BMI_Glucose_Interaction` (BMXBMI * LBXGLU).
- **Preprocessing**:
  - Imputes missing values using `KNNImputer` (n_neighbors=5) for numerical features and mode (`SimpleImputer`) for categorical features.
  - Scales numerical features with `RobustScaler` or `StandardScaler`.
  - Handles missing target values (`age_group`) using logistic regression imputation.
  - Converts categorical features (`RIAGENDR`, `PAQ605`, `DIQ010`) to numeric and ensures consistent data types.
- **Models**:
  - **Logistic Regression**: Used for imputing missing target values, with balanced class weights.
  - **XGBoost**: Tuned with `GridSearchCV` (parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) and `scale_pos_weight` for class imbalance.
  - **Random Forest**: Tuned with `GridSearchCV` (parameters: `n_estimators`, `max_depth`, `min_samples_split`) with balanced class weights.
  - Applies `SMOTE` for class imbalance and optimizes thresholds using precision-recall curves for better F1 scores.
- **Evaluation**:
  - Reports ROC AUC scores, classification reports, and visualizes feature importance using bar plots.
- **Outputs**:
  - Saves test predictions as `submission.csv` and `submission1.csv`.
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imblearn`.

**How to Run**:
1. Place `Train_Data.csv` and `Test_Data.csv` in the working directory.
2. Run the script in Google Colab (preferred for visualization and model training).
3. Outputs include visualizations, classification reports, and prediction CSV files.

### Code 5: Dynamic Parking Pricing (`dataset.csv`)

**File**: `code5_dynamic_pricing.py`

**Objective**: Implement a baseline dynamic pricing model for urban parking lots using real-time data streaming with Pathway and visualization with Bokeh.

**Key Features**:
- **Data Preprocessing**:
  - Loads `dataset.csv` and combines `LastUpdatedDate` and `LastUpdatedTime` into a `Timestamp` column.
  - Saves relevant columns (`Timestamp`, `Occupancy`, `Capacity`) to `parking_stream.csv` for streaming.
- **Streaming**:
  - Uses Pathway’s `replay_csv` to simulate real-time data ingestion at 1000 rows/second.
  - Parses timestamps and extracts the day for daily aggregation using tumbling windows.
- **Pricing Model**:
  - Implements a baseline linear pricing formula: `price = 10 + (occ_max - occ_min) / cap`, where `occ_max` and `occ_min` are the maximum and minimum occupancy in a daily window, and `cap` is the capacity.
- **Visualization**:
  - Uses Bokeh to create a line plot with data points for daily parking prices, served as an interactive web app via Panel.
- **Libraries**: `pandas`, `numpy`, `pathway`, `bokeh`, `panel`, `PIL`.
- **Note**: This is a simplified version of the dynamic pricing model. For a comprehensive implementation including Baseline Linear, Demand-Based, and Competitive Pricing models (using features like `QueueLength`, `TrafficConditionNearby`, `VehicleType`, `IsSpecialDay`, and competitor pricing), refer to the updated project submission notebook.

**How to Run**:
1. Place `dataset.csv` in the working directory.
2. Run the script in Google Colab.
3. Outputs a Bokeh plot displaying daily parking prices, served as an interactive web app.


### **Dataset Dependency**: Each code requires specific datasets. Ensure they are correctly named and placed in the working directory.

### **Hardware**: Some deep learning models (Code 2) may benefit from GPU acceleration in Colab.

### **Customization**: The codes can be extended with additional features or models as needed for specific project requirements.

## **Acknowledgments**

This project is part of the Summer Analytics 2025 Capstone, hosted by the Consulting & Analytics Club × Pathway. The codes demonstrate data analysis, machine learning, and real-time data processing skills tailored to the project's objectives.
