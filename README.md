# Urinary Biomarkers for Pancreatic Cancer Analysis

## Project Overview:

This project focuses on analyzing urinary biomarkers to predict pancreatic cancer diagnoses using machine learning techniques. The dataset includes biomarker values and clinical information for patients, enabling the identification of significant predictors and the development of classification models.

# Dataset

The dataset, sourced from Debernardi et al. (2020), https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer contains the following key columns:

sample_id: Unique identifier for each sample.

patient_cohort: Indicates the cohort (e.g., previously used samples or new samples).

sample_origin: Source of the sample.

age: Patient's age.

sex: Patient's gender (M/F).

diagnosis: Diagnostic category (1=Control, 2=Benign, 3=PDAC).

stage: Cancer stage (for applicable cases).

plasma_CA19_9: Plasma biomarker levels.

creatinine: Biomarker for kidney function.

LYVE1, REG1B, TFF1, REG1A: Urinary biomarkers potentially linked to pancreatic cancer.

# Data Preprocessing:

Handled missing values using mean imputation for numerical features and categorical encoding for the "sex" variable.

Dropped irrelevant or highly incomplete columns, such as "REG1A" and "stage."

Standardized numerical features for consistency.

# Exploratory Data Analysis

Correlation Analysis: Identified key features, such as LYVE1 and TFF1, with the highest correlation to the target variable (diagnosis).

Visualizations: Histograms, box plots, scatter plots, and a heatmap of the correlation matrix provided insights into data distribution and relationships.

# Machine Learning Models

Logistic Regression:

Achieved an accuracy of 63% with moderate precision and recall across diagnostic categories.

Random Forest Classifier

Improved performance with an accuracy of 64% and better recall for the PDAC category.

Confusion Matrix:

[[21 12  4]
 [14 19  8]
 [ 3  2 35]]

Classification Report:

              precision    recall  f1-score   support

          1       0.55      0.57      0.56        37
          2       0.58      0.46      0.51        41
          3       0.74      0.88      0.80        40

   accuracy                           0.64       118
  macro avg       0.62      0.64      0.63       118

weighted avg       0.63      0.64      0.63       118

- Feature importance analysis confirmed LYVE1, TFF1, and REG1B as key predictors.

### Feature Engineering
- Numerical features standardized using `StandardScaler`.
- Categorical feature ("sex") one-hot encoded to ensure compatibility with machine learning models.

## Key Findings
1. **LYVE1**: Strongest predictor of pancreatic cancer diagnosis.
2. **TFF1 and REG1B**: Moderately strong predictors.
3. **Creatinine**: Weak correlation with diagnosis, indicating limited predictive power.

## Visualizations
- Histograms and box plots for numerical features.
- Pair plots to visualize relationships between biomarkers and the target variable.
- Correlation heatmap to identify feature relationships.

## Instructions for Running the Code
1. Clone this repository:
 ```bash
 git clone <repository-url>
 cd <repository-folder>

Install required dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook or Python script:

jupyter notebook analysis.ipynb

Preprocessed data and visualizations will be generated in the output directory.

Future Work

Experiment with additional models like XGBoost and LightGBM.

Perform hyperparameter tuning for optimal model performance.

Explore advanced imputation techniques for missing data.

Incorporate SMOTE to address potential class imbalance.

File Structure

|-- data/
|   |-- Debernardi_et_al_2020_data.csv
|-- notebooks/
|   |-- analysis.ipynb
|-- scripts/
|   |-- preprocess.py
|   |-- model_train.py
|-- results/
|   |-- visualizations/
|   |-- model_evaluations/
|-- README.md

References

Debernardi, et al. (2020). "Urinary Biomarkers for Pancreatic Cancer." [https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer]


