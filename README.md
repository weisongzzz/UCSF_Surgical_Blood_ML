# UCSF Surgical Blood Product Recommendation System
Leveraging machine learning to provide real-time recommendations for blood product and test ordering in non-emergency surgeries at UCSF, reducing waste and operational costs.

## Overview

## Key Features
- Multi-Class Prediction: Generates probabilities for the need of RBC, FFP, and Platelet transfusions.

- Actionable Recommendations: Translates model outputs into clear, actionable recommendations for clinicians.

- Operational Cost Reduction: Aims to mitigate waste of blood products and associated costs.

## Getting Started

## Installation:

git clone [your_repo_link]

cd [your_repo_folder]

conda env create -f environment.yml

conda activate [your_env_name]

Usage: Explain the core scripts and their purpose.

## Model Training & Development:

## Data Analysis & Visualization:

## Utilities:

### environment.yml
run it in the terminal to set up all packages for the Python environment.

### build_hipac_model.py  
 (build_hipac_model_gridsearchCV performs gridsearch and saves the best model)
- Split the retrospective dataset into training, testing, and validation sets
- Perform feature transformation and train a default XGB classifier on the retrospective training set
- Save the split data (untransformed), trained model, feature transformer, and threshold to pickle files
- Train the multi-classification XGB model and save the model & features

### multi.py  
Playground for the multi-classification model

### get_recomm.py
- Create a statistics table (masterT.csv, includes model thresholds, sensitivity, and PPV).
- Generate multi-classification labels for a given data split as recommendations for downstream implementation. 

-- 
## data exploration & visualization

### table1_wei.py
- Generate table 1 (summary statistics for group comparisons for selected features)

### performance_curves.py
- Plot model performance curves with de Long based confidence intervals (ROC, Precision-Recall, and Calibration).
- Calculate and output model performance metrics: sensitivity, specificity, positive predictive value, negative predictive value, accuracy, and other statistics related to the performance curves

### feature_ranking.py
- Create the figure of Shapley value plots that rank the feature importance

### multiclass_model_2025_output_analysis.py
- Create a dataframe that combines essential features and outputs associated with the model for analysis

### lhs report/order_request_cleanup_all_blood_products.ipynb
- Data check on clinician PRBC requests and T&S orders

### lhs report/timetrend_visualization-no-prosp.ipynb
- Sensitivity and PPV performance between model and clinician history overtime comparison 

### prospective.ipynb (to be uploaded)
- Visualize HIPAC model performance metrics overtime (saved from dashboard)

### silent_study_timepoint.ipynb
- Visualize and compare model performance at the four timepoints (1hr, 3pm day before, 1 week, 1 month) prior to the reference (anesthesia start)

### MSBOS_files (mostly unused except for)
1. MSBOS_05d_Performance_GBM_Bias_Metrics_V2.py)
- Generate and save race-ethnicity based thresholds
- Calculate performance metrics and generate performance curves with group-specific thresholds are applied
2. MSBOS_02_a_DataCharacteristics.py
- Report baseline characteristics of the data

-- 
## Utilities:

### MSBOS_06_Analysis_Tools.py  
- Functions for generating performance curves and calculating performance metrics (sensitivity, specificity, PPV, NPV, etc.)

### hipac_ml_msbos module  
- Functions for splitting datasets and performing feature transformations.
1. hipac_modeling_tools_mono: only uses the feature historic transfusion rate
2. hipac_modeling_tools_old: for paper, the model depoyed on HIPAC a.t.m. (202409)
3. hipac_modeling_tools_reduced: using Sunny et al. features
4. hipac_modeling_tools: iteration 2 model with data prior to 202409
5. hipac_modeling_tools: iteration 2 model with data after 202409

## Contributors 
