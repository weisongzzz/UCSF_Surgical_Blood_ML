# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:22:55 2025

@author: wzhang10
"""

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_recall_curve,precision_score,recall_score
from sklearn.multioutput import MultiOutputClassifier
from MSBOS_06_Analysis_Tools import stat_metrics2, stat_metrics3
import shap
import re
from scipy.stats import norm
rae_folder = 'N:' # irb folder
exclude = 1 #flag, if set to 1 exclude specialties
model_name = ''

#%% get threshold
from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_recall_curve,precision_score,recall_score
import pickle
rae_folder = 'N:' # irb folder
np.random.seed(123)
with open(f"{rae_folder}/Models/20250116_wei_multi_final.pkl", 'rb') as f:
    multi_target_model, feature_transformer, thre_dict = pickle.load(f)
with open(f"{rae_folder}/Data/20250116/train_test_elective_wei_123_multi_final.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)
#%%
#df_pre = pd.read_csv(f"{rae_folder}/Data/20250325/model_clinician_analysis_data_040725.csv")
#df = df_pre.loc[:, 'sched_surgeon_cnt':'cci_dialysis'].drop('picis_codes', axis=1, errors='ignore')

df = X_valid.loc[:, 'sched_est_case_length':].drop('picis_codes', axis=1, errors='ignore')
#%%
# Show column names and their data types
print("\nColumn names and data types:")
for col in df.columns:
    # Get unique values
    n_unique = df[col].nunique()
    
    # Determine variable type
    if df[col].dtype in ['int64', 'float64']:
        if n_unique <= 2:
            var_type = "Binary"
        elif n_unique <= 10:
            var_type = "Categorical (Numeric)"
        else:
            var_type = "Continuous"
    elif df[col].dtype == 'object':
        if n_unique <= 2:
            var_type = "Binary (String)"
        else:
            var_type = "Categorical (String)"
    else:
        var_type = "Other"
    
    print(f"{col}: {var_type} (dtype: {df[col].dtype}, unique values: {n_unique})")

#%% Predict on the validation set
X_valid = feature_transformer.transform(X_valid)[feature_transformer.features]

y_valid_pred = multi_target_model.predict_proba(X_valid)

#%%
sensi_target = 0.967908
# check thresholds for each product
y_valid['ffp'] = y_valid.periop_ffp_units_transfused > 0
y_valid['prbc'] = y_valid.periop_prbc_units_transfused > 0
y_valid['platelet'] = y_valid.periop_platelets_units_transfused > 0

y_valid_multi = y_valid[['ffp', 'prbc', 'platelet']]
# Convert pandas DataFrames to numpy arrays if they are DataFrames
if isinstance(y_valid_multi, pd.DataFrame):
    y_valid_multi = y_valid_multi.to_numpy()

#%%    
num_labels = y_valid_multi.shape[1]
label_names = ['ffp', 'prbc', 'platelet'] 
thre_dict = {}
for i in range(num_labels):
    precision, recall, thresholds = precision_recall_curve(
        y_valid_multi[:, i], y_valid_pred[i][:, 1])
    j = np.argmin(np.abs(recall - sensi_target))
    # Store the threshold in the dictionary
    thre_dict[label_names[i]] = thresholds[j]
    print(f'Label {i}:')
    print(f'Precision[{j}]: {precision[j]}')
    print(f'Recall: {recall}')
    print(f'Thresholds[{j}]: {thresholds[j]}')
print(thre_dict)

# %% monthly prediction
# Reload model and data
#with open(f"{rae_folder}/Models/elective_only_20230828_thresholds_corrected_lhs_olddat_exclude_pt74.pkl", 'rb') as f:
    #optimal_threshold = pickle.load(f)
with open(f"{rae_folder}/Models/20250116_wei_multi_final.pkl", 'rb') as f:
    multi_target_model, feature_transformer, thre_dict = pickle.load(f)
with open(f"{rae_folder}/Data/20250116/train_test_elective_wei_123_multi_final.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)
threshold_name = 'Colin'
# New values to update
new_thres = [0.1732, 0.1444, 0.0484]

# Update the dictionary
thre_dict['ffp'], thre_dict['prbc'], thre_dict['platelet'] = new_thres

# Transform and predict
X_train['Data Split'] = 'train'
X_valid['Data Split'] = 'valid'
X_test['Data Split'] = 'test'

X = pd.concat([X_train,X_valid,X_test], axis=0).reset_index(drop=True).copy()
X_copy = X[['msbos_ts','msbos_tc','msbos_prbc','Data Split','scheduled_for_dttm']].copy() #variables that are needed later

y_full_msbos = X['msbos_prbc'].fillna(0) > 0
y_full_ts = X['msbos_ts'].fillna(0) > 0
y_full_tc = X['msbos_tc'].fillna(0) > 0

X = feature_transformer.transform(X)[feature_transformer.features]
#%%
pred_prob = multi_target_model.predict_proba(X)   # ['ffp', 'prbc', 'platelet']  [0, 1, 2]
# Extract the second column (index 1) from each array in pred_prob and assign it to new columns in X
X[['pred_prob_ffp', 'pred_prob_prbc', 'pred_prob_platelet']] = pd.DataFrame({
    'pred_prob_ffp': pred_prob[0][:, 1],
    'pred_prob_prbc': pred_prob[1][:, 1],
    'pred_prob_platelet': pred_prob[2][:, 1]
})
#%%
# Add month and data split columns
X['Month'] = X_copy['scheduled_for_dttm'].dt.to_period('M').astype(str)
X['Data Split'] = X_copy['Data Split']

# Concatenate y data
y = pd.concat([y_train, y_valid, y_test], axis=0).reset_index(drop=True).copy()

# Create a new column for blood product names
blood_products = ['ffp', 'prbc', 'platelet']
final_metrics_list = []

y.rename(columns = {'periop_platelets_units_transfused':'periop_platelet_units_transfused'},inplace = True)


for product in blood_products:
    X['predicted'] = (X[f'pred_prob_{product}'] >= thre_dict[f'{product}'])
    X[f'{product}_transfused'] = y[f'periop_{product}_units_transfused'].fillna(0) > 0
    product_metrics = stat_metrics3(X, 'predicted', f'{product}_transfused', 'Month', 'Data Split').rename(columns={'Group': 'Month'}).melt(id_vars=['Month', 'Split'], var_name='Metric', value_name='Value')
    product_metrics['Type'] = 'Model'
    product_metrics['Spec'] = threshold_name
    product_metrics['Blood Product'] = product
    final_metrics_list.append(product_metrics)

# Concatenate all metrics into one DataFrame
final_metrics = pd.concat(final_metrics_list, axis=0)

# Save the final metrics to a CSV file
#final_metrics.to_csv('monthly_metrics_multi_wei_full_multi_newThres.csv', index=False)

# %% Obtain Recommendations

# Reload model and data
#with open(f"{rae_folder}/Models/elective_only_20230828_thresholds_corrected_lhs_olddat_exclude_pt74.pkl", 'rb') as f:
    #optimal_threshold = pickle.load(f)
with open(f"{rae_folder}/Models/20250116_wei_multi_final.pkl", 'rb') as f:
    multi_target_model, feature_transformer, thre_dict = pickle.load(f)
with open(f"{rae_folder}/Data/20250116/train_test_elective_wei_123_multi_final.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)

#%% Define all functions 
# step 1: a function that takes in thresholds set for each product & which datasplit, 
# return sensitivity & PPV for each product and store in the master table 
def calculate_metrics(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    
    valid_indices = ~y_true.isna()
    y_true = y_true[valid_indices]
    y_probs = y_probs[valid_indices]
    y_pred = (y_probs >= threshold).astype(int)

    y_true = y_true.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    ppv = tp / (tp + fp)

    return sensitivity, ppv

def update_master_table(masterT, thresholds, y, multi_target_model, X):
    for product in thresholds.keys():
        threshold = thresholds[product]
        y_true = y[product]

        product_index = {'ffp': 0, 'prbc': 1, 'platelet': 2}[product]

        y_probs = multi_target_model.predict_proba(X)[product_index][:, 1]
        sensitivity, ppv = calculate_metrics(y_true, y_probs, threshold)

        masterT.loc[masterT[''] == 'Sensitivity', product] = sensitivity
        masterT.loc[masterT[''] == 'PPV', product] = ppv

    return masterT

def calculate_ci(value, n, alpha=0.05):
    """Calculate confidence interval using Wilson score interval"""
    z = norm.ppf(1 - alpha/2)
    denominator = 1 + z*z/n
    
    center = (value + z*z/(2*n))/denominator
    spread = z * np.sqrt(value*(1-value)/n + z*z/(4*n*n))/denominator
    
    lower = max(0, value - spread)
    upper = min(1, value + spread)
    
    return lower, upper

def calculate_metrics_with_ci(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    
    valid_indices = ~y_true.isna()
    y_true = y_true[valid_indices]
    y_probs = y_probs[valid_indices]
    y_pred = (y_probs >= threshold).astype(int)

    y_true = y_true.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    ppv = tp / (tp + fp)
    
    # Calculate CIs
    n_sens = tp + fn  # number of positive cases
    n_ppv = tp + fp   # number of predicted positives
    
    sens_ci = calculate_ci(sensitivity, n_sens)
    ppv_ci = calculate_ci(ppv, n_ppv)

    return sensitivity, ppv, sens_ci, ppv_ci

def update_master_table_with_ci(masterT, thresholds, y, multi_target_model, X):
    for product in thresholds.keys():
        threshold = thresholds[product]
        y_true = y[product]

        product_index = {'ffp': 0, 'prbc': 1, 'platelet': 2}[product]

        y_probs = multi_target_model.predict_proba(X)[product_index][:, 1]
        sensitivity, ppv, sens_ci, ppv_ci = calculate_metrics_with_ci(y_true, y_probs, threshold)

        # Format the values with CIs
        sens_str = f"{sensitivity:.3f} ({sens_ci[0]:.3f}-{sens_ci[1]:.3f})"
        ppv_str = f"{ppv:.3f} ({ppv_ci[0]:.3f}-{ppv_ci[1]:.3f})"

        masterT.loc[masterT[''] == 'Sensitivity', product] = sens_str
        masterT.loc[masterT[''] == 'PPV', product] = ppv_str

    return masterT
# Function to adjust sensitivity
def adjust_sensitivity(df, ABO_increment, TS_increment):
    for col in prbc_ffp_adjust:
        prod = re.search(r'(?<=_).+', col).group()
        # Extract the numeric value from the string (taking only the value before the parentheses)
        base_sensitivity = float(df.loc[df[''] == 'Sensitivity', prod].values[0].split(' ')[0])
        
        if base_sensitivity >= 0.8:
            ABO_increment = 0.05
            TS_increment = 0.1
            
        if 'ABO' in col:
            new_sens = base_sensitivity + ABO_increment
            df.loc[df[''] == 'Sensitivity', col] = str(round(new_sens, 3))
        elif 'T/S' in col:
            new_sens = base_sensitivity + TS_increment
            df.loc[df[''] == 'Sensitivity', col] = str(round(new_sens, 3))
    return df
def adjust_sensitivity_platelet(df, ABO_increment, TS_increment):
    for col in platelet_to_adjust:
        if 'ABO' in col:
            df.loc[df[''] == 'Sensitivity', col] = '0.76'  # 0.4 for new thres (0.3 + 0.1)
        elif 'T/S' in col:
            df.loc[df[''] == 'Sensitivity', col] = '0.86'  # 0.5 for new thres (0.3 + 0.2)
    return df
def get_thresholds(Sensitivity, prod):
    # Convert Sensitivity to float if it's a string
    if isinstance(Sensitivity, str):
        Sensitivity = float(Sensitivity.split(' ')[0])  # Extract the numeric value before the CI
        
    prod_ind = {
        "ffp": 0,
        "prbc": 1,
        "platelet": 2
    }
    precision, recall, thresholds = precision_recall_curve(
        y[:, prod_ind[prod]], y_pred[prod_ind[prod]][:, 1])
    j = np.argmin(np.abs(recall - Sensitivity))
    return thresholds[j]

def adjust_thresholds(df):
    for col in columns_to_adjust:
        prod = re.search(r'(?<=_).+', col).group()
        # Get sensitivity value and convert to float
        sensitivity_val = df.loc[df[''] == 'Sensitivity', col].values[0]
        if isinstance(sensitivity_val, str):
            sensitivity_val = float(sensitivity_val.split(' ')[0])
        
        df.loc[df[''] == 'Threshold Value', col] = get_thresholds(sensitivity_val, prod)
    return df
def update_master_table_ppv(masterT, y, multi_target_model, X):
    for product in columns_to_adjust:
        threshold = masterT.loc[masterT[''] == 'Threshold Value', product].values[0]
        prod = re.search(r'(?<=_).+', product).group()
        y_true = y[prod]
        # Assuming these columns have the same model and product_index logic
        product_index = {'ABO_prbc': 1, 'T/S_prbc': 1, 'ABO_ffp': 0, 'T/S_ffp': 0, 'ABO_platelet': 2, 'T/S_platelet': 2}[product]
        y_probs = multi_target_model.predict_proba(X)[product_index][:, 1]
        _, ppv = calculate_metrics(y_true, y_probs, threshold)
        masterT.loc[masterT[''] == 'PPV', product] = ppv

    return masterT

def update_master_table_ppv_with_ci(masterT, y, multi_target_model, X):
    for product in columns_to_adjust:
        threshold = masterT.loc[masterT[''] == 'Threshold Value', product].values[0]
        prod = re.search(r'(?<=_).+', product).group()
        y_true = y[prod]
        
        product_index = {'ABO_prbc': 1, 'T/S_prbc': 1, 'ABO_ffp': 0, 'T/S_ffp': 0, 
                        'ABO_platelet': 2, 'T/S_platelet': 2}[product]
        
        y_probs = multi_target_model.predict_proba(X)[product_index][:, 1]
        sensitivity, ppv, sens_ci, ppv_ci = calculate_metrics_with_ci(y_true, y_probs, threshold)

        # Format the values with CIs
        sens_str = f"{sensitivity:.3f} ({sens_ci[0]:.3f}-{sens_ci[1]:.3f})"
        ppv_str = f"{ppv:.3f} ({ppv_ci[0]:.3f}-{ppv_ci[1]:.3f})"

        masterT.loc[masterT[''] == 'Sensitivity', product] = sens_str
        masterT.loc[masterT[''] == 'PPV', product] = ppv_str

    return masterT
# Function to extract just the main value from strings with confidence intervals
def extract_main_value(x):
    try:
        if '(' in x:
            return float(x.split(' ')[0])
        return float(x)
    except:
        return x
#%%
# Create a controller 0 for valid, 1 for test, 2 for combined valid and test
Controller = 1

# Set dataset based on controller
if Controller == 0:
    X = X_valid
    y = y_valid
elif Controller == 1:
    X = X_test
    y = y_test
elif Controller == 2:
    X = pd.concat([X_valid, X_test])
    y = pd.concat([y_valid, y_test])

# a function that creates a df which stores threshold, sensitivity, and PPV for all 3 
# blood products and their respective ABO and T/S

# Create a master table to store thresholds, sensitivity, ppv for all products
# Define the data with None for all columns except 'Product'
data = {
    '': ['Threshold Value', 'Sensitivity', 'PPV'],
    'prbc': [None, None, None],
    'ABO_prbc': [None, None, None],
    'T/S_prbc': [None, None, None],
    'ffp': [None, None, None],
    'ABO_ffp': [None, None, None],
    'T/S_ffp': [None, None, None],
    'platelet': [None, None, None],
    'ABO_platelet': [None, None, None],
    'T/S_platelet': [None, None, None]
}

# Create the DataFrame
masterT = pd.DataFrame(data)

# Define the columns to adjust sensitivity for
columns_to_adjust = ['ABO_prbc', 'T/S_prbc', 'ABO_ffp', 'T/S_ffp', 'ABO_platelet', 'T/S_platelet']
prbc_ffp_adjust = ['ABO_prbc', 'T/S_prbc', 'ABO_ffp', 'T/S_ffp']
platelet_to_adjust = ['ABO_platelet', 'T/S_platelet']


if Controller == 1:  # For test set
    # Load thresholds from MasterT_valid.csv
    masterT_valid = pd.read_csv('MasterT_valid.csv')
    
    # Get the name of the first column
    index_column = masterT_valid.columns[0]
    
    # Create a new threshold dictionary from masterT_valid for all products and their ABO/T/S variants
    test_thre_dict = {}
    products = ['ffp', 'prbc', 'platelet']
    all_columns = ['ffp', 'prbc', 'platelet', 
                   'ABO_ffp', 'T/S_ffp', 
                   'ABO_prbc', 'T/S_prbc', 
                   'ABO_platelet', 'T/S_platelet']
    
    # Copy all thresholds from masterT_valid to masterT
    for column in all_columns:
        # Convert threshold to float
        threshold_value = float(masterT_valid.loc[masterT_valid[index_column] == 'Threshold Value', column].values[0])
        masterT.loc[masterT[masterT.columns[0]] == 'Threshold Value', column] = threshold_value
        if column in products:  # Only base products go into test_thre_dict
            test_thre_dict[column] = threshold_value

    # Calculate metrics using valid set thresholds
    y['ffp'] = y.periop_ffp_units_transfused > 0
    y['prbc'] = y.periop_prbc_units_transfused > 0
    y['platelet'] = y.periop_platelets_units_transfused > 0

    X = feature_transformer.transform(X)[feature_transformer.features]
    
    # First update base products metrics
    masterT = update_master_table_with_ci(masterT, test_thre_dict, y, multi_target_model, X)
    
    # Then update ABO and T/S metrics using their respective thresholds
    y = pd.DataFrame(y[['ffp', 'prbc', 'platelet']])
    masterT = update_master_table_ppv_with_ci(masterT, y, multi_target_model, X)
    
    # Save results to MasterT_test.csv
    masterT.to_csv('MasterT_test.csv', index=False)
    
    print("Test set analysis complete")
#%%

# set the threshold
old_thres = [0.22516264, 0.23850866, 0.11482752] # based on 0.66 sensitivity target for paper
# new_thres = [0.1732, 0.1444, 0.6295964] # updated thresholds by Colin

# Update the dictionary
thre_dict['ffp'], thre_dict['prbc'], thre_dict['platelet'] = old_thres

# Set thresholds for blood products & fill into the table
for key, value in thre_dict.items():
    masterT.loc[masterT[''] == 'Threshold Value', key] = value


y['ffp'] = y.periop_ffp_units_transfused > 0
y['prbc'] = y.periop_prbc_units_transfused > 0
y['platelet'] = y.periop_platelets_units_transfused > 0

X = feature_transformer.transform(X)[feature_transformer.features]

#masterT = update_master_table(masterT, thre_dict, y, multi_target_model, X)
masterT = update_master_table_with_ci(masterT, thre_dict, y, multi_target_model, X)

masterT1 = masterT.copy()

# Define the sensitivity increments
ABO_increment = 0.1
TS_increment = 0.2

print(masterT)
#%%
# Adjust the sensitivity values
masterT = adjust_sensitivity(masterT, ABO_increment, TS_increment)

masterT = adjust_sensitivity_platelet(masterT, ABO_increment, TS_increment)

y = y[['ffp', 'prbc', 'platelet']]
# Convert y to a NumPy array if it is a DataFrame
if isinstance(y, pd.DataFrame):
    columns = y.columns  # Save the column names
    y = y.to_numpy()
else:
    y = y

# X already being feature transformed
y_pred = multi_target_model.predict_proba(X)

masterT = adjust_thresholds(masterT)

y = pd.DataFrame(y, columns=columns)
#masterT = update_master_table_ppv(masterT, y, multi_target_model, X)
masterT = update_master_table_ppv_with_ci(masterT, y, multi_target_model, X)

print(masterT)

y_prob = multi_target_model.predict_proba(X)

# Dropping the index and transposing
masterT1 = masterT.transpose()

# Extract the first row of masterT1 to use it as column names
new_columns = masterT1.iloc[0, :].tolist()

# Update the column names of MasterT
masterT1.columns = new_columns

# Drop the first row of masterT1
masterT1 = masterT1.drop(masterT1.index[0])

masterT1 = masterT1.astype(str)

# Apply the conversion only to numeric columns, keeping 'Threshold Value' rows as is
for col in masterT1.columns:
    # Convert strings with confidence intervals to just the main value
    masterT1[col] = masterT1[col].apply(extract_main_value)
    # Then convert to numeric
    masterT1[col] = pd.to_numeric(masterT1[col], errors='ignore')


masterT1 = masterT1.round(9)
MasterT = masterT1
#%%

# Function to categorize probabilities for a given product and its thresholds
def categorize(prob, product, MasterT):
    thresholds = {
        f'{product}': MasterT.loc[product, 'Threshold Value'],
        f'T/S_{product}': MasterT.loc[f'T/S_{product}', 'Threshold Value'],
        f'ABO_{product}': MasterT.loc[f'ABO_{product}', 'Threshold Value']
    }
    if prob[1] >= thresholds[f'{product}']:
        return f'{product}'
    elif prob[1] >= thresholds[f'ABO_{product}']:
        return 'ABO'
    elif prob[1] >= thresholds[f'T/S_{product}']:
        return 'T/S'
    else:
        return 'None'

# Generate recommendations for each product from y_prob using MasterT and create the recommendation dataframe
recommendations = {
    'ffp': [categorize(prob, 'ffp', MasterT) for prob in y_prob[0]],
    'prbc': [categorize(prob, 'prbc', MasterT) for prob in y_prob[1]],
    'platelet': [categorize(prob, 'platelet', MasterT) for prob in y_prob[2]]
}

recommendation_df = pd.DataFrame(recommendations)

# Function to create the final recommendation
def create_final_rec(row):
    products = []
    for product in ['ffp', 'prbc', 'platelet']:
        if row[product] == product:
            products.append(product)
    if products:
        return ', '.join(products)
    # Check for T/S and ABO recommendations in order of priority
    for product in ['ffp', 'prbc', 'platelet']:
        if row[product] == 'ABO':
            return 'ABO'
    for product in ['ffp', 'prbc', 'platelet']:
        if row[product] == 'T/S':
            return 'T/S'
    return 'None'

# Apply the function to create the final_rec column
recommendation_df['final_rec'] = recommendation_df.apply(create_final_rec, axis=1)

print(recommendation_df)

frequency_table = recommendation_df['final_rec'].value_counts()

# Convert the frequency table to a DataFrame for better readability
frequency_table_df = frequency_table.reset_index()
frequency_table_df.columns = ['Label', 'Frequency']

# Print the resulting frequency table
print(frequency_table_df)

masterT.to_csv('MasterT_valid.csv', index=False)

#%% get metrics for ABO and T/S

y_valid['ffp'] = y_valid.periop_ffp_units_transfused > 0
y_valid['prbc'] = y_valid.periop_prbc_units_transfused > 0
y_valid['platelet'] = y_valid.periop_platelets_units_transfused > 0
y_true = y_valid[['ffp',  'platelet', 'prbc']].any(axis=1)

y_pred =  recommendation_df['final_rec'] == 'T/S'   # ABO or T/S

# Confusion Matrix components
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Sensitivity (Recall) and Specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Positive Predictive Value (Precision) and Negative Predictive Value
ppv = tp / (tp + fp)
npv = tn / (tn + fn)

# Confidence Intervals
ci_sensitivity = norm.interval(0.95, loc=sensitivity, scale=np.sqrt(sensitivity * (1 - sensitivity) / len(y_true)))
ci_specificity = norm.interval(0.95, loc=specificity, scale=np.sqrt(specificity * (1 - specificity) / len(y_true)))
ci_ppv = norm.interval(0.95, loc=ppv, scale=np.sqrt(ppv * (1 - ppv) / len(y_true)))

# Percent Recommended
percent_recommended = (tp + fp) / len(y_true)

print(f"Sensitivity & CI: {sensitivity:.2f} [{ci_sensitivity[0]:.2f}, {ci_sensitivity[1]:.2f}]")
print(f"Specificity & CI: {specificity:.3f} [{ci_specificity[0]:.3f}, {ci_specificity[1]:.3f}]")
print(f"PPV & CI: {ppv:.2f} [{ci_ppv[0]:.2f}, {ci_ppv[1]:.2f}]")
print(f"NPV: {npv:.3f}")
print(f"FN: {fn}")
print(f"FP: {fp}")
print(f"Percent Recommended: {percent_recommended:.3f}")
