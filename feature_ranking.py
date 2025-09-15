# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:17:35 2025

@author: wzhang10
"""

import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

rae_folder = 'N:' # irb folder
    
np.random.seed(123)

#%%
with open(f"{rae_folder}/Models/20250116_wei_multi_final.pkl", 'rb') as f:
    multi_target_model, feature_transformer, thre_dict = pickle.load(f)
with open(f"{rae_folder}/Data/20250325/train_test_elective_wei_123_multi_final.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)
#%%
X = pd.concat([X_valid, X_test], axis=0)
X_t = feature_transformer.transform(X)[feature_transformer.features]

# Define the feature name mappings for all classes
feature_name_mappings = {
    'hist_transfused_prbc': 'PRBC Transfusion Rate',
    'sched_est_case_length': 'Est. Case Length',
    'preop_hemoglobin': 'Preop Hgb',
    'any_transfusion_rate': 'Any Transfusion Rate',
    'msbos_rbc_cnt': 'MSBOS RBC Count',
    'hist_transfused_platelets': 'PLT Transfusion Rate',
    'msbos_ts': 'MSBOS T/S',
    'prepare_asa_2.0': 'ASA 2',
    'preop_hematocrit': 'Preop Hct',
    'age': 'Age',
    'sched_surgical_service_Otolaryngology Head Neck Surgery': 'Surgical Service: ENT/Head Neck',
    'preop_creatinine': 'Preop Creatinine',
    'preop_platelets': 'Preop Plt',
    'weight_kg': 'Weight (Kg)',
    'bmi': 'BMI',
    'preop_wbc': 'Preop WBC',
    'height_cm': 'Height (cm)',
    'preop_bun': 'Preop BUN',
    'sched_proc_cnt': 'Scheduled Procedure Count',
    'preop_rbc': 'Preop RBC',
    'hist_transfused_ffp': 'FFP Transfusion Rate',
    'preop_inr': 'Preop INR',
    'preop_ptt': 'Preop PTT',
    'preop_sodium': 'Preop Na',
    'sched_surgeon_cnt': 'Scheduled Surgeon Count',
    'preop_neutrophil_cnt': 'Preop Neutrophil Count',
    'prior_dept_location_PERIOP MZ': 'Prior Dept Location MZ',
    'preop_chloride': 'Preop Cl',
    'eci_coagulopathy_diagnosis': 'ECI Coagulopathy',
    'hist_transfused_cryoprecipitate': 'Cryoprecipitate Transfusion Rate',
    'prepare_asa_1.0': 'ASA 1'
}

# Rename the columns of the original DataFrame
X_t_renamed = X_t.rename(columns=feature_name_mappings)
#%% SHAP feature ranking

# Define the product names and the order for the subplots
product_names = ['FFP', 'RBC', 'Platelets']
plot_order = [1, 0, 2]  # Switch positions of class 0 and class 1

# Prepare the plot
n_classes = len(multi_target_model.estimators_)
plt.figure(dpi=600) # set high dpi when creating the plot
fig, axes = plt.subplots(n_classes, 1, figsize=(10, 5 * n_classes))

shap_values_dict = {}
mean_abs_shap_values_dict = {}

# For each class (i.e., each product)
for i, estimator in enumerate(multi_target_model.estimators_):
    # Initialize SHAP Explainer for the current estimator
    explainer = shap.Explainer(estimator)
    
    # Generate SHAP values for the transformed validation data
    shap_values = explainer(X_t_renamed)
    
    # Store SHAP values in a dictionary
    shap_values_dict[f'class_{i}'] = shap_values
    
    # Calculate the mean absolute SHAP values for each feature for the current class
    mean_abs_shap_values = pd.Series(np.mean(np.abs(shap_values.values), axis=0), index=X_t_renamed.columns)
    
    # Store mean absolute SHAP values in a dictionary
    mean_abs_shap_values_dict[f'class_{i}'] = mean_abs_shap_values
    
    # Visualize the SHAP values using a beeswarm plot for the current class
    plt.sca(axes[plot_order[i]])  # Use plot_order to switch positions
    shap.summary_plot(shap_values.values, X_t_renamed, plot_type="dot", max_display=20, show=False)
    axes[plot_order[i]].set_title(f'{product_names[i]}')
    
    # Customize the x-axis title and feature name font size
    axes[plot_order[i]].set_xlabel('SHAP value', fontsize=12)
    for tick in axes[plot_order[i]].get_xticklabels():
        tick.set_fontsize(10)
    for tick in axes[plot_order[i]].get_yticklabels():
        tick.set_fontsize(7)  # Adjust the font size for feature names

# Adjust layout to avoid overlap
plt.tight_layout()
# Save the plot with high DPI
plt.savefig('Feature_rankings.png', dpi=600, bbox_inches='tight')
plt.show()

#%%
# Sort features based on their mean absolute SHAP values for each class
sorted_features_dict = {class_name: mean_shap.sort_values(ascending=False) for class_name, mean_shap in mean_abs_shap_values_dict.items()}

# Select the top 50% of features for each class
selected_features_dict = {class_name: sorted_feat.head(int(len(sorted_feat) * 0.5)) for class_name, sorted_feat in sorted_features_dict.items()}

# Print the selected features for each class
for class_name, selected in selected_features_dict.items():
    print(f'Selected features for {class_name}:')
    print(selected[:20])