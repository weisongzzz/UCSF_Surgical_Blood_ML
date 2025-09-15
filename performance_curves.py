# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 20:11:08 2024

@author: mlin2
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss, confusion_matrix
from sklearn.utils import resample
from sklearn.calibration import calibration_curve 
from statsmodels.stats.proportion import proportion_confint

# %%
# Load the model and data
rae_folder = 'N:'
with open(f"{rae_folder}/Models/20250116_wei_multi_final.pkl", 'rb') as f:
    multi_target_model, feature_transformer, thre_dict = pickle.load(f)
with open(f"{rae_folder}/Data/20250325/train_test_elective_wei_123_multi_final.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)
#with open(f"{rae_folder}/Data/silent_study/silent_study_data_20231001_20240315.pkl", 'rb') as f:
    # p_df = pickle.load(f)

#p_df = pd.read_csv(f"{rae_folder}/Data/20241002/silent_study_4b77285.csv")  

# %%  

# Define helper functions
def bootstrap_curve(y_true, y_pred, curve_func, n_bootstraps=1000):
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    bootstrapped_curves = []
    for _ in range(n_bootstraps):
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_boot, y_pred_boot = y_true[indices], y_pred[indices]
        curve = curve_func(y_true_boot, y_pred_boot)
        bootstrapped_curves.append(curve)
    return bootstrapped_curves

def bootstrap_auc(y_true, y_pred, n_bootstraps=1000):
    # Convert to numpy arrays if they're pandas Series
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    auc_values = []
    for _ in range(n_bootstraps):
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_boot, y_pred_boot = y_true[indices], y_pred[indices]
        fpr, tpr, _ = roc_curve(y_true_boot, y_pred_boot)
        auc_values.append(auc(fpr, tpr))
    return np.percentile(auc_values, [2.5, 97.5])

def bootstrap_pr_auc(y_true, y_pred, n_bootstraps=1000):
    # Convert to numpy arrays if they're pandas Series
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    auc_values = []
    for _ in range(n_bootstraps):
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_boot, y_pred_boot = y_true[indices], y_pred[indices]
        precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_boot)
        auc_values.append(auc(recall, precision))
    return np.percentile(auc_values, [2.5, 97.5])

def plot_curve_with_ci(ax, x, y, bootstrapped_curves, label, color):
    ax.plot(x, y, label=label, color=color)
    interp_curves = []
    for curve in bootstrapped_curves:
        interp_x = np.linspace(0, 1, 100)
        interp_y = np.interp(interp_x, curve[0], curve[1])
        interp_curves.append(interp_y)
    y_lower = np.percentile(interp_curves, 2.5, axis=0)
    y_upper = np.percentile(interp_curves, 97.5, axis=0)
    ax.fill_between(interp_x, y_lower, y_upper, alpha=0.2, color=color)
    # Increase legend font size
    ax.legend(fontsize=14)
    
# Set up plot environment
plt.figure(dpi=600) # set high dpi when creating the plot
fig, axes = plt.subplots(3, 3, figsize=(24, 24), sharey='row')
fig.suptitle('Multi-Model Performance Plots', fontweight='bold', fontsize=30)
colors = {
    'valid': '#4A90E2',  # Royal blue
    'test': '#D35400'    # Burnt orange
}
outcomes = ['ffp', 'prbc', 'platelet']

# Function to prepare data and plot for each outcome
def prepare_and_plot(ax1, ax2, ax3, outcome, outcome_title, outcome_index, X_valid, y_valid, X_test, y_test, multi_target_model, feature_transformer):
    y_valid = y_valid.rename(columns={f'periop_{outcome}_units_transfused': 'target'})
    y_test = y_test.rename(columns={f'periop_{outcome}_units_transfused': 'target'})
    
    # Transform features
    X_valid_transformed = feature_transformer.transform(X_valid)[feature_transformer.features]
    X_test_transformed = feature_transformer.transform(X_test)[feature_transformer.features]
    
    # Predict probabilities
    y_valid['probs_test'] = multi_target_model.predict_proba(X_valid_transformed)[outcome_index][:, 1]
    y_test['probs_test'] = multi_target_model.predict_proba(X_test_transformed)[outcome_index][:, 1]
    
    # Create target binary column
    y_valid['target_binary'] = y_valid['target'] > 0
    y_test['target_binary'] = y_test['target'] > 0
    
    # Drop NaN values
    y_valid = y_valid.dropna(subset=['target', 'probs_test'])
    y_test = y_test.dropna(subset=['target', 'probs_test'])
    
    # ROC Curves
    for i, (data, color, label) in enumerate([(y_valid, colors['valid'], 'Valid'), (y_test, colors['test'], 'Test')]):
        fpr, tpr, thresholds = roc_curve(data['target_binary'], data['probs_test'])
        roc_auc = auc(fpr, tpr)
        auc_ci = bootstrap_auc(data['target_binary'], data['probs_test'])
        roc_curves = bootstrap_curve(data['target_binary'], data['probs_test'], roc_curve)
        plot_curve_with_ci(ax1, fpr, tpr, roc_curves, 
                          f'{label} ROC (AUC = {roc_auc:.2f} [{auc_ci[0]:.2f}-{auc_ci[1]:.2f}])', 
                          color)
    
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax1.set_xlabel('1-Specificity', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Sensitivity', fontsize=18, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=14)
    ax1.set_title(f'ROC Curves ({outcome_title})', fontsize=22, fontweight='bold')
    
    # Precision-Recall Curves
    for i, (data, color, label) in enumerate([(y_valid, colors['valid'], 'Valid'), (y_test, colors['test'], 'Test')]):
        precision, recall, thresholds = precision_recall_curve(data['target_binary'], data['probs_test'])
        pr_auc = auc(recall, precision)
        pr_auc_ci = bootstrap_pr_auc(data['target_binary'], data['probs_test'])
        pr_curves = bootstrap_curve(data['target_binary'], data['probs_test'], precision_recall_curve)
        plot_curve_with_ci(ax2, recall, precision, pr_curves, 
                          f'{label} Precision-Recall (PR AUC = {pr_auc:.2f} [{pr_auc_ci[0]:.2f}-{pr_auc_ci[1]:.2f}])', 
                          color)
    
    ax2.set_xlabel('Sensitivity', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Positive Predictive Value', fontsize=18, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=14)
    ax2.set_title(f'Precision-Recall Curves ({outcome_title})', fontsize=22, fontweight='bold')
    
    # Calibration Curves
    ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i, (data, color, label) in enumerate([(y_valid, colors['valid'], 'Valid'), (y_test, colors['test'], 'Test')]):
        prob_true, prob_pred = calibration_curve(data['target_binary'], data['probs_test'], n_bins=10)
        cal_curves = bootstrap_curve(data['target_binary'], data['probs_test'], 
                                   lambda y, p: calibration_curve(y, p, n_bins=10)[0])
        cal_curves_interp = [np.interp(prob_pred, np.linspace(0, 1, len(curve)), curve) 
                            for curve in cal_curves]
        prob_true_lower = np.percentile(cal_curves_interp, 2.5, axis=0)
        prob_true_upper = np.percentile(cal_curves_interp, 97.5, axis=0)
        brier_score = brier_score_loss(data['target_binary'], data['probs_test'])
        ax3.plot(prob_pred, prob_true, "s-", 
                label=f"{label} Calibration (Brier = {brier_score:.4f})", 
                color=color)
        ax3.fill_between(prob_pred, prob_true_lower, prob_true_upper, alpha=0.2, color=color)

    ax3.set_xlabel('Average Predicted Probability', fontsize=18, fontweight='bold')
    ax3.set_ylabel('Fraction of Positives', fontsize=18, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=14)
    ax3.set_title(f'Calibration Plots ({outcome_title})', fontsize=22, fontweight='bold')

    # Set tick label sizes for all plots
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=12)
# Rename first
y_valid.rename(columns={'periop_platelets_units_transfused': 'periop_platelet_units_transfused'}, inplace=True)
y_test.rename(columns={'periop_platelets_units_transfused': 'periop_platelet_units_transfused'}, inplace=True)

# Plot for each outcome, rearranged
order = ['prbc', 'ffp', 'platelet']
for i, outcome in enumerate(order):
    outcome_title = 'RBC' if outcome == 'prbc' else ('PLATELETS' if outcome == 'platelet' else outcome.upper())
    prepare_and_plot(axes[i, 0], axes[i, 1], axes[i, 2], outcome, outcome_title, outcomes.index(outcome), X_valid, y_valid, X_test, y_test, multi_target_model, feature_transformer)

# Adjust the layout
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('Performance_curves.png', dpi=600, bbox_inches='tight')
plt.show()
#%% 
# Still needs work! ---metrics output
prosp_df = p_df[['probs','target_binary']] 
prosp_df['prediction'] = p_df['probs'] > rbc_threshold 

# Assuming prosp_df is the dataframe with columns: target_binary, prediction, probs
# Example:
# prosp_df = pd.DataFrame({
#     'target_binary': [...],
#     'prediction': [...],
#     'probs': [...]
# })

# Calculate confusion matrix components
tn, fp, fn, tp = confusion_matrix(prosp_df['target_binary'], prosp_df['prediction']).ravel()

# Calculate Sensitivity (Recall) and its Confidence Interval
sensitivity = tp / (tp + fn)
sensitivity_ci_lower, sensitivity_ci_upper = proportion_confint(tp, tp + fn, method='wilson')

# Calculate Specificity
specificity = tn / (tn + fp)

# Calculate PPV (Precision) and its Confidence Interval
ppv = tp / (tp + fp)
ppv_ci_lower, ppv_ci_upper = proportion_confint(tp, tp + fp, method='wilson')

# Calculate NPV
npv = tn / (tn + fn)

# Calculate Percent Recommended (Proportion of Positive Predictions)
percent_recommended = (tp + fp) / (tp + fp + tn + fn)

# Calculate 1-Sensitivity (False Negative Rate)
one_minus_sensitivity = fn / (tp + fn)

# Prepare the result dictionary
result = {
    'Sensitivity & CI': f"{sensitivity:.3f} [{sensitivity_ci_lower:.3f}, {sensitivity_ci_upper:.3f}]",
    'Specificity': f"{specificity:.3f}",
    'PPV & CI': f"{ppv:.3f} [{ppv_ci_lower:.3f}, {ppv_ci_upper:.3f}]",
    'NPV': f"{npv:.3f}",
    'FN': fn,
    'FP': fp,
    'Percent Recommended': f"{percent_recommended:.3f}",
    '1-Sensitivity': f"{one_minus_sensitivity:.3f}"
}

# Print the results
for key, value in result.items():
    print(f"{key}: {value}")
