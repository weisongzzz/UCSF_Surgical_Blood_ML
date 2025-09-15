# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:59:34 2024

@author: pramaswamy

contains 4 functions; no execution
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
from sklearn.calibration import calibration_curve
from pathlib import Path
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint 
import pprint
import pandas as pd
from scipy.stats import beta



#%%

def stat_metrics(y_valid, test_name='', model_name='',set_sensitivity=None,  predictions=None):
    """
    Calculates performance metrics for classification models, 
        including sensitivity (&95%CI), specificity, PPV (&95%CI), NPV, FN, FP, 
        percentage of cases with transfusion recommendations predicted by the model, and sample size.
    Optionally adjusts the prediction threshold based on a target sensitivity.
    
    Parameters:
    - y_valid: DataFrame with true labels ('target') and predicted probabilities ('prediction_prob').
    - test_name: Optional name of the test.
    - model_name: Optional name of the model.
    - set_sensitivity: Optional target sensitivity to adjust the prediction threshold. 
        Default probability threshold to binarize the prediction is 0.5
    - predictions: Optional precomputed binary predictions.
    
    Returns:
    - Dictionary containing the metrics
    """
    
   
    fpr, tpr, _ = roc_curve(y_valid['target'], y_valid['prediction_prob'])
    roc_auc = auc(fpr, tpr)
    if set_sensitivity!=None:
        precision, recall, thresholds = precision_recall_curve(
            y_valid['target'], y_valid['prediction_prob'])

    
        i = np.argmin(np.abs(recall - set_sensitivity))

        print(recall[i], precision[i], thresholds[i])
        threshold = thresholds[i]

        predictions = (y_valid['prediction_prob'] >= threshold).astype(int)
        

    # Compute confusion matrix
    cm = confusion_matrix(y_valid['target'], predictions, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    # Calculate sensitivity, specificity, PPV, NPV, and 1-Sensitivity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    one_minus_sens = 1 - sensitivity
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    pcnt_recommended = predictions.mean()

    # sensitivity CI
    sen_l, sen_u = proportion_confint(
        tp, tp + fn, alpha=0.05, method='wilson'
    )
    ppv_l, ppv_u = proportion_confint(
        tp, tp + fp, alpha=0.05, method='wilson'
    )
    generate_results(y_valid['target'], y_valid['prediction_prob'], predictions, test_name, model_name, len(y_valid['target']))



    # Create a dictionary with the desired output values
    if set_sensitivity!=None:
        
        return {
            'Sample Size': len(y_valid),
            
            'threshold': threshold,
            'Sensitivity & CI': '{:.1f}%, [{:.0f}, {:.0f}]'.format(
                    sensitivity * 100, sen_l * 100, sen_u * 100
                ),
            'Sensitivity': sensitivity,
            'sensitivity_CI': '[%.2f, %.2f]' % (sen_l, sen_u),
            'Specificity': specificity,
            'FP': fp,
            'FN': fn,
            '1-Sensitivity': one_minus_sens,
            'PPV & CI': '{:.1f}%, [{:.0f}, {:.0f}]'.format(
                    ppv * 100, ppv_l * 100, ppv_u * 100
                ),
            'NPV': npv,
            'pcnt_recommended': pcnt_recommended }
    else: 
        return{
        'Sample Size': len(y_valid),
        
        'Sensitivity & CI': '{:.1f}%, [{:.0f}, {:.0f}]'.format(
                sensitivity * 100, sen_l * 100, sen_u * 100
            ),
        'Sensitivity': sensitivity,
        'sensitivity_CI': '[%.2f, %.2f]' % (sen_l, sen_u),
        'Specificity': specificity,
        'FP': fp,
        'FN': fn,
        '1-Sensitivity': one_minus_sens,
        'PPV & CI': '{:.1f}%, [{:.0f}, {:.0f}]'.format(
                ppv * 100, ppv_l * 100, ppv_u * 100
            ),
        'NPV': npv,
        'pcnt_recommended': pcnt_recommended
    } 
def compute_confidence_interval(successes, total, alpha=0.05):
    """Compute the 95% CI using the Wilson score interval."""
    if total == 0:
        return (0, 0)  # Handle cases with zero division gracefully
    
    lower = beta.ppf(alpha / 2, successes, total - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    
    return (round(lower,2), round(upper,2))

def stat_metrics2(df, pred_col, outcome_col, group, CI=False):
    """
    Extended stat_metrics function with optional 95% CI calculation.
    """
    results = []

    # Loop over each unique value in 'group' (e.g., 'train', 'test', etc.)
    for ds_value in df[group].unique():
        # Filter the dataframe for the current data split
        split_data = df[df[group] == ds_value]

        # Get the outcome and prediction columns
        predictions = split_data[pred_col].astype(int).fillna(0) > 0
        outcome = split_data[outcome_col].astype(int).fillna(0) > 0

        try:
            cm = confusion_matrix(outcome, predictions, labels=[0, 1])
            if cm.size == 4:  # Normal case: cm is a 2x2 matrix
                tn, fp, fn, tp = cm.ravel()
            else:  # Handle edge cases with fewer elements
                tn = fp = fn = tp = 0
                if cm.size == 1:  # Only one class present
                    if predictions.sum() == 0:  # Only negative class
                        tn = cm[0, 0]
                    else:  # Only positive class
                        tp = cm[0, 0]
        except ValueError:
            tn = fp = fn = tp = 0  # Handle unexpected errors gracefully

        # Calculate metrics
        total_cases = tp + fn + tn + fp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fnr = fn / total_cases * 100
        fpr = fp / total_cases * 100

        # Store the basic results
        result = {
            'Group': ds_value,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv,
            'FN%': fnr,
            'FP%': fpr
        }

        # Calculate confidence intervals if CI=True
        if CI:
            result.update({
                'Sensitivity_CI': compute_confidence_interval(tp, tp + fn),
                'Specificity_CI': compute_confidence_interval(tn, tn + fp),
                'PPV_CI': compute_confidence_interval(tp, tp + fp),
                'NPV_CI': compute_confidence_interval(tn, tn + fn),
                'FN%_CI': compute_confidence_interval(fn, total_cases),
                'FP%_CI': compute_confidence_interval(fp, total_cases)
            })

        results.append(result)

    # Convert results to DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return round(results_df, 2)

def stat_metrics3(df, pred_col, outcome_col, time, data_split, CI=False):
    """
    Extended stat_metrics function with optional 95% CI calculation.
    """
    results = []
    
    # Loop over each unique value in 'time'
    for ds_value in df[time].unique():
        for split in df[df[time] == ds_value][data_split].unique():
            # Filter the dataframe for the current data split
            split_data = df[(df[time] == ds_value) & (df[data_split] == split)]

            # Get the outcome and prediction columns
            predictions = split_data[pred_col].astype(int).fillna(0) > 0
            outcome = split_data[outcome_col].astype(int).fillna(0) > 0

            try:
                cm = confusion_matrix(outcome, predictions, labels=[0, 1])
                if cm.size == 4:  # Normal case: cm is a 2x2 matrix
                    tn, fp, fn, tp = cm.ravel()
                else:  # Handle edge cases with fewer elements
                    tn = fp = fn = tp = 0
                    if cm.size == 1: # Only one class present
                        if predictions.sum() == 0:  # Only negative class
                            tn = cm[0, 0]
                        else:  # Only positive class
                            tp = cm[0, 0]
            except ValueError:
                tn = fp = fn = tp = 0  # Handle unexpected errors gracefully

            # Calculate metrics
            total_cases = tp + fn + tn + fp
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            fnr = fn / total_cases * 100
            fpr = fp / total_cases * 100

            # Store the basic results
            result = {
                'Group': ds_value,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'PPV': ppv,
                'NPV': npv,
                'FN%': fnr,
                'FP%': fpr,
                'Split': split
                }

        # Calculate confidence intervals if CI=True
            if CI:
                result.update({
                    'Sensitivity_CI': compute_confidence_interval(tp, tp + fn),
                    'Specificity_CI': compute_confidence_interval(tn, tn + fp),
                    'PPV_CI': compute_confidence_interval(tp, tp + fp),
                    'NPV_CI': compute_confidence_interval(tn, tn + fn),
                    'FN%_CI': compute_confidence_interval(fn, total_cases),
                    'FP%_CI': compute_confidence_interval(fp, total_cases)
                    })

            results.append(result)

    # Convert results to DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return round(results_df, 2)

def generate_results(y_true, y_score, predictions, test_name, model_name, sample_size,folder_path=None):
    """
    Generates and plots ROC, Precision-Recall, and Calibration curves for a classification model, along with accuracy and AUC scores. 
    The plots are optionally saved to a specified folder.
    
    Parameters:
    - y_true: True labels of the dataset.
    - y_score: Predicted probabilities from the model.
    - predictions: Binary predictions from the model.
    - test_name: Name of the test (included in the plot title).
    - model_name: Name of the model (included in the plot title).
    - sample_size: The number of samples in the dataset.
    - folder_path: Optional path to save the figures (default: 'N:/Results/Figures').
    
    Returns:
    - Displays the plots for ROC, Precision-Recall, and Calibration curves. Optionally saved.
    """

    accuracy = accuracy_score(y_true, predictions)
    if folder_path is None:
        pathfigures = Path('N:/Results/Figures')
    else:
        pathfigures = folder_path
    
    # Set the overall aesthetics
    plt.style.use('seaborn-v0_8-white')  # Use a style with a white background

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=15, fontweight='bold')
    plt.title('Receiver Operating Characteristic', fontsize=18, fontweight='bold')
    plt.legend(loc=4, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, label='PR-AUC = %0.2f' % pr_auc, linewidth=2)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=15, fontweight='bold')
    plt.ylabel('Precision', fontsize=15, fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=18, fontweight='bold')
    plt.legend(loc='lower left', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=5)
    plt.subplot(1, 3, 3)
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', label='Calibration curve', markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=15, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=15, fontweight='bold')
    plt.title('Calibration Curve', fontsize=18, fontweight='bold')
    plt.legend(loc='lower right',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.suptitle(model_name + ' ' + test_name + '\n' + ' N = ' + str(sample_size) + " & Accuracy: %.2f%%" % (accuracy * 100.0), fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f_name = model_name + '_sz_' + str(sample_size) + "_" + test_name + '.png'
    try:
        plt.savefig(pathfigures / f_name)
        plt.show()
    except:
        print('figure not saved due to error in folder_path')
 


