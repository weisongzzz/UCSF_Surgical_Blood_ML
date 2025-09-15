import numpy as np
import pandas as pd
import pickle
from sklearn.multioutput import MultiOutputClassifier

from scipy.stats import chi2_contingency,norm
from sklearn.metrics import (classification_report,multilabel_confusion_matrix,
                             roc_auc_score,confusion_matrix, roc_curve,
                             precision_recall_curve, auc, brier_score_loss)

rae_folder = 'N:' # irb folder
np.random.seed(123)
#%%
with open(f"{rae_folder}/Models/20250116_wei_multi_final.pkl", 'rb') as f:
    multi_target_model, feature_transformer, thre_dict = pickle.load(f)
with open(f"{rae_folder}/Data/20250325/train_test_elective_wei_123_multi_final.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)

#%% Model output
X_train['Data Split'] = 'train'
X_valid['Data Split'] = 'valid'
X_test['Data Split'] = 'test'

X = pd.concat([X_train, X_valid, X_test], axis=0)
y = pd.concat([y_train, y_valid, y_test], axis=0)


X_t = feature_transformer.transform(X)[feature_transformer.features]

pred_cols = ['FFP_model_prob', 'RBC_model_prob', 'PLT_model_prob']
for i, col in enumerate(pred_cols):
    X[col] = multi_target_model.predict_proba(X_t)[i][:, 1]
    
#%%
thresholds_dict = {
    'prbc': 0.1444,
    'ABO_prbc': 0.05062586,
    'T/S_prbc': 0.005055785,
    'ffp': 0.1732,
    'ABO_ffp': 0.079465225,
    'T/S_ffp': 0.014884387,
    'platelet': 0.6295964,
    'ABO_platelet': 0.40844685,
    'T/S_platelet': 0.21478316
}

MSBOS_threshold = 0.66
#%%
# Define the MasterT DataFrame containing the thresholds and other metrics
data = {
    'Threshold Value': [0.1444, 0.05062586, 0.005055785, 0.1732, 0.079465225, 0.014884387, 0.6295964, 0.40844685, 0.21478316]
    }

columns = ['prbc', 'ABO_prbc', 'T/S_prbc', 'ffp', 'ABO_ffp', 'T/S_ffp', 'platelet', 'ABO_platelet', 'T/S_platelet']
MasterT = pd.DataFrame(data, index=columns)

# Function to categorize probabilities for a given product and its thresholds
def categorize(prob, product, MasterT):
    thresholds = {
        f'{product}': MasterT.loc[product, 'Threshold Value'],
        f'T/S_{product}': MasterT.loc[f'T/S_{product}', 'Threshold Value'],
        f'ABO_{product}': MasterT.loc[f'ABO_{product}', 'Threshold Value']
    }
    if prob >= thresholds[f'{product}']:
        return f'{product}'
    elif prob >= thresholds[f'ABO_{product}']:
        return 'ABO'
    elif prob >= thresholds[f'T/S_{product}']:
        return 'T/S'
    else:
        return 'None'

# Generate recommendations for each product from y_prob using MasterT and create the recommendation dataframe
recommendations = {
    'ffp': [categorize(prob, 'ffp', MasterT) for prob in X['FFP_model_prob']],
    'prbc': [categorize(prob, 'prbc', MasterT) for prob in X['RBC_model_prob']],
    'platelet': [categorize(prob, 'platelet', MasterT) for prob in X['PLT_model_prob']]
}

recommendation_df = pd.DataFrame(recommendations, index = X.index)

#%%
# Function to create the final recommendation
def create_final_rec(row):
    products = []
    for product in ['prbc', 'ffp', 'platelet']:
        if row[product] == product:
            products.append(product)
    if products:
        return ', '.join(products)
    # Check for T/S and ABO recommendations in order of priority
    for product in ['prbc', 'ffp', 'platelet']:
        if row[product] == 'ABO':
            return 'ABO'
    for product in ['prbc', 'ffp', 'platelet']:
        if row[product] == 'T/S':
            return 'T/S'
        return 'None'

# Apply the function to create the final_rec column
X['model_rec'] = recommendation_df.apply(create_final_rec, axis=1)
X['model_rec'].value_counts(dropna=False)

#%%
X['outcome_ffp'] = y.periop_ffp_units_transfused > 0
X['outcome_prbc'] = y.periop_prbc_units_transfused > 0
X['outcome_platelet'] = y.periop_platelets_units_transfused > 0


X['outcome_ffp_units'] = y.periop_ffp_units_transfused
X['outcome_prbc_units'] = y.periop_prbc_units_transfused
X['outcome_platelet_units'] = y.periop_platelets_units_transfused


#%% Create Race variable & MSBOS

X.to_csv(f"{rae_folder}/Data/20250325/model_analysis_data_040725.csv")

#X = pd.read_csv(f"{rae_folder}/Data/20250325/model_analysis_data_040725.csv")    
#%%

#% Highlight Features

col= ['enc_id', 'an_episode_id',  'deid_mrn', 'retrospective_or_prd', 'Data Split', 'scheduled_for_dttm', 'sched_est_case_length', 'prepare_visit_yn',  'msbos_cnt', 'msbos_ts', 'msbos_tc', 'msbos_cryo', 'msbos_prbc', 'msbos_ffp', 'msbos_platelets', 'msbos_bppp', 'msbos_wholeblood', 'msbos_rbc_cnt', 'msbos_wb_cnt', 'prepare_asa', 'sched_surgical_service', 'prepare_asa_e',  'age', 'sex_female', 'sex_male', 'sex_nonbinary', 'sex_unknown', 'language_english_yn', 'language_interpreter_needed_yn', 'race_ethnicity_asian', 'race_ethnicity_black', 'race_ethnicity_latinx', 'race_ethnicity_multi', 'race_ethnicity_native_am_alaska', 'race_ethnicity_hi_pac_islander', 'race_ethnicity_other', 'race_ethnicity_swana', 'race_ethnicity_unknown', 'race_ethnicity_white', 'race_ethnicity', 'sched_surgical_dept_campus',  'cci', 'FFP_model_prob', 'RBC_model_prob', 'PLT_model_prob', 'model_rec', 'outcome_ffp', 'outcome_prbc', 'outcome_platelet']

X_lim = X[col]
X_lim.to_csv(f"{rae_folder}/Data/20250325/model_analysis_data_limfeatures_040725.csv")


#%% Full List of columns

['Unnamed: 0', 'enc_id', 'an_episode_id', 'an_record_id', 'retrospective_or_prd', 'deid_mrn', 'scheduled_for_dttm', 'sched_est_case_length', 'sched_surgeon_cnt', 'sched_proc_cnt', 'sched_proc_diag_cnt', 'sched_addon_yn', 'sched_emergency_case', 'sched_or_equip_cell_saver_yn', 'sched_neuromonitoring_yn', 'prepare_visit_yn', 'sched_bypass_yn', 'msbos_cnt', 'msbos_ts', 'msbos_tc', 'msbos_cryo', 'msbos_prbc', 'msbos_ffp', 'msbos_platelets', 'msbos_bppp', 'msbos_wholeblood', 'msbos_rbc_cnt', 'msbos_wb_cnt', 'prepare_asa', 'sched_case_class', 'sched_surgical_service', 'sched_prim_surgeon_provid_1', 'prepare_asa_e', 'sched_proc_max_complexity', 'age', 'sex_female', 'sex_male', 'sex_nonbinary', 'sex_unknown', 'language_english_yn', 'language_interpreter_needed_yn', 'race_ethnicity_asian', 'race_ethnicity_black', 'race_ethnicity_latinx', 'race_ethnicity_multi', 'race_ethnicity_native_am_alaska', 'race_ethnicity_hi_pac_islander', 'race_ethnicity_other', 'race_ethnicity_swana', 'race_ethnicity_unknown', 'race_ethnicity_white', 'race_ethnicity', 'sched_surgical_dept_campus', 'problem_list', 'eci_alcohol_abuse_diagnosis', 'eci_blood_loss_anemia_diagnosis', 'eci_cardiac_arrhythmias_diagnosis', 'eci_chf_diagnosis', 'eci_chronic_pulmonary_disease_diagnosis', 'eci_coagulopathy_diagnosis', 'eci_deficiency_anemia_diagnosis', 'eci_depression_diagnosis', 'eci_diabetes_complicated_diagnosis', 'eci_diabetes_uncomplicated_diagnosis', 'eci_drug_abuse_diagnosis', 'eci_fluid_disorder_diagnosis', 'eci_hiv_aids_diagnosis', 'eci_hypertension_complicated_diagnosis', 'eci_hypertension_uncomplicated_diagnosis', 'eci_hypothyroidism_diagnosis', 'eci_liver_disease_diagnosis', 'eci_lymphoma_diagnosis', 'eci_metastatic_cancer_diagnosis', 'eci_obesity_diagnosis', 'eci_other_neurological_disorders_diagnosis', 'eci_paralysis_diagnosis', 'eci_peptic_ulcer_disease_diagnosis', 'eci_peripheral_vascular_disease_diagnosis', 'eci_psychoses_diagnosis', 'eci_pulmonary_circulation_disease_diagnosis', 'eci_renal_failure_diagnosis', 'eci_rheumatic_disease_diagnosis', 'eci_solid_tumor_wo_metastatic_diagnosis', 'eci_valvular_disease_diagnosis', 'eci_weight_loss_diagnosis', 'cci_cerebrovascular_disease_diagnosis', 'cci_chf_diagnosis', 'cci_chronic_pulmonary_disease_diagnosis', 'cci_dementia_diagnosis', 'cci_diabetes_w_chronic_complication_diagnosis', 'cci_diabetes_wo_chronic_complication_diagnosis', 'cci_hemiplegia_paraplegia_diagnosis', 'cci_hiv_aids_diagnosis', 'cci_malignancy_diagnosis', 'cci_metastatic_solid_tumor_diagnosis', 'cci_mi_diagnosis', 'cci_mild_liver_disease_diagnosis', 'cci_moderate_severe_liver_disease_diagnosis', 'cci_peptic_ulcer_disease_diagnosis', 'cci_peripheral_vascular_disease_diagnosis', 'cci_renal_disease_diagnosis', 'cci_rheumatic_disease_diagnosis', 'preop_base_excess_abg', 'preop_base_excess_vbg', 'preop_bicarbonate_abg', 'preop_bicarbonate_vbg', 'preop_bun', 'preop_chloride', 'preop_chloride_abg', 'preop_chloride_vbg', 'preop_creatinine', 'preop_hematocrit', 'preop_hematocrit_from_hb_abg', 'preop_hematocrit_from_hb_vbg', 'preop_hemoglobin', 'preop_hemoglobin_abg', 'preop_hemoglobin_vbg', 'preop_lymphocyte_cnt', 'preop_neutrophil_cnt', 'preop_pco2_abg', 'preop_pco2_vbg', 'preop_platelets', 'preop_ph_abg', 'preop_ph_vbg', 'preop_potassium', 'preop_potassium_abg', 'preop_potassium_vbg', 'preop_rbc', 'preop_wbc', 'preop_sodium', 'preop_sodium_abg', 'preop_sodium_vbg', 'preop_albumin_serum', 'preop_bilirubin_total', 'preop_bilirubin_direct', 'preop_bilirubin_indirect', 'preop_inr', 'preop_ptt', 'weight_kg', 'height_cm', 'bmi', 'home_meds_anticoag_warfarin_yn', 'home_meds_anticoag_heparin_sq_yn', 'home_meds_anticoag_heparin_iv_yn', 'home_meds_anticoag_fondaparinux_yn', 'home_meds_anticoag_exoxaparin_yn', 'home_meds_anticoag_argatroban_yn', 'home_meds_anticoag_bivalirudin_yn', 'home_meds_anticoag_lepirudin_yn', 'home_meds_anticoag_dabigatran_yn', 'home_meds_anticoag_clopidogrel_yn', 'home_meds_anticoag_prasugrel_yn', 'home_meds_anticoag_ticlodipine_yn', 'home_meds_anticoag_abxicimab_yn', 'home_meds_anticoag_eptifibatide_yn', 'home_meds_anticoag_tirofiban_yn', 'home_meds_anticoag_alteplase_yn', 'home_meds_anticoag_apixaban_yn', 'enc_los_to_surg', 'prior_dept_inpt_yn', 'icu_admit_prior_24hr_yn', 'arrival_ed_yn', 'prior_dept_location', 'hist_prior_transf_platelets_days', 'hist_prior_transf_prbc_days', 'hist_prior_transf_ffp_days', 'hist_prior_transf_cryoprecipitate_days', 'hist_transf_1week_yn', 'hist_transf_1day_yn', 'hist_prior_transf_yn', 'picis_codes', '_store_into_s3', 'has_preop_dialysis', 'Data Split', 'cci', 'cci_diabetes_diagnosis', 'cci_liver_disease_diagnosis', 'eci', 'eci_diabetes_diagnosis', 'eci_hypertension_diagnosis', 'cci_dialysis', 'FFP_model_prob', 'RBC_model_prob', 'PLT_model_prob', 'model_rec', 'outcome_ffp', 'outcome_prbc', 'outcome_platelet']

#%%%% Code debugging

X_valid['scheduled_for_dttm'] = pd.to_datetime(X_valid['scheduled_for_dttm'])
min_date = X_valid['scheduled_for_dttm'].min()
max_date = X_valid['scheduled_for_dttm'].max()
print (min_date, max_date)

#%% Global sensitivity & PPV of ABO, T/S recommendations against any_transfusion
df = pd.read_csv(f"{rae_folder}/Data/20250325/model_all_clin_orders_optimizedthresholds_data_04_07_25.csv")

# For ABO: everything except T/S and NaN
abo_mask = (~df['model_rec'].isin(['T/S'])) & (~df['model_rec'].isna())

# For T/S: everything except NaN
ts_mask = ~df['model_rec'].isna()

# Create any_transfusion column
df['any_transfusion'] = (df['outcome_ffp'] > 0) | (df['outcome_prbc'] > 0) | (df['outcome_platelet'] > 0)

# Calculate metrics for ABO predictions (excluding T/S and NaN)
abo_true_pos = (abo_mask & df['any_transfusion']).sum()
abo_false_pos = (abo_mask & ~df['any_transfusion']).sum()
abo_false_neg = (~abo_mask & df['any_transfusion']).sum()

abo_sensitivity = abo_true_pos / (abo_true_pos + abo_false_neg)
abo_ppv = abo_true_pos / (abo_true_pos + abo_false_pos)

# Calculate metrics for T/S predictions (everything except NaN)
ts_true_pos = (ts_mask & df['any_transfusion']).sum()
ts_false_pos = (ts_mask & ~df['any_transfusion']).sum()
ts_false_neg = (~ts_mask & df['any_transfusion']).sum()

ts_sensitivity = ts_true_pos / (ts_true_pos + ts_false_neg)
ts_ppv = ts_true_pos / (ts_true_pos + ts_false_pos)

# Calculate confidence intervals
def calculate_ci(value, n, alpha=0.05):
    """Calculate confidence interval using Wilson score interval"""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    denominator = 1 + z*z/n
    
    center = (value + z*z/(2*n))/denominator
    spread = z * np.sqrt(value*(1-value)/n + z*z/(4*n*n))/denominator
    
    lower = max(0, value - spread)
    upper = min(1, value + spread)
    
    return lower, upper

# Calculate CIs
abo_sens_ci = calculate_ci(abo_sensitivity, abo_true_pos + abo_false_neg)
abo_ppv_ci = calculate_ci(abo_ppv, abo_true_pos + abo_false_pos)
ts_sens_ci = calculate_ci(ts_sensitivity, ts_true_pos + ts_false_neg)
ts_ppv_ci = calculate_ci(ts_ppv, ts_true_pos + ts_false_pos)

# Create results table
results_data = {
    'Metric': ['Sensitivity', 'PPV'],
    'ABO': [f"{abo_sensitivity:.3f} ({abo_sens_ci[0]:.3f}-{abo_sens_ci[1]:.3f})",
            f"{abo_ppv:.3f} ({abo_ppv_ci[0]:.3f}-{abo_ppv_ci[1]:.3f})"],
    'T/S': [f"{ts_sensitivity:.3f} ({ts_sens_ci[0]:.3f}-{ts_sens_ci[1]:.3f})",
            f"{ts_ppv:.3f} ({ts_ppv_ci[0]:.3f}-{ts_ppv_ci[1]:.3f})"]
}

results_df = pd.DataFrame(results_data)

#print(results_df)

# You can also save the results to csv if needed
# results_df.to_csv('combined_metrics.csv', index=False)

#%% Create monthly metrics of ABO & T/S from the model

# Convert scheduled_for_dttm to datetime if not already
df['scheduled_for_dttm'] = pd.to_datetime(df['scheduled_for_dttm'])

# Convert to Pacific time - check if timezone aware first
if df['scheduled_for_dttm'].dt.tz is None:
    # If naive, localize to UTC first
    df['scheduled_for_dttm'] = df['scheduled_for_dttm'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
else:
    # If already tz-aware, just convert
    df['scheduled_for_dttm'] = df['scheduled_for_dttm'].dt.tz_convert('US/Pacific')

# Rest of the code remains the same
df['Month'] = df['scheduled_for_dttm'].dt.strftime('%Y-%m')

# Initialize empty lists to store results
monthly_metrics = []

# Calculate monthly metrics for both ABO and T/S
for month in df['Month'].unique():
    month_df = df[df['Month'] == month]
    
    # ABO metrics
    month_abo_mask = (~month_df['model_rec'].isin(['T/S'])) & (~month_df['model_rec'].isna())
    
    abo_true_pos = (month_abo_mask & month_df['any_transfusion']).sum()
    abo_false_pos = (month_abo_mask & ~month_df['any_transfusion']).sum()
    abo_false_neg = (~month_abo_mask & month_df['any_transfusion']).sum()
    
    if (abo_true_pos + abo_false_neg) > 0:
        abo_sensitivity = abo_true_pos / (abo_true_pos + abo_false_neg)
    else:
        abo_sensitivity = None
        
    if (abo_true_pos + abo_false_pos) > 0:
        abo_ppv = abo_true_pos / (abo_true_pos + abo_false_pos)
    else:
        abo_ppv = None
    
    # T/S metrics
    month_ts_mask = ~month_df['model_rec'].isna()
    
    ts_true_pos = (month_ts_mask & month_df['any_transfusion']).sum()
    ts_false_pos = (month_ts_mask & ~month_df['any_transfusion']).sum()
    ts_false_neg = (~month_ts_mask & month_df['any_transfusion']).sum()
    
    if (ts_true_pos + ts_false_neg) > 0:
        ts_sensitivity = ts_true_pos / (ts_true_pos + ts_false_neg)
    else:
        ts_sensitivity = None
        
    if (ts_true_pos + ts_false_pos) > 0:
        ts_ppv = ts_true_pos / (ts_true_pos + ts_false_pos)
    else:
        ts_ppv = None
    
    # Get data split for this month
    data_split = month_df['Data Split'].iloc[0] if not month_df.empty else None
    
    # Add ABO results
    monthly_metrics.extend([
        {
            'Month': month,
            'Metric': 'Sensitivity',
            'Value': abo_sensitivity,
            'Type': 'Model',
            'Product': 'abo',
            'Data Split': data_split
        },
        {
            'Month': month,
            'Metric': 'PPV',
            'Value': abo_ppv,
            'Type': 'Model',
            'Product': 'abo',
            'Data Split': data_split
        }
    ])
    
    # Add T/S results
    monthly_metrics.extend([
        {
            'Month': month,
            'Metric': 'Sensitivity',
            'Value': ts_sensitivity,
            'Type': 'Model',
            'Product': 'ts',
            'Data Split': data_split
        },
        {
            'Month': month,
            'Metric': 'PPV',
            'Value': ts_ppv,
            'Type': 'Model',
            'Product': 'ts',
            'Data Split': data_split
        }
    ])

# Convert to DataFrame
monthly_metrics_df = pd.DataFrame(monthly_metrics)

# Sort by Month
monthly_metrics_df = monthly_metrics_df.sort_values('Month')


# Save to CSV
monthly_metrics_df.to_csv('monthly_metrics_abo_ts.csv', index=False)

print(monthly_metrics_df)

#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pickle
#%% Get clinician's abo & T/S monthly sensitivity & PPV

# Modified color dictionary to match ABO and T/S
color_dict = {
    'Clinician': 'blue',
    'Model': 'green'  # Changed to use same color for model predictions
}
# Define labels
ref_time_labels = [
    'Clinician Request',
    'Model Prediction'
]
metrics_labels = ['Sensitivity', 'PPV']

# Create legend handles for RefTime (lines with different colors)
ref_time_lines = [ 
    plt.Line2D([0], [0], color='blue', lw=2, label=ref_time_labels[0]),
    plt.Line2D([0], [0], color='green', lw=2, label=ref_time_labels[1])
    #plt.Line2D([0], [0], color='red', lw=2, label=ref_time_labels[2]),
]

# Create legend handles for Data Split (lines with different styles)
data_split_lines = [
    plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label=metrics_labels[0]),
    plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label=metrics_labels[1])
]

# Create and save legend for Reference Timepoints
fig1, ax1 = plt.subplots(figsize=(8, 4))
legend1 = ax1.legend(
    handles=ref_time_lines, 
    title='Type', 
    loc='center', 
    frameon=True
)
legend1.get_frame().set_edgecolor('black')
legend1.get_frame().set_linewidth(1)
plt.axis('off')
#plt.savefig('legend_type.png', bbox_inches='tight', dpi=300, transparent=True)
plt.show()
plt.close()

# Create and save legend for Metrics
fig2, ax2 = plt.subplots(figsize=(8, 4))
legend2 = ax2.legend(
    handles=data_split_lines, 
    title='Metric', 
    loc='center', 
    frameon=True
)
legend2.get_frame().set_edgecolor('black')
legend2.get_frame().set_linewidth(1)
plt.axis('off')
#plt.savefig('legend_metric.png', bbox_inches='tight', dpi=300, transparent=True)
plt.show()
plt.close()
#%% Read in data
# Read and process clinician data
clinician_dat = pd.read_csv('timetrend_abo_ts.csv') 
clinician_dat['Month'] = pd.to_datetime(clinician_dat['Month'])
# Map RefTime to Product for clinician data
clinician_dat['Product'] = clinician_dat['RefTime'].map({
    'ts_any': 'ts',
    'cs_any': 'abo'
})

# Read and process model data
model_retro = pd.read_csv('monthly_metrics_abo_ts.csv')
model_retro.rename(columns={'Split': 'Data Split'}, inplace=True)
model_retro['Month'] = pd.to_datetime(model_retro['Month'])
model_retro = model_retro[model_retro.Type == 'Model']

# Fix the Data Split for May 2024 onwards
model_retro.loc[model_retro['Month'] >= '2024-05', 'Data Split'] = 'test'

# Combine data
combine_dat = pd.concat([model_retro, clinician_dat], axis=0, ignore_index=True)
combine_dat = combine_dat[combine_dat.Month >= '2024-01']
combine_dat = combine_dat[~combine_dat.Month.isnull()]

# Print verification
print("\nUnique Products:", combine_dat['Product'].unique())
print("Clinician data sample:")
print(combine_dat[combine_dat['Type'] == 'Clinician'].head())
#%% 2 plots together
def plot_metric(combine_dat, color_dict, product_type, main_title, ax1, ax2):
    metrics = ['Sensitivity', 'PPV']
    
    # Filter data based on whether it's ABO or T/S
    model_data = combine_dat[(combine_dat['Product'] == product_type) & 
                            (combine_dat['Type'] == 'Model')]
    clinician_data = combine_dat[(combine_dat['Product'] == product_type) & 
                                (combine_dat['Type'] == 'Clinician')]
    
    # Sort data by date
    model_data = model_data.sort_values('Month')
    clinician_data = clinician_data.sort_values('Month')
    
    data_splits = ['valid', 'test']
    axes = [ax1, ax2]

    ref_time_lines = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Clinician Request'),
        plt.Line2D([0], [0], color='green', lw=2, label='Model Prediction')
    ]
    
    data_split_lines = [
        plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='Sensitivity'),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='PPV')
    ]

    for i, split in enumerate(data_splits):
        # Plot model data
        split_data = model_data[model_data['Data Split'] == split]
        for metric in metrics:
            metric_data = split_data[split_data['Metric'] == metric].sort_values('Month')
            if not metric_data.empty:
                axes[i].plot(metric_data['Month'], metric_data['Value'], 
                           color='green',
                           linestyle='-' if metric == 'Sensitivity' else '--',
                           marker='o',
                           markersize=8,
                           label=f'Model {metric}')

        # Plot clinician data for the appropriate time period
        if i == 0:  # validation period
            period_mask = (clinician_data['Month'] < '2024-05')
        else:  # test period
            period_mask = (clinician_data['Month'] >= '2024-05')
            
        for metric in metrics:
            clinician_metric = clinician_data[
                (clinician_data['Metric'] == metric) & period_mask
            ]
            if not clinician_metric.empty:
                axes[i].plot(clinician_metric['Month'], clinician_metric['Value'],
                           color='blue',
                           linestyle='-' if metric == 'Sensitivity' else '--',
                           marker='o',
                           markersize=8,
                           label=f'Clinician {metric}')

        axes[i].set_title(f"{main_title}\n{split.capitalize()}", 
                         fontsize=18, 
                         pad=20, 
                         weight='bold')
        
        axes[i].set_xlabel('Month', fontsize=15)
        axes[i].tick_params(axis='x', rotation=90, labelsize=14)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.7)
        axes[i].set_ylabel('Value', fontsize=15)
        
        axes[i].xaxis.set_major_locator(mdates.MonthLocator())
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Set specific x-range for each subplot
        if i == 0:  # validation period
            axes[i].set_xlim([pd.to_datetime('2024-01'), pd.to_datetime('2024-04')])
        else:  # test period
            axes[i].set_xlim([pd.to_datetime('2024-05'), pd.to_datetime('2024-10')])

        axes[i].set_ylim([0, 1])

    # Add legends
    legend1 = axes[0].legend(handles=ref_time_lines, title='Type', 
                            loc='upper left', bbox_to_anchor=(0.02, 0.98),
                            frameon=True)
    axes[0].add_artist(legend1)
    legend2 = axes[0].legend(handles=data_split_lines, title='Metric', 
                            loc='upper left', bbox_to_anchor=(0.02, 0.75),
                            frameon=True)
    
    for legend in [legend1, legend2]:
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)

# Create figure and plot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
plot_metric(combine_dat, color_dict, 'abo', 'ABO Compatibility', axes[0, 0], axes[0, 1])
plot_metric(combine_dat, color_dict, 'ts', 'Type & Screen', axes[1, 0], axes[1, 1])
plt.tight_layout()
plt.savefig('combined_metrics_abo_ts.png', bbox_inches='tight', dpi=300)
plt.show()

