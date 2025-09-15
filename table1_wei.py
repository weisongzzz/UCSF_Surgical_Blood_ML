# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:07:51 2025

@author: wzhang10
"""

import pandas as pd
from tabulate import tabulate

rae_folder = 'N:' # irb folder

#%% read file
X = pd.read_csv(f"{rae_folder}/Data/20250325/model_analysis_data_040725.csv")


#%%%%
# change date format
X['scheduled_for_dttm'] = (
    pd.to_datetime(X['scheduled_for_dttm'], utc=True)
    .dt.tz_convert('US/Pacific')
)
#%% Setting

X['bmi'] = X['bmi'].mask((X['bmi'] < 13) | (X['bmi'] > 60)) # make the invalid bmi NaN

X['preop_hct_any'] = X[['preop_hematocrit', 'preop_hematocrit_from_hb_abg', 'preop_hematocrit_from_hb_vbg']].mean(axis=1, skipna=True)

X['anemia'] = X['eci_blood_loss_anemia_diagnosis']|X['eci_deficiency_anemia_diagnosis']

X['liver_disease'] = X['eci_liver_disease_diagnosis']| X['cci_mild_liver_disease_diagnosis'] | X[ 'cci_moderate_severe_liver_disease_diagnosis'] | X['cci_liver_disease_diagnosis']

X['vascular_surgery'] = X['sched_surgical_service'] == 'Vascular Surgery'
X['transplant'] = X['sched_surgical_service'] == 'Transplant'
X['cardiac_surgery'] = X['sched_surgical_service'] == 'Cardiac Surgery'
#%%
# Group by 'Data Split' and calculate the number of cases and percentages
grouped_df = X.groupby('Data Split').size().reset_index(name='Number of Cases')
total_cases = len(X)
grouped_df['Number of Cases (%)'] = (
    grouped_df['Number of Cases'].astype(str) + 
    ' (' + 
    (grouped_df['Number of Cases'] / total_cases * 100).round(2).astype(str) + 
    ')'
)
grouped_df = grouped_df.drop(columns=['Number of Cases'])

# Calculate descriptive statistics for numeric columns
descriptive_stats = X.groupby('Data Split').agg({
    'age': ['mean', 'std'],
    'bmi': ['mean', 'std'],
    'prepare_asa': ['mean', 'std'],
    'preop_hct_any': ['mean', 'std'],
    'preop_platelets': ['mean', 'std'],
    'preop_ptt': ['mean', 'std'],
    'preop_creatinine': ['mean', 'std'],
    'preop_inr': ['mean', 'std'],
    'sched_est_case_length': ['mean', 'std'],
    'cci': ['mean', 'std'],
    'eci': ['mean', 'std'],
}).reset_index()

# Format mean and std into a single string
list_table1 = ['age', 'bmi', 'prepare_asa', 'preop_hct_any', 'preop_platelets', 'preop_ptt', 'preop_creatinine', 'preop_inr', 'sched_est_case_length', 'cci', 'eci']
for col in list_table1:
    descriptive_stats[(col, 'mean')] = descriptive_stats[(col, 'mean')].round(2).astype(str) + ' (' + descriptive_stats[(col, 'std')].round(2).astype(str) + ')'

# Select and rename columns
descriptive_stats = descriptive_stats.drop(columns=[(col, 'std') for col in list_table1])
descriptive_stats.columns = descriptive_stats.columns.droplevel(1)

# Merge the grouped_df with descriptive_stats
summary_df = pd.merge(grouped_df, descriptive_stats, on='Data Split')

# Calculate the date range for each 'Data Split'
date_range = X.groupby('Data Split')['scheduled_for_dttm'].agg([pd.Series.min, pd.Series.max]).reset_index()
date_range['Date Range'] = date_range['min'].dt.strftime('%Y-%m-%d') + ' - ' + date_range['max'].dt.strftime('%Y-%m-%d')
date_range = date_range[['Data Split', 'Date Range']]

# Merge the date range with the summary data
summary_df = pd.merge(summary_df, date_range, on='Data Split')

# Calculate binary statistics
list_table_bi = ['sex_male', 'race_ethnicity_white', 'race_ethnicity_latinx', 'race_ethnicity_asian', 'race_ethnicity_black', 'prior_dept_inpt_yn', 'icu_admit_prior_24hr_yn', 'arrival_ed_yn', 'anemia', 'liver_disease', 'sched_or_equip_cell_saver_yn', 'sched_neuromonitoring_yn', 'sched_bypass_yn', 'vascular_surgery', 'transplant', 'cardiac_surgery', 'outcome_prbc', 'outcome_ffp', 'outcome_platelet']
binary_stats = X.groupby('Data Split')[list_table_bi].sum().reset_index()
binary_stats['Total'] = X.groupby('Data Split').size().values

# Format binary stats
for col in list_table_bi:
    binary_stats[f'{col} (%)'] = binary_stats[col].astype(str) + ' (' + ((binary_stats[col] / binary_stats['Total']) * 100).round(2).astype(str) + ')'

# Drop original columns and keep formatted ones
binary_stats = binary_stats.drop(columns=list_table_bi + ['Total'])
binary_stats = binary_stats.set_index('Data Split').T[['train', 'valid', 'test']]

# Concatenate binary_stats vertically below the pivot_summary_df
final_summary_df = pd.concat([summary_df.set_index('Data Split').T[['train', 'valid', 'test']], binary_stats], axis=0)

# Create a dictionary for column name mapping
column_mapping = {
    'Date Range': 'Date Range',
    'Number of Cases (%)': 'Number of Cases (%)',
    'age': 'Age',
    'bmi': 'BMI (Kg/m^2)',
    'sex_male (%)': 'Sex Male (%)',
    'race_ethnicity_white (%)': 'White (%)',
    'race_ethnicity_latinx (%)': 'Latinx (%)',
    'race_ethnicity_asian (%)': 'Asian (%)',
    'race_ethnicity_black (%)': 'Black (%)',
    'prior_dept_inpt_yn (%)': 'Inpatient (%)',
    'icu_admit_prior_24hr_yn (%)': 'ICU (%)',
    'arrival_ed_yn (%)': 'Emergency Room (%)',
    'anemia (%)': 'Anemia (%)',
    'liver_disease (%)': 'Liver Disease (%)',
    'cci': 'CCI',
    'eci': 'ECI',
    'prepare_asa': 'Prepare ASA-PS',
    'preop_hct_any': 'Preop Hct',
    'preop_platelets': 'Preop Platelets',
    'preop_ptt': 'Preop PTT',
    'preop_creatinine': 'Preop Creatinine',
    'preop_inr': 'Preop INR',
    'sched_est_case_length': 'Est. Case Length',
    'sched_or_equip_cell_saver_yn (%)': 'Cell Saver (%)',
    'sched_neuromonitoring_yn (%)': 'Neuromonitoring (%)',
    'sched_bypass_yn (%)': 'Cardiac Bypass (%)',
    'vascular_surgery (%)': 'Vascular (%)',
    'transplant (%)': 'Transplant (%)',
    'cardiac_surgery (%)': 'Cardiac (%)',
    'outcome_prbc (%)': 'RBC Transfused (%)',
    'outcome_ffp (%)': 'FFP Transfused (%)',
    'outcome_platelet (%)': 'Platelet Transfused (%)'
}

# Define the correct order of columns
ordered_columns = [
    'Date Range', 'Number of Cases (%)', 'Age', 'BMI (Kg/m^2)', 'Sex Male (%)',
    'White (%)', 'Latinx (%)', 'Asian (%)', 'Black (%)', 'Inpatient (%)', 'ICU (%)', 
    'Emergency Room (%)', 'Anemia (%)', 'Liver Disease (%)', 'CCI', 'ECI', 
    'Prepare ASA-PS', 'Preop Hct', 'Preop Platelets', 'Preop PTT', 'Preop Creatinine', 
    'Preop INR', 'Est. Case Length', 'Cell Saver (%)', 'Neuromonitoring (%)', 
    'Cardiac Bypass (%)', 'Vascular (%)', 'Transplant (%)', 'Cardiac (%)', 
    'RBC Transfused (%)', 'FFP Transfused (%)', 'Platelet Transfused (%)'
]

# Rename the columns in final_summary_df
final_summary_df = final_summary_df.rename(index=column_mapping)

# Reorder the columns in the final summary DataFrame
final_summary_df = final_summary_df.reindex(ordered_columns)

# Rename the columns to capitalize them
final_summary_df = final_summary_df.rename(columns={'train': 'Train', 'valid': 'Valid', 'test': 'Test'})

# Save the final summary DataFrame to a CSV file
final_summary_df.to_csv('table_1.csv')

# Display the final summary DataFrame
print(final_summary_df)
#%%%%

#%% calculate case # and %
# Group by 'Data Split' and calculate the number of cases
case_X = X.groupby('Data Split').size().reset_index(name='Number of Cases')

# Calculate the percentages
total_cases = len(X)
case_X['Percentage'] = (case_X['Number of Cases'] / total_cases) * 100

# Display the grouped DataFrame
print(case_X)

#%% Table 1 formatting
data = {
    'Variable': ['number and percentage of cases'],
    'Train': ['228336 (90.0)'],
    'Validation': ['12696 (5.0)'],
    'Test': ['12696 (5.0)']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Format the table for publication
formatted_table = tabulate(df, headers='keys', tablefmt='grid')

# Print the formatted table
print(formatted_table)
