import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_recall_curve,precision_score,recall_score
from sklearn.multioutput import MultiOutputClassifier
from MSBOS_06_Analysis_Tools import stat_metrics2, stat_metrics3
import shap
rae_folder = 'N:' # irb folder
exclude = 1 #flag, if set to 1 exclude specialties
model_name = ''

# import the correct utilties based on version of the data
if model_name == 'slou': # comparing to sunny lou et al. with reduced features
    from hipac_ml_msbos.hipac_modeling_tools_reduced import (
        FeatureTransformer, train_valid_test_split)
elif model_name == 'mono': # only with the historic transfusion rate feature
    from hipac_ml_msbos.hipac_modeling_tools_mono import (
        FeatureTransformer, train_valid_test_split)
elif model_name == 'old': # for old hipac model/paper (no commorbidities or hist transfused rates)
    from hipac_ml_msbos.hipac_modeling_tools_old import (
        FeatureTransformer, train_valid_test_split)
else: # all else
    from hipac_ml_msbos.hipac_modeling_tools import (
        FeatureTransformer, train_valid_test_split)

np.random.seed(123)
# Create a controller to indicate whether retrospective data
# or prospective data (0 for retro, 1 for prospective, 2 for both)
controller = 0
# %% Data pre-processing & data split
# data from 20250116
#data_df = pd.read_csv(f'{rae_folder}/Data/20250116/data_commit_b08825f__.csv') #added comorbidities  
# data from 20250325
data_df = pd.read_csv(f'{rae_folder}/Data/20250325/data_commit_4da9286.csv')
# open the train_id
train_id = pd.read_csv(f'{rae_folder}/Data/20250325/train_id.csv')
train_id = train_id['an_episode_id']
# rename deid_case_id to an_episode_id 
data_df.rename(columns={'deid_case_id':'an_episode_id'},inplace=True)
#data_df = pd.read_csv(f'{rae_folder}/Data/20230828/data_elective_only.csv') #added comorbidities
# change date format
data_df['scheduled_for_dttm'] = (
    pd.to_datetime(data_df['scheduled_for_dttm'], utc=True)
    .dt.tz_convert('US/Pacific')
)
# retro & prospective data spliting
silent =  data_df.loc[
    data_df['scheduled_for_dttm']>'2025-01-01']
data_df = data_df.loc[
    (data_df['scheduled_for_dttm']<'2025-01-01')
    & (data_df['scheduled_for_dttm']>'2015-12-31')
]

    # exclude cases w/ more than 100 cases
#data_df = data_df[data_df.periop_prbc_units_transfused<100]

outcomes_only = [
    'periop_platelets_units_transfused',
    'periop_prbc_units_transfused',
    'periop_ffp_units_transfused',
    'periop_cryoprecipitate_units_transfused'
]
#%%
X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(
    data_df,  target=outcomes_only, episode_ids = train_id, train_size=0.9, 
    valid_size=0.05, test_size=0.05,
    method='sorted', sort_by_col='scheduled_for_dttm'
)

# print characteristics of retrospective data
# Cases for each Train/Valid/Test Set
print("Train # Records: " + str(len(X_train)))
print("Valid # Records: " + str(len(X_valid)))
print("Test # Records: " + str(len(X_test)))
print("Total # Records: " + str(len(X_train)+len(X_valid)+len(X_test)))

# Print Min/Max Date Range for T/V/Te
print("Train Start Date: " + str(X_train['scheduled_for_dttm'].min()) + " End Date: " + str(X_train['scheduled_for_dttm'].max()))
print("Valid Start Date: " + str(X_valid['scheduled_for_dttm'].min()) + " End Date: " + str(X_valid['scheduled_for_dttm'].max()))
print("Test Start Date: " + str(X_test['scheduled_for_dttm'].min()) + " End Date: " + str(X_test['scheduled_for_dttm'].max()))
 
#%%saving the pickle files prior to calling transformer
with open(f"{rae_folder}/Data/20250325/train_test_elective_wei_123_multi_final.pkl", 'wb') as f:
    pickle.dump([X_train, y_train, X_valid, y_valid, X_test, y_test], f)

# %% Get MSBOS performance matched threshold
 
def MSBOSrecs(df):

    df['rec_any'] = (
        (df['msbos_ts'] > 0) |
        (df['msbos_tc'] > 0) |
        (df['msbos_rbc_cnt'] > 0)|
        (df['msbos_ffp'] > 0) |
        (df['msbos_platelets'] > 0) |
        (df['msbos_cryo'] > 0)
    )
    
    sz = len(df)

    def format_pcnt(value):
        return round((sum(value.fillna(0) > 0) / sz) * 100, 2)

    print('T/S Match: ' + str(format_pcnt(df['msbos_ts'])))
    print('ABO Check Match: ' + str(format_pcnt(df['msbos_tc'])))
    print('RBC Match: ' + str(format_pcnt(df['msbos_rbc_cnt'])))
    print('FFP Match: ' + str(format_pcnt(df['msbos_ffp'])))
    print('PLT Match: ' + str(format_pcnt(df['msbos_platelets'])))
    print('CRYO Match: ' + str(format_pcnt(df['msbos_cryo'])))
    print('Any Rec: '+  str(format_pcnt(df['rec_any'])))

    #return df_case
#%% Results from MSBOS predictions
# can change msbos_prbc to msbos_other blood product
MSBOSrecs(X_valid)
predictions = X_valid['msbos_prbc'].fillna(0) > 0
y_msbos = y_valid['periop_prbc_units_transfused'].fillna(0) > 0
 
# Compute confusion matrix
cm = confusion_matrix(y_msbos, predictions)
tn, fp, fn, tp = cm.ravel()

# Calculate sensitivity, specificity, PPV, NPV, and LR
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
lr_pos = sensitivity / (1 - specificity)
lr_neg = (1 - sensitivity) / specificity
 
# Print results
print(f"Confusion Matrix:\n{cm}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Positive Predictive Value (PPV): {ppv:.2f}")
print(f"Negative Predictive Value (NPV): {npv:.2f}")
print(f"Positive Likelihood Ratio (LR+): {lr_pos:.2f}")
print(f"Negative Likelihood Ratio (LR-): {lr_neg:.2f}")



# %% feature transformation ~ 1 min running
 
feature_transformer = FeatureTransformer()
X_train = feature_transformer.fit_transform(
        pd.concat([X_train, y_train], axis=1))[feature_transformer.features]
X_valid = feature_transformer.transform(X_valid)[feature_transformer.features]
X_test = feature_transformer.transform(X_test)[feature_transformer.features]

y_train = y_train.fillna(0)
y_valid = y_valid.fillna(0)
y_test = y_test.fillna(0)

#%% set indicators

# Create transfusion indicators for the training set
y_train['ffp'] = y_train.periop_ffp_units_transfused > 0
y_train['prbc'] = y_train.periop_prbc_units_transfused > 0
y_train['platelet'] = y_train.periop_platelets_units_transfused > 0

y_train = y_train[['ffp', 'platelet', 'prbc']]

# Create transfusion indicators for the validation set
y_valid['ffp'] = y_valid.periop_ffp_units_transfused > 0
y_valid['prbc'] = y_valid.periop_prbc_units_transfused > 0
y_valid['platelet'] = y_valid.periop_platelets_units_transfused > 0

y_valid = y_valid[['ffp', 'platelet', 'prbc']]

# %% model training

gbm = XGBClassifier()
gbm.fit(X_train, y_train['prbc'])

# multilebel prediction w/ MultiOutputClassifier from sklearn

# outcome
y_valid_multi = y_valid[['ffp', 'prbc', 'platelet']].copy()
y_train_multi = y_train[['ffp', 'prbc', 'platelet']].copy()
# Define the base model
base_model = gbm

# Use MultiOutputClassifier to handle multilabel classification
multi_target_model = MultiOutputClassifier(base_model, n_jobs=-1)

# Fit the model
multi_target_model.fit(X_train, y_train_multi)

# Predict on the validation set
y_valid_pred = multi_target_model.predict_proba(X_valid)

#%%
sensi_target = 0.66
#%% check thresholds for each product

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

#%% save

with open(f"{rae_folder}/Models/20250116_wei_multi_final.pkl", 'wb') as f:
    pickle.dump([multi_target_model, feature_transformer, thre_dict], f)


# %% monthly prediction
# Reload model and data
#with open(f"{rae_folder}/Models/elective_only_20230828_thresholds_corrected_lhs_olddat_exclude_pt74.pkl", 'rb') as f:
    #optimal_threshold = pickle.load(f)
with open(f"{rae_folder}/Models/20250116_wei_multi_final.pkl", 'rb') as f:
    multi_target_model, feature_transformer, thre_dict = pickle.load(f)
with open(f"{rae_folder}/Data/20250116/train_test_elective_wei_123_multi_final.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)
threshold_name = 0.66
if controller == 1 or controller == 2: #prospective data flag, case 'both' also need it
    X_silent = silent[X_train.columns].reset_index(drop=True)
    y_silent = silent[outcomes_only]
    X_silent['Data Split'] = 'silent'
# Transform and predict
X_train['Data Split'] = 'train'
X_valid['Data Split'] = 'valid'
X_test['Data Split'] = 'test'
if controller == 0:
    X = pd.concat([X_train,X_valid,X_test], axis=0).reset_index(drop=True).copy()
elif controller == 1:
    X = X_silent.copy()
elif controller == 2:
    X = pd.concat([X_train,X_valid,X_test,X_silent], axis=0).reset_index(drop=True).copy()
X_copy = X[['msbos_ts','msbos_tc','msbos_prbc','Data Split','scheduled_for_dttm']].copy() #variables that are needed later

y_full_msbos = X['msbos_prbc'].fillna(0) > 0
y_full_ts = X['msbos_ts'].fillna(0) > 0
y_full_tc = X['msbos_tc'].fillna(0) > 0

X = feature_transformer.transform(X)[feature_transformer.features]
#%%
X['pred_proba'] = multi_target_model.predict_proba(X)[1][:, 1]

X['predicted'] = (X['pred_proba'] >= thre_dict['prbc']) #optimal_threshold 
X['Month'] = X_copy['scheduled_for_dttm'].dt.to_period('M').astype(str) 
X['Data Split'] = X_copy['Data Split']
if controller == 0:
    y = pd.concat([y_train,y_valid,y_test], axis=0).reset_index(drop=True).copy()
elif controller == 1:
    y = y_silent.reset_index(drop=True).copy()
elif controller == 2:
    y = pd.concat([y_train,y_valid,y_test,y_silent], axis=0).reset_index(drop=True).copy()

X['prbc_transfused'] = y['periop_prbc_units_transfused'].fillna(0)> 0   
output_stat = stat_metrics2(X,'predicted', 'prbc_transfused','Data Split',CI=True)#.to_clipboard() but this requires authendication
# Save the DataFrame to an Excel file
# output_stat.to_csv('prosp_model_perform.csv', index=False)
#%%
#X['split'] = X_copy['Data Split']
#%%
monthly_metrics_model = stat_metrics3(X, 'predicted', 'prbc_transfused', 'Month', 'Data Split').rename(columns = {'Group':'Month'}).melt(id_vars=['Month','Split'], var_name='Metric', value_name='Value')
monthly_metrics_model['Type'] = 'Model'
monthly_metrics_model['Spec'] = threshold_name
# vertial combine
final_metrics = pd.concat([monthly_metrics_model],axis=0)#monthly_metrics_msbos,
# If you want to store the result:
final_metrics.to_csv('monthly_metrics_multi_wei_full_pbc_newDates_newSplit.csv', index=False)


# %% SHAP (will take some time)
explainer = shap.Explainer(gbm)
explanation = explainer(X_train)
shap_values = explanation.values
# Calculate the mean absolute SHAP values for each feature
mean_abs_shap_values = pd.Series(np.mean(abs(shap_values)), index=X_train.columns)
shap.plots.beeswarm(explanation,max_display=20)
# Sort features based on their mean absolute SHAP values
sorted_features = mean_abs_shap_values.sort_values(ascending=False)

# Select the top 50% of features
selected_features = sorted_features.head(int(len(sorted_features) * 0.5))

# Print the selected features
print(selected_features)

