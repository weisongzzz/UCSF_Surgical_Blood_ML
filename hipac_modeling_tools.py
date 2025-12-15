# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:42:13 2023

@author: pramaswamy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureTransformer:
    def _get_picis_code_with_transfusion(self, data_df,\
                                         transfusion_columns=None):
        # Default transfusion columns
        if transfusion_columns is None:
            transfusion_columns = [
                'periop_prbc_units_transfused',
                'periop_platelets_units_transfused',
                'periop_ffp_units_transfused',
                'periop_cryoprecipitate_units_transfused'
            ]
        # Split picis_codes and explode into separate rows
        # create dummy features
        df_transfusion = (
            data_df.assign(
                picis_codes=data_df.picis_codes.str.split(";")
            ).explode('picis_codes')
            .reset_index(drop=True)
            .fillna('NA0000')  # Replace NaN values with a unique identifier
        )
    
        # Create separate hist_transfused columns for each transfusion column
        # To indicate if either type of transfusion occurs
        for col in transfusion_columns:
            hist_col_name = f'hist_transfused_{col.split("_")[1]}'
            df_transfusion[hist_col_name] = df_transfusion[col] > 0
            #df_transfusion['hist_any_transfsion']= max(df_transfusion[hist_col_name])
        # Group by picis code. Take # rows w/ any transfusion indictor 
        # divided by picis code repetition count
        Any_transfusion_rate = df_transfusion.groupby('picis_codes')[transfusion_columns].apply(lambda x: (x > 0).any(axis=1).mean().round(decimals=4)) 
        # Calculate mean transfusion rate for each picis_code and transfusion column
        # only for training data; use those hist_trans for validation
        # how many transfusion cases
        transfusion_means = []
        # create a dataframe w/ only each picis code & the count for each
        transfusion_counts = df_transfusion.groupby('picis_codes').size().reset_index(name='count')
    
        for col in transfusion_columns:
            hist_col_name = f'hist_transfused_{col.split("_")[1]}'
            # Group the df_transfusion DataFrame by picis_codes, 
            # calculate the mean of the hist_col_name column for each group
            transfusion_mean = (
                df_transfusion.groupby('picis_codes', as_index=False)\
                    [hist_col_name].mean().round(decimals=4)
            )
            transfusion_mean.columns = ['picis_codes', hist_col_name]
            # append the result to a list
            transfusion_means.append(transfusion_mean)
    
        # Merge all mean transfusion rates into a single dataframe
        df_tx_picis_means = transfusion_counts
        for mean_df in transfusion_means:
            df_tx_picis_means = pd.merge(df_tx_picis_means, mean_df, how='left', on='picis_codes')
        # merge the any_transfusion_rate as well
        df_tx_picis_means = df_tx_picis_means.merge(Any_transfusion_rate.rename('any_transfusion_rate'), on='picis_codes')
        # Sort the DataFrame by the specified columns
        df_tx_picis_means = df_tx_picis_means.sort_values(\
            by=['any_transfusion_rate', 'count', 'hist_transfused_ffp', 'picis_codes'],\
            ascending=[False, False, False, True]\
            )
        # Reset the index to make it ascending and drop the old index
        df_tx_picis_means.reset_index(drop=True, inplace=True)
        # Set self.df_tx_picis_means with the calculated means
        self.df_tx_picis_means = df_tx_picis_means
        self.transfusion_rates_by_picis = df_tx_picis_means.reset_index().rename(columns={'index': 'index'})
        # Identify picis_codes with any transfusion history
        self.df_PICIS_txn = df_tx_picis_means[
            df_tx_picis_means[[f'hist_transfused_{col.split("_")[1]}'\
                               for col in transfusion_columns]].sum(axis=1) > 0
        ]['picis_codes'].tolist()
    
        # One-hot encode picis_codes
        # for some code
        self.picis_one_hot_encoder = OneHotEncoder(
            # Specifies the categories per feature
            # add ['PI0000'] + ['NA0000'] for N/A picis code
            categories=[self.df_PICIS_txn + ['PI0000'] + ['NA0000']],
            sparse=False,
            handle_unknown='infrequent_if_exist'
        )
        # .fit to have the encoder learn picis codes from the original df
        self.picis_one_hot_encoder.fit(df_transfusion[['picis_codes']])
        # store a full list of picis codes names
        self.picis_one_hot_feature_names = list(self.picis_one_hot_encoder.get_feature_names_out())
        
    # how sick of patients using index code
    def _calculate_comorbidities(self, data_df): 
        eci_points = {'eci_drug_abuse_diagnosis': -7,
         'eci_obesity_diagnosis': -4,
         'eci_psychoses_diagnosis': -3,
         'eci_depression_diagnosis': -3,
         'eci_blood_loss_anemia_diagnosis': -2,
         'eci_deficiency_anemia_diagnosis': -2,
         'eci_valvular_disease_diagnosis': -1,
         #'eci_hypertension_uncomplicated_diagnosis': 0,
         #'eci_hypertension_complicated_diagnosis': 0,
         #'eci_diabetes_uncomplicated_diagnosis': 0,
         #'eci_diabetes_complicated_diagnosis': 0,
         
         'eci_hypothyroidism_diagnosis': 0,
         'eci_hiv_aids_diagnosis': 0,
         'eci_peptic_ulcer_disease_diagnosis': 0,
         'eci_rheumatic_disease_diagnosis': 0,
         'eci_alcohol_abuse_diagnosis': 0,
         'eci_peripheral_vascular_disease_diagnosis': 2,
         'eci_chronic_pulmonary_disease_diagnosis': 3,
         'eci_coagulopathy_diagnosis': 3,
         'eci_solid_tumor_wo_metastatic_diagnosis': 4,
         'eci_pulmonary_circulation_disease_diagnosis': 4,
         'eci_cardiac_arrhythmias_diagnosis': 5,
         'eci_renal_failure_diagnosis': 5,
         'eci_fluid_disorder_diagnosis': 5,
         'eci_weight_loss_diagnosis': 6,
         'eci_other_neurological_disorders_diagnosis': 6,
         'eci_chf_diagnosis': 7,
         'eci_paralysis_diagnosis': 7,
         'eci_lymphoma_diagnosis': 9,
         'eci_liver_disease_diagnosis': 11,
         'eci_metastatic_cancer_diagnosis': 12}

        # child commb
        cci_points = {'cci_mi_diagnosis': 1,
              #'cci_mild_liver_disease_diagnosis' : 1,
             'cci_chf_diagnosis': 1,
             'cci_peripheral_vascular_disease_diagnosis': 1,
             'cci_cerebrovascular_disease_diagnosis': 1,
             'cci_dementia_diagnosis': 1,
             'cci_chronic_pulmonary_disease_diagnosis': 1,
             'cci_rheumatic_disease_diagnosis': 1,
             'cci_peptic_ulcer_disease_diagnosis': 1,
             #'cci_diabetes_wo_chronic_complication_diagnosis': 1,
             'cci_malignancy_diagnosis':2,
             #'cci_diabetes_w_chronic_complication_diagnosis': 2,
             'cci_hemiplegia_paraplegia_diagnosis': 2,
             'cci_renal_disease_diagnosis': 2,
             #'cci_moderate_severe_liver_disease_diagnosis': 3,
             'cci_metastatic_solid_tumor_diagnosis': 6,
             'cci_hiv_aids_diagnosis': 6}  #missing dialysis, leukemia, lymphoma for cci.
        
        # calculate age adjusted cci
        data_df['cci'] = 0
        for condition, points in cci_points.items():
            data_df['cci'] += data_df[condition] * points
        # diabetes
        data_df['cci'] += pd.DataFrame({
             'wo_chronic_complication':data_df['cci_diabetes_wo_chronic_complication_diagnosis']*1,
             'w_chronic_complication':data_df['cci_diabetes_w_chronic_complication_diagnosis']*2
            }).max(axis=1)
        data_df['cci_diabetes_diagnosis'] = pd.DataFrame({
             'wo_chronic_complication':data_df['cci_diabetes_wo_chronic_complication_diagnosis']*1,
             'w_chronic_complication':data_df['cci_diabetes_w_chronic_complication_diagnosis']*2
            }).max(axis=1)
        # liver
        data_df['cci'] += pd.DataFrame({
             'mild':data_df['cci_mild_liver_disease_diagnosis']*1,
             'severe':data_df['cci_moderate_severe_liver_disease_diagnosis']*3
            }).max(axis=1)
        data_df['cci_liver_disease_diagnosis'] = pd.DataFrame({
             'mild':data_df['cci_mild_liver_disease_diagnosis']*1,
             'servere':data_df['cci_diabetes_w_chronic_complication_diagnosis']*2
            }).max(axis=1)
        # tumor
        data_df['cci'] += pd.DataFrame({
             'wo':data_df['cci_malignancy_diagnosis']*2,
             'w':data_df['cci_moderate_severe_liver_disease_diagnosis']*6
            }).max(axis=1)
        #data_df['cci_solid_tumor_diagnosis'] = pd.DataFrame({
         #   'wo':data_df['cci_malignancy_diagnosis']*1,
          #  'we':data_df['cci_metastatic_solid_tumor_diagnosi']*2
           # }).max(axis=1)
        # age adjustment for cci
        data_df['cci']+=pd.cut(data_df['age'], bins=[-float('inf'), 49, 59, 69, 79, float('inf')],labels=[0, 1, 2, 3, 4]).astype(int)
        # calcualte eci index
        data_df['eci'] = 0
        for condition, points in eci_points.items():
            data_df['eci'] += data_df[condition] * points
        # eci diabetes
        data_df['eci_diabetes_diagnosis'] = pd.DataFrame({
            'wo_chronic_complication':data_df['eci_diabetes_uncomplicated_diagnosis']*1,
            'w_chronic_complication':data_df['eci_diabetes_complicated_diagnosis']*2
        }).max(axis=1)
        # eci hypertension
        data_df['eci_hypertension_diagnosis'] = pd.DataFrame({
             'wo_chronic_complication':data_df['eci_hypertension_uncomplicated_diagnosis']*1,
             'w_chronic_complication':data_df['eci_hypertension_complicated_diagnosis']*2
        }).max(axis=1)
        # drop combined (not need) 
        
        #dialysis
        # List of ICD 9/10 codes related to dialysis
        dialysis_codes = [
            '585.x', 'V45.1', 'V42.1', 'V56.x', '588', '586', '583', '582', 
            '404.92', '404.93', '404.12', '404.13', '404.02', '404.03', 
            '403.01', '403.11', '403.91'
        ]

        # Function to check if any dialysis code is in the problem list
        def check_dialysis(problem_list):
            if pd.isna(problem_list):
                return 0
            if isinstance(problem_list, float):
                problem_list = str(problem_list)
            problems = problem_list.split(';')
            for code in dialysis_codes:
                if 'x' in code:
                    code_prefix = code.replace('x', '')
                    if any(problem.startswith(code_prefix) for problem in problems):
                        return 1
                else:
                    if any(problem == code for problem in problems):
                        return 1
            return 0

        # Apply the function to create the new column
        data_df['cci_dialysis'] = data_df['problem_list'].apply(check_dialysis)
        
        return data_df
        
    # if repeated process? no, it adds control to the data
    def _create_picis_features(self, data_df):
        # explode picis codes to one in a row w/ duplicated an_episode_id
        X = data_df[['an_episode_id', 'picis_codes']]
        df_picis_codes = (
            X.assign(picis_codes=X.picis_codes.str.split(";"))
            .explode('picis_codes')
            .reset_index(drop=True)
        )

        # Create dummy variable if not in list of transfused codes
        df_picis_codes.loc[
                ~df_picis_codes.picis_codes.isin(self.df_PICIS_txn),
                'picis_codes'
            ] = 'PI0000'

        # Create dummy variable of NA pices codes
        df_picis_codes['picis_codes'] = (
            df_picis_codes['picis_codes'].fillna('NA0000')
        )
        # create a df w/ transformed indicator matrix 
        # (which uses the picis_one_hot_encoder fitted above)
        df = pd.DataFrame(
            (
                self.picis_one_hot_encoder.transform(
                    df_picis_codes[['picis_codes']]
                )
                .astype(int)
            ),
            columns=self.picis_one_hot_feature_names
        )
        # concat the id column the the df above
        df_picis_codes = pd.concat(
            [df_picis_codes[['an_episode_id']], df], axis=1
        )
        
        return (
            # Groupby id & sum up the score of each feature for every id 
            # and merge to the original data_df
            data_df.merge(
                (   # reset_index() to maintain before groupby index 
                    # and make an_episode_id a regular column 
                    df_picis_codes.groupby('an_episode_id', as_index=False)
                    [self.picis_one_hot_feature_names].sum()
                ),
                on='an_episode_id',
                how='left'
            )
        )

    # skipna = True for calculations like sum(), mean() to work
    def _set_hist_transfused(self, data_df, skipna=True, transfusion_columns=None):
        ## Prepare df_transfusion
        # Default transfusion columns
        if transfusion_columns is None:
            transfusion_columns = [
                'periop_prbc_units_transfused',
                'periop_platelets_units_transfused',
                'periop_ffp_units_transfused',
                'periop_cryoprecipitate_units_transfused'
            ]
        # Drop existing hist_transfused columns if they exist
        for col in transfusion_columns:
            hist_col_name = f'hist_transfused_{col.split("_")[1]}'
            if hist_col_name in data_df.columns:
                data_df.drop(columns=[hist_col_name], inplace=True)
        
        # Split picis_codes and explode into separate rows & rename na
        df_transfusion = (
            data_df.assign(
                picis_codes=data_df.picis_codes.str.split(";")
            ).explode('picis_codes')
            .reset_index(drop=True)
            .fillna('NA0000')
        )
        
        ## Match picis_code & merge transfusion rates to df_transfusion by episode_id
        # Step 1: Read the picis_mean.csv
        picis_mean = self.transfusion_rates_by_picis
        '''
        ## check unique picis_codes btw both df
        # Get the unique values from each DataFrame's picis_codes column
        unique_transfusion = set(df_transfusion['picis_codes'].unique())
        unique_mean = set(picis_mean['picis_codes'].unique())

        # Compare the unique values to see if they are the same
        
        if unique_transfusion == unique_mean:
            print("The unique values in 'picis_codes' are the same in both DataFrames.")
        else:
            print("The unique values in 'picis_codes' are different in the two DataFrames.")
    
            # Step 3: Count the number of differences
            diff_transfusion_to_mean = unique_transfusion - unique_mean
            diff_mean_to_transfusion = unique_mean - unique_transfusion
    
            print(f"Number of unique 'picis_codes' in df_transfusion not in picis_mean: {len(diff_transfusion_to_mean)}")
            print(f"Number of unique 'picis_codes' in picis_mean not in df_transfusion: {len(diff_mean_to_transfusion)}")
            
            #print(f"Unique 'picis_codes' in df_transfusion not in picis_mean: {diff_transfusion_to_mean}")
            '''
        # Step 2: Merge the two DataFrames based on picis_codes
        merged_df = df_transfusion.merge(picis_mean, on='picis_codes', how='left')
        ## Fill.na of the merged_df 'index' column w/ index of 'NA0000'
        # Identify the row where picis_codes is 'NA0000'
        na0000_row = picis_mean[picis_mean['picis_codes'] == 'NA0000'].iloc[0]
        # Columns to fill
        fill_columns = ['index', 'hist_transfused_prbc', 'hist_transfused_platelets',
                                        'hist_transfused_ffp', 'hist_transfused_cryoprecipitate', 'any_transfusion_rate']
        # Fill NaN values in the specified columns with values from the 'NA0000' row
        for col in fill_columns:
            merged_df[col].fillna(na0000_row[col], inplace=True)
        # Step 3: Group by episode_id and get the row with the lowest index
        grouped_df = merged_df.loc[merged_df.groupby('an_episode_id')['index'].idxmin()]
        # Step 4: Select the necessary columns and merge back to data_df
        transfusion_rates = grouped_df[['an_episode_id', 'hist_transfused_prbc', 'hist_transfused_platelets',
                                        'hist_transfused_ffp', 'hist_transfused_cryoprecipitate', 'any_transfusion_rate']]
        result_df = data_df.merge(transfusion_rates, on='an_episode_id', how='left')
        return result_df
        
    def fit_transform(self, data_df):
        # recode prior to any fit/transform
        data_df = self._calculate_comorbidities(data_df) 
        # get and set transfusion rate
        self._get_picis_code_with_transfusion(data_df) 
        data_df = self._set_hist_transfused(data_df)
        # one-hot encoding
        self.categorical_features = [
            'sched_case_class', #?
            'sched_surgical_service',
            'sched_prim_surgeon_provid_1',
            'prepare_asa',
            'prepare_asa_e',
            'sched_proc_max_complexity',
            'sched_surgical_dept_campus',
            'prior_dept_location',
             # eci cat
            'eci_hypertension_diagnosis', 
            'eci_diabetes_diagnosis', 
            # cci - cat
            'cci_diabetes_diagnosis', 
            'cci_liver_disease_diagnosis'
        ]
        self.continuous_features = [
            # dem_var
            'age',
            # bmi_var
            'weight_kg', 'height_cm', 'bmi',
            # case_cont
            'sched_est_case_length', 'sched_surgeon_cnt',
            'sched_proc_cnt', 'sched_proc_diag_cnt',
            # day_cont
            'enc_los_to_surg', 'hist_prior_transf_platelets_days',
            'hist_prior_transf_prbc_days',
            'hist_prior_transf_ffp_days',
            'hist_prior_transf_cryoprecipitate_days',
        # ? why commented out
            # 'hist_prior_transf_platelets_days_toanesstart',
            # 'hist_prior_transf_prbc_days_toanesstart',
            # 'hist_prior_transf_ffp_days_toanesstart',
            # 'hist_prior_transf_cryoprecipitate_days_toanesstart',
            # labs_var
            'preop_base_excess_abg',
            'preop_base_excess_vbg', 'preop_bicarbonate_abg',
            'preop_bicarbonate_vbg', 'preop_bun', 'preop_chloride',
            'preop_chloride_abg', 'preop_chloride_vbg', 'preop_creatinine',
            'preop_hematocrit', 'preop_hematocrit_from_hb_abg',
            'preop_hematocrit_from_hb_vbg', 'preop_hemoglobin',
            'preop_hemoglobin_abg', 'preop_hemoglobin_vbg',
            'preop_lymphocyte_cnt', 'preop_neutrophil_cnt', 'preop_pco2_abg',
            'preop_pco2_vbg', 'preop_platelets', 'preop_ph_abg',
            'preop_ph_vbg', 'preop_potassium', 'preop_potassium_abg',
            'preop_potassium_vbg', 'preop_rbc', 'preop_sodium',
            'preop_sodium_abg', 'preop_sodium_vbg', 'preop_wbc','preop_albumin_serum','preop_bilirubin_total','preop_bilirubin_direct','preop_bilirubin_indirect',
            'preop_inr', 'preop_ptt',
            # comorbidity index
            #'cci','eci',
            # msbos
            'msbos_cnt', 'msbos_wb_cnt', 'msbos_rbc_cnt'] + self.df_tx_picis_means.columns[2:].tolist() #transfusion rate
           
           
        self.binary_features = [
            # case
            'sched_addon_yn', 'sched_emergency_case',
            'sched_or_equip_cell_saver_yn', 'sched_neuromonitoring_yn',
            'prepare_visit_yn', 'sched_bypass_yn',
            # adt
            'hist_transf_1week_yn', 'hist_transf_1day_yn',
            'hist_prior_transf_yn',
            'language_interpreter_needed_yn', 'language_english_yn',
            'arrival_ed_yn', 'icu_admit_prior_24hr_yn', 'prior_dept_inpt_yn',
            # msbos
            'msbos_ts', 'msbos_tc', 'msbos_cryo', 'msbos_prbc', 'msbos_ffp',
            'msbos_platelets', 'msbos_bppp', 'msbos_wholeblood',
            # home meds
            'home_meds_anticoag_warfarin_yn',
            'home_meds_anticoag_heparin_sq_yn',
            'home_meds_anticoag_heparin_iv_yn',
            'home_meds_anticoag_fondaparinux_yn',
            'home_meds_anticoag_exoxaparin_yn',
            'home_meds_anticoag_argatroban_yn',
            'home_meds_anticoag_bivalirudin_yn',
            'home_meds_anticoag_lepirudin_yn',
            'home_meds_anticoag_dabigatran_yn',
            'home_meds_anticoag_clopidogrel_yn',
            'home_meds_anticoag_prasugrel_yn',
            'home_meds_anticoag_ticlodipine_yn',
            'home_meds_anticoag_abxicimab_yn',
            'home_meds_anticoag_eptifibatide_yn',
            'home_meds_anticoag_tirofiban_yn',
            'home_meds_anticoag_alteplase_yn',
            'home_meds_anticoag_apixaban_yn',
            # dem
            'sex_female',
            'sex_male',
            'sex_nonbinary',
            'sex_unknown',
         # comorbidities - eci binary
            'eci_alcohol_abuse_diagnosis', 
            'eci_blood_loss_anemia_diagnosis',
            'eci_cardiac_arrhythmias_diagnosis',
            'eci_chf_diagnosis', 
            'eci_chronic_pulmonary_disease_diagnosis',
            'eci_coagulopathy_diagnosis', 
            'eci_deficiency_anemia_diagnosis',
            'eci_depression_diagnosis', 
            'eci_drug_abuse_diagnosis',
            'eci_fluid_disorder_diagnosis', 
            'eci_hiv_aids_diagnosis', 
            'eci_hypothyroidism_diagnosis', 
            'eci_liver_disease_diagnosis', 
            'eci_lymphoma_diagnosis', 
            'eci_metastatic_cancer_diagnosis', 
            'eci_obesity_diagnosis',
            'eci_other_neurological_disorders_diagnosis',
            'eci_paralysis_diagnosis', 
            'eci_peptic_ulcer_disease_diagnosis',
            'eci_peripheral_vascular_disease_diagnosis',
             'eci_psychoses_diagnosis',
            'eci_pulmonary_circulation_disease_diagnosis', 
            'eci_renal_failure_diagnosis', 
            'eci_rheumatic_disease_diagnosis',
            'eci_solid_tumor_wo_metastatic_diagnosis',
            'eci_valvular_disease_diagnosis', 
            'eci_weight_loss_diagnosis',
             # comorbidities - cci binary
            'cci_cerebrovascular_disease_diagnosis',
            'cci_chf_diagnosis', 
            'cci_chronic_pulmonary_disease_diagnosis',
            'cci_dementia_diagnosis', 
            'cci_hemiplegia_paraplegia_diagnosis', 
            'cci_hiv_aids_diagnosis', 
            'cci_malignancy_diagnosis',
            'cci_metastatic_solid_tumor_diagnosis', 
            'cci_mi_diagnosis', 
            'cci_peptic_ulcer_disease_diagnosis',
            'cci_peripheral_vascular_disease_diagnosis',
            'cci_renal_disease_diagnosis', 
            'cci_rheumatic_disease_diagnosis',
            'has_preop_dialysis']
            #'cci_dialysis'] 
        # one hot encoder
        self.one_hot_encoder = OneHotEncoder(
            sparse=False, handle_unknown='infrequent_if_exist'
        )
        self.one_hot_encoder.fit(data_df[self.categorical_features])
        self.one_hot_feature_names = list(
            self.one_hot_encoder.get_feature_names_out()
        )

        # %%Features to Normalize using Standard Scalar
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(data_df[self.continuous_features])

        # final list of features
        self.features = (
            self.continuous_features
            + self.one_hot_feature_names 
            + self.binary_features 
            #+ self.picis_one_hot_feature_names
        )
        return self.transform(data_df,transform=False)

    def transform(self, data_df, transform=True):
        if transform:
        #if 'cci' not in data_df.columns:
            data_df = self._calculate_comorbidities(data_df) 
        #if 'hist_transfused' not in data_df.columns: 
            data_df = self._set_hist_transfused(data_df)
              
        data_df[self.continuous_features] = self.standard_scaler.transform(
                data_df[self.continuous_features]
            )

        df = pd.DataFrame(
            (
                self.one_hot_encoder.transform(
                    data_df[self.categorical_features]
                )
                .astype(int)
            ),
            columns=self.one_hot_feature_names
        )
        df['an_episode_id'] = data_df['an_episode_id']

        data_df = data_df.merge(df, how='left', on=['an_episode_id'])

        #data_df = self._create_picis_features(data_df)

        return data_df

'''
def train_valid_test_split(
        df, target, train_size=0.8, valid_size=0.1,
        test_size=0.1, method='random', sort_by_col = None, random_state=None):
    #''
    For a given input dataframe this prepares X_train, y_train, X_valid,
    y_valid, X_test, y_test for final model development

    Parameters:
    -----------
    df: 'dataframe', input dataframe
    target: 'str' , target variable
    train_size: 'float', proportion of train dataset
    valid_size: 'float', proportion of valid dataset
    test_size: 'float', proportion of test dataset
    method: 'str', default 'random'.
    2 methods available ['random', 'sorted']. in sorted dataframe is sorted by
    the input column and then splitting is done
    sort_by_col : 'str', defaul None. Required when method = 'sorted'
    random_state : random_state for train_test_split


    Output:
    -------
    X_train, y_train, X_valid, y_valid, X_test, y_test

    #''

    total = train_size + valid_size + test_size
    if total>1:
        raise Exception(
            " Total of train_size + valid_size + test_size should be 1"
        )
    else:

        if method=='random':
            df_train, df_rem = train_test_split(df, train_size=train_size, random_state=random_state)
            test_prop = test_size/(test_size+valid_size)
            df_valid, df_test = train_test_split(df_rem, test_size=test_prop, random_state=random_state)

            X_train, y_train = (
                df_train.drop(columns=target).copy(), df_train[target].copy()
            )
            X_valid, y_valid = (
                df_valid.drop(columns=target).copy(), df_valid[target].copy()
            )
            X_test, y_test = (
                df_test.drop(columns=target).copy(), df_test[target].copy()
            )

        if method == 'sorted':
            train_index = int(len(df)*train_size)
            valid_index = int(len(df)*valid_size)

            df.sort_values(by = sort_by_col, ascending=True, inplace=True)
            df_train = df[0:train_index]
            df_rem = df[train_index:]
            df_valid = df[train_index:train_index+valid_index]
            df_test = df[train_index+valid_index:]

            X_train, y_train = (
                df_train.drop(columns=target).copy(), df_train[target].copy()
            )
            X_valid, y_valid = (
                df_valid.drop(columns=target).copy(), df_valid[target].copy()
            )
            X_test, y_test = (
                df_test.drop(columns=target).copy(), df_test[target].copy()
            )


        return X_train, y_train, X_valid, y_valid, X_test, y_test
    '''
def train_valid_test_split(
        df, target, train_size=0.8, valid_size=0.1,
        test_size=0.1, method='random', sort_by_col=None, 
        random_state=None, episode_ids=None):
    '''
    For a given input dataframe, this prepares X_train, y_train, X_valid,
    y_valid, X_test, y_test for final model development

    Parameters:
    -----------
    df: 'dataframe', input dataframe
    target: 'str', target variable
    train_size: 'float', proportion of train dataset
    valid_size: 'float', proportion of valid dataset
    test_size: 'float', proportion of test dataset
    method: 'str', default 'random'.
    2 methods available ['random', 'sorted']. In sorted, dataframe is sorted by
    the input column and then splitting is done
    sort_by_col: 'str', default None. Required when method = 'sorted'
    random_state: random_state for train_test_split
    episode_ids: list of 'an_episode_id'. Splits those for training and then 
    evenly splits validation and test sets ignoring the split percent rule

    Output:
    -------
    X_train, y_train, X_valid, y_valid, X_test, y_test
    '''
    
    total = train_size + valid_size + test_size
    if total > 1:
        raise Exception("Total of train_size + valid_size + test_size should be 1")
    
    if episode_ids is not None:
        # Split based on episode_ids
        df_train = df[df['an_episode_id'].isin(episode_ids)]
        df_rem = df[~df['an_episode_id'].isin(episode_ids)]
        
        if method == 'sorted' and sort_by_col is not None:
            df_rem = df_rem.sort_values(by=sort_by_col, ascending=True)
        
        # Debugging statements
        print("Records in df_train: ", len(df_train))
        print("Records in df_rem: ", len(df_rem))
        
        # Evenly split remaining data for validation and test sets
        df_valid, df_test =  train_test_split(df_rem, test_size=0.5, shuffle=False)
        
        X_train, y_train = df_train.drop(columns=target).copy(), df_train[target].copy()
        X_valid, y_valid = df_valid.drop(columns=target).copy(), df_valid[target].copy()
        X_test, y_test = df_test.drop(columns=target).copy(), df_test[target].copy()
    
    else:
        if method == 'random':
            df_train, df_rem = train_test_split(df, train_size=train_size, random_state=random_state)
            test_prop = test_size / (test_size + valid_size)
            df_valid, df_test = train_test_split(df_rem, test_size=test_prop, random_state=random_state)
            
            X_train, y_train = df_train.drop(columns=target).copy(), df_train[target].copy()
            X_valid, y_valid = df_valid.drop(columns=target).copy(), df_valid[target].copy()
            X_test, y_test = df_test.drop(columns=target).copy(), df_test[target].copy()
        
        elif method == 'sorted':
            if sort_by_col is None:
                raise Exception("sort_by_col must be provided when method is 'sorted'")
            
            df = df.sort_values(by=sort_by_col, ascending=True)
            train_index = int(len(df) * train_size)
            valid_index = int(len(df) * valid_size)
            
            df_train = df[:train_index]
            df_rem = df[train_index:]
            df_valid = df_rem[:valid_index]
            df_test = df_rem[valid_index:]
            
            X_train, y_train = df_train.drop(columns=target).copy(), df_train[target].copy()
            X_valid, y_valid = df_valid.drop(columns=target).copy(), df_valid[target].copy()
            X_test, y_test = df_test.drop(columns=target).copy(), df_test[target].copy()
    
    # Debugging statements
    print("Records in X_train: ", len(X_train))
    print("Records in X_valid: ", len(X_valid))
    print("Records in X_test: ", len(X_test))
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test