"""
Feature engineering module for the main application data.
"""

import numpy as np
import pandas as pd
import gc
from utils import reduce_mem_usage
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

def engineer_main_features(X):
    """
    Engineer features from the main application data.
    
    Args:
        X: DataFrame with main application data
        
    Returns:
        DataFrame with engineered features
    """
    # Replace specific placeholder values with NaN as done in notebook's preprocess_main
    if 'DAYS_EMPLOYED' in X.columns:
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
    if 'DAYS_LAST_PHONE_CHANGE' in X.columns:
        X['DAYS_LAST_PHONE_CHANGE'] = X['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan)
    # Assuming XNA/XAP already handled by preprocessing.replace_XNA_XAP if called before this function
    # X['CODE_GENDER'].replace('XNA', np.nan, inplace=True) # Example if needed
    # X['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True) # Example if needed
    if 'NAME_FAMILY_STATUS' in X.columns:
        X['NAME_FAMILY_STATUS'] = X['NAME_FAMILY_STATUS'].replace('Unknown', np.nan)
    if 'NAME_INCOME_TYPE' in X.columns:
        X['NAME_INCOME_TYPE'] = X['NAME_INCOME_TYPE'].replace('Maternity leave', np.nan)
    if 'REGION_RATING_CLIENT_W_CITY' in X.columns:
        X['REGION_RATING_CLIENT_W_CITY'] = X['REGION_RATING_CLIENT_W_CITY'].replace(-1, np.nan)

    # Calculate EXT_SOURCE_MEAN early as it's needed for ratio features
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    # Use fillna(0) temporarily if any source is missing, though imputation might be better
    X['EXT_SOURCE_MEAN'] = X[ext_sources].fillna(0).mean(axis=1)

    # Calculate Age/Employment Years early as they might be needed for ratios
    if 'DAYS_BIRTH' in X.columns:
        X['AGE_YEARS'] = X['DAYS_BIRTH'] / -365.25
    if 'DAYS_EMPLOYED' in X.columns:
        X['YEARS_EMPLOYED'] = X['DAYS_EMPLOYED'] / -365.25

    aggregation_recipes = [
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
                                                  ('AMT_CREDIT', 'max'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('OWN_CAR_AGE', 'max'),
                                                  ('OWN_CAR_AGE', 'sum')]),
        (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                                ('AMT_INCOME_TOTAL', 'mean'),
                                                ('DAYS_REGISTRATION', 'mean'),
                                                ('EXT_SOURCE_1', 'mean')]),
        (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                     ('CNT_CHILDREN', 'mean'),
                                                     ('DAYS_ID_PUBLISH', 'mean')]),
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                               ('EXT_SOURCE_2', 'mean')]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                      ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                      ('APARTMENTS_AVG', 'mean'),
                                                      ('BASEMENTAREA_AVG', 'mean'),
                                                      ('EXT_SOURCE_1', 'mean'),
                                                      ('EXT_SOURCE_2', 'mean'),
                                                      ('EXT_SOURCE_3', 'mean'),
                                                      ('NONLIVINGAREA_AVG', 'mean'),
                                                      ('OWN_CAR_AGE', 'mean'),
                                                      ('YEARS_BUILD_AVG', 'mean')]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                                ('EXT_SOURCE_1', 'mean')]),
        (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                               ('CNT_CHILDREN', 'mean'),
                               ('CNT_FAM_MEMBERS', 'mean'),
                               ('DAYS_BIRTH', 'mean'),
                               ('DAYS_EMPLOYED', 'mean'),
                               ('DAYS_ID_PUBLISH', 'mean'),
                               ('DAYS_REGISTRATION', 'mean'),
                               ('EXT_SOURCE_1', 'mean'),
                               ('EXT_SOURCE_2', 'mean'),
                               ('EXT_SOURCE_3', 'mean')]),
    ]
    
    # Groupby categorical features, calculate the mean and or max 
    # of various numerical statistics.
    for groupby_cols, specs in aggregation_recipes:
        group_object = X.groupby(groupby_cols, observed=False)
        for select, agg in specs:
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg.upper(), select)
            X = X.merge(group_object[select]
                                  .agg(agg)
                                  .reset_index()
                                  .rename(index=str,
                                          columns={select: groupby_aggregate_name})
                                  [groupby_cols + [groupby_aggregate_name]],
                                  on=groupby_cols,
                                  how='left')
            
    # Get the difference and absolute difference between two 
    # categorical features' mean and or max values of various
    # numerical statistics.
    for groupby_cols, specs in aggregation_recipes:
        for select, agg in specs:
            if agg in ['mean','median','max','min']:
                groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg.upper(), select)
                diff_name = '{}_DIFF'.format(groupby_aggregate_name)
                abs_diff_name = '{}_ABS_DIFF'.format(groupby_aggregate_name)

                X[diff_name] = X[select] - X[groupby_aggregate_name] 
                X[abs_diff_name] = np.abs(X[select] - X[groupby_aggregate_name]) 

    # Ratios - Be careful with division by zero or NaN
    # Ensure relevant columns are numeric first
    ratio_cols_numeric = [
        'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_ADULT_FAM_MEMBER', 'AMT_INCOME_TOTAL',
        'AMT_GOODS_PRICE', 'LIVINGAREA_AVG', 'LANDAREA_AVG', 'FLOORSMAX_AVG',
        'LIVINGAPARTMENTS_AVG', 'YEARS_BUILD_AVG', 'DAYS_EMPLOYED', 'CNT_CHILDREN',
        'SUM_AMT_INCOME_TOTAL_AMT_ANNUITY', 'EXT_SOURCE_3', 'REGION_POPULATION_RELATIVE',
        'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION', 'DAYS_BIRTH', 'CNT_FAM_MEMBERS',
        'TOTAL_ENQUIRIES_CREDIT_BUREAU', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    for col in ratio_cols_numeric:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    print("Creating ratio features (using concat)...")
    ratio_features = pd.DataFrame(index=X.index)
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore division by zero or NaN warnings
        ratio_features['RATIO_AMT_CREDIT_TO_AMT_GOODS_PRICE'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE'].replace(0, np.nan)
        ratio_features['RATIO_AMT_CREDIT_TO_AMT_INCOME_TOTAL'] = X['AMT_CREDIT'] / (1 + X['AMT_INCOME_TOTAL'])
        ratio_features['RATIO_AMT_CREDIT_TO_AMT_ANNUITY'] = X['AMT_CREDIT'] / X['AMT_ANNUITY'].replace(0, np.nan)
        ratio_features['RATIO_AMT_CREDIT_TO_EXT_SOURCE_MEAN'] = X['AMT_CREDIT'] / (1 + X['EXT_SOURCE_MEAN'])
        ratio_features['RATIO_AMT_ANNUITY_AMT_CREDIT'] = X['AMT_ANNUITY'] / X['AMT_CREDIT'].replace(0, np.nan)
        ratio_features['RATIO_AMT_ANNUITY_TO_AMT_GOODS_PRICE'] = X['AMT_ANNUITY'] / X['AMT_GOODS_PRICE'].replace(0, np.nan)
        ratio_features['RATIO_AMT_ANNUITY_TO_AMT_INCOME_TOTAL'] = X['AMT_ANNUITY'] / (1 + X['AMT_INCOME_TOTAL'])
        ratio_features['RATIO_AMT_INCOME_TOTAL_TO_AMT_CREDIT'] = X['AMT_INCOME_TOTAL'] / (1 + X['AMT_CREDIT'])
        ratio_features['RATIO_AMT_INCOME_TOTAL_TO_AMT_ANNUITY'] = X['AMT_INCOME_TOTAL'] / X['AMT_ANNUITY'].replace(0, np.nan)
        ratio_features['RATIO_AMT_INCOME_TOTAL_TO_AMT_GOODS_PRICE'] = X['AMT_INCOME_TOTAL'] / X['AMT_GOODS_PRICE'].replace(0, np.nan)
        ratio_features['RATIO_AMT_INCOME_TOTAL_TO_AGE_YEARS'] = X['AMT_INCOME_TOTAL'] / (1 + X['AGE_YEARS'])
        ratio_features['RATIO_AMT_INCOME_TOTAL_TO_EMPLOYED_YEARS'] = X['AMT_INCOME_TOTAL'] / (1 + X['YEARS_EMPLOYED'])
        ratio_features['RATIO_CNT_CHILDREN_TO_CNT_FAM_MEMBERS'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS'].replace(0, np.nan)

        ratio_features['RATIO_EXT_SOURCE_1_TO_AGE_YEARS'] = X['EXT_SOURCE_1'] / (1 + X['AGE_YEARS'])
        ratio_features['RATIO_EXT_SOURCE_2_TO_AGE_YEARS'] = X['EXT_SOURCE_2'] / (1 + X['AGE_YEARS'])
        ratio_features['RATIO_EXT_SOURCE_3_TO_AGE_YEARS'] = X['EXT_SOURCE_3'] / (1 + X['AGE_YEARS'])
        ratio_features['RATIO_EXT_SOURCE_MEAN_TO_AGE_YEARS'] = X['EXT_SOURCE_MEAN'] / (1 + X['AGE_YEARS'])

        ratio_features['RATIO_EXT_SOURCE_1_TO_EMPLOYED_YEARS'] = X['EXT_SOURCE_1'] / (1 + X['YEARS_EMPLOYED'])
        ratio_features['RATIO_EXT_SOURCE_2_TO_EMPLOYED_YEARS'] = X['EXT_SOURCE_2'] / (1 + X['YEARS_EMPLOYED'])
        ratio_features['RATIO_EXT_SOURCE_3_TO_EMPLOYED_YEARS'] = X['EXT_SOURCE_3'] / (1 + X['YEARS_EMPLOYED'])
        ratio_features['RATIO_EXT_SOURCE_MEAN_TO_EMPLOYED_YEARS'] = X['EXT_SOURCE_MEAN'] / (1 + X['YEARS_EMPLOYED'])

        ratio_features['RATIO_AMT_GOODS_PRICE_TO_LIVINGAREA_AVG'] = X['AMT_GOODS_PRICE'] / X['LIVINGAREA_AVG'].replace(0, np.nan)
        ratio_features['RATIO_AMT_GOODS_PRICE_TO_LANDAREA_AVG'] = X['AMT_GOODS_PRICE'] / X['LANDAREA_AVG'].replace(0, np.nan)
        ratio_features['RATIO_AMT_GOODS_PRICE_TO_FLOORSMAX_AVG_AVG'] = X['AMT_GOODS_PRICE'] / X['FLOORSMAX_AVG'].replace(0, np.nan)
        ratio_features['RATIO_AMT_GOODS_PRICE_TO_LIVINGAPARTMENTS_AVG'] = X['AMT_GOODS_PRICE'] / X['LIVINGAPARTMENTS_AVG'].replace(0, np.nan)

    ratio_features = ratio_features.replace([np.inf, -np.inf], np.nan)
    X = pd.concat([X, ratio_features], axis=1)
    print(f"Shape after ratio features: {X.shape}")
    
    # Add Z-Score features for key numerical variables
    print("Creating Z-Score features for key numerical variables...")
    z_score_features = pd.DataFrame(index=X.index)
    key_numeric_features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
        'DAYS_LAST_PHONE_CHANGE'
    ]

    for feature in key_numeric_features:
        if feature in X.columns:
            # Ensure numeric
            X[feature] = pd.to_numeric(X[feature], errors='coerce')
            mean = X[feature].mean()
            std = X[feature].std()
            if std > 0:  # Avoid division by zero
                z_score_features[f'Z_{feature}'] = (X[feature] - mean) / std

    X = pd.concat([X, z_score_features], axis=1)
    print(f"Shape after Z-Score features: {X.shape}")
    del z_score_features; gc.collect()

    print("Creating flag and interaction features (using concat)...")
    flag_interaction_features = pd.DataFrame(index=X.index)
    docs = [col for col in X.columns if col.startswith('FLAG_DOCUMENT')]
    flag_interaction_features['FLAG_DOCUMENTS_TOTAL'] = X[docs].sum(axis=1)
    flag_interaction_features['FLAG_DOCUMENTS_PROVIDED_RATIO'] = flag_interaction_features['FLAG_DOCUMENTS_TOTAL'] / len(docs)

    flag_interaction_features['FLAG_AGE_GT_EMPLOYED'] = (X['AGE_YEARS'] > X['YEARS_EMPLOYED']).astype(int)
    flag_interaction_features['INTERACTION_AGE_EMPLOYED'] = X['AGE_YEARS'] * X['YEARS_EMPLOYED']
    flag_interaction_features['FLAG_DAYS_LAST_PHONE_CHANGE_NEGATIVE'] = (X['DAYS_LAST_PHONE_CHANGE'] < 0).astype(int)

    flag_interaction_features['INTERACTION_AMT_INCOME_CREDIT'] = X['AMT_INCOME_TOTAL'] * X['AMT_CREDIT']
    flag_interaction_features['INTERACTION_AMT_INCOME_ANNUITY'] = X['AMT_INCOME_TOTAL'] * X['AMT_ANNUITY']
    flag_interaction_features['INTERACTION_AMT_INCOME_GOODS'] = X['AMT_INCOME_TOTAL'] * X['AMT_GOODS_PRICE']
    flag_interaction_features['INTERACTION_AMT_CREDIT_ANNUITY'] = X['AMT_CREDIT'] * X['AMT_ANNUITY']
    flag_interaction_features['INTERACTION_AMT_CREDIT_GOODS'] = X['AMT_CREDIT'] * X['AMT_GOODS_PRICE']
    flag_interaction_features['INTERACTION_AMT_ANNUITY_GOODS'] = X['AMT_ANNUITY'] * X['AMT_GOODS_PRICE']

    flag_interaction_features['INTERACTION_EXT_SOURCE_1_AGE'] = X['EXT_SOURCE_1'] * X['AGE_YEARS']
    flag_interaction_features['INTERACTION_EXT_SOURCE_2_AGE'] = X['EXT_SOURCE_2'] * X['AGE_YEARS']
    flag_interaction_features['INTERACTION_EXT_SOURCE_3_AGE'] = X['EXT_SOURCE_3'] * X['AGE_YEARS']
    flag_interaction_features['INTERACTION_EXT_SOURCE_MEAN_AGE'] = X['EXT_SOURCE_MEAN'] * X['AGE_YEARS']

    flag_interaction_features['INTERACTION_EXT_SOURCE_1_EMPLOYED'] = X['EXT_SOURCE_1'] * X['YEARS_EMPLOYED']
    flag_interaction_features['INTERACTION_EXT_SOURCE_2_EMPLOYED'] = X['EXT_SOURCE_2'] * X['YEARS_EMPLOYED']
    flag_interaction_features['INTERACTION_EXT_SOURCE_3_EMPLOYED'] = X['EXT_SOURCE_3'] * X['YEARS_EMPLOYED']
    flag_interaction_features['INTERACTION_EXT_SOURCE_MEAN_EMPLOYED'] = X['EXT_SOURCE_MEAN'] * X['YEARS_EMPLOYED']

    flag_interaction_features['YEARS_EMPLOYED_PERCENT'] = X['YEARS_EMPLOYED'] / X['AGE_YEARS']

    X = pd.concat([X, flag_interaction_features], axis=1)
    print(f"Shape after flag/interaction features: {X.shape}")
    
    # Add Z-Score for important ratio features
    print("Creating Z-Score features for important ratio features...")
    ratio_z_features = pd.DataFrame(index=X.index)
    key_ratio_features = [
        'RATIO_AMT_CREDIT_TO_AMT_INCOME_TOTAL',
        'RATIO_AMT_ANNUITY_TO_AMT_INCOME_TOTAL',
        'RATIO_AMT_CREDIT_TO_AMT_ANNUITY',
        'RATIO_AMT_CREDIT_TO_AMT_GOODS_PRICE',
        'RATIO_AMT_INCOME_TOTAL_TO_AMT_CREDIT',
        'RATIO_AMT_INCOME_TOTAL_TO_AMT_ANNUITY',
        'RATIO_EXT_SOURCE_MEAN_TO_AGE_YEARS',
        'RATIO_EXT_SOURCE_MEAN_TO_EMPLOYED_YEARS'
    ]

    for feature in key_ratio_features:
        if feature in X.columns:
            # Replace infinities with NaN first
            X[feature] = X[feature].replace([np.inf, -np.inf], np.nan)
            mean = X[feature].mean()
            std = X[feature].std()
            if std > 0:  # Avoid division by zero
                ratio_z_features[f'Z_{feature}'] = (X[feature] - mean) / std

    X = pd.concat([X, ratio_z_features], axis=1)
    print(f"Shape after ratio Z-Score features: {X.shape}")
    del ratio_z_features; gc.collect()

    print("Creating polynomial features (using concat)...")
    poly_features_cols = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_MEAN',
        'AGE_YEARS', 'YEARS_EMPLOYED',
        'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION'
    ]
    poly_features_cols = [col for col in poly_features_cols if col in X.columns] # Ensure columns exist

    if not poly_features_cols:
        print("Skipping polynomial features as no base columns found.")
    else:
        X_poly = X[poly_features_cols].copy()
        imputer = SimpleImputer(strategy='median')
        X_poly = imputer.fit_transform(X_poly)

        poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        poly_features_array = poly_transformer.fit_transform(X_poly)
        poly_feature_names = poly_transformer.get_feature_names_out(poly_features_cols)

        poly_features_df = pd.DataFrame(poly_features_array, columns=[f'POLY_{name}' for name in poly_feature_names], index=X.index)

        X = pd.concat([X, poly_features_df], axis=1)
        print(f"Shape after polynomial features: {X.shape}")
        del X_poly, poly_features_array, poly_features_df; gc.collect()

    # Engineer Simple Binary Features from Notebook
    if 'CNT_CHILDREN' in X.columns:
        X['HAS_CHILDREN'] = X['CNT_CHILDREN'].apply(lambda x: 1 if x > 0 else 0)
    if 'DAYS_EMPLOYED' in X.columns:
        X['DAYS_EMPLOYED'] = pd.to_numeric(X['DAYS_EMPLOYED'], errors='coerce')
        X['HAS_JOB'] = X['DAYS_EMPLOYED'].apply(lambda x: 1 if pd.notna(x) and x < 0 else 0)
        X['LONG_EMPLOYMENT'] = (X['DAYS_EMPLOYED'] < -2000).astype(int)
    if 'DAYS_BIRTH' in X.columns:
        X['DAYS_BIRTH'] = pd.to_numeric(X['DAYS_BIRTH'], errors='coerce')
        X['RETIREMENT_AGE'] = (X['DAYS_BIRTH'] < -14000).astype(int)

    # Engineer Numerical Features (Sums, Diffs) from Notebook
    # Sums
    if 'AMT_INCOME_TOTAL' in X.columns and 'AMT_ANNUITY' in X.columns:
        X['AMT_INCOME_TOTAL'] = pd.to_numeric(X['AMT_INCOME_TOTAL'], errors='coerce')
        X['AMT_ANNUITY'] = pd.to_numeric(X['AMT_ANNUITY'], errors='coerce')
        X['SUM_AMT_INCOME_TOTAL_AMT_ANNUITY'] = X['AMT_INCOME_TOTAL'] + X['AMT_ANNUITY']
    
    credit_bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    existing_bureau_cols = [col for col in credit_bureau_cols if col in X.columns]
    if existing_bureau_cols: # Check if at least one column exists
        for col in existing_bureau_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X['TOTAL_ENQUIRIES_CREDIT_BUREAU'] = X[existing_bureau_cols].sum(axis=1)
        
    ext_source_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    existing_ext_cols = [col for col in ext_source_cols if col in X.columns]
    if len(existing_ext_cols) >= 1: # Check if at least one ext source exists
        for col in existing_ext_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        X['EXT_SOURCES_SUM'] = X[existing_ext_cols].sum(axis=1)
        X['EXT_SOURCES_MEAN'] = X[existing_ext_cols].mean(axis=1)
        X['EXT_SOURCES_MAX'] = X[existing_ext_cols].max(axis=1)
        X['EXT_SOURCES_MIN'] = X[existing_ext_cols].min(axis=1)
        X['EXT_SOURCES_MEDIAN'] = X[existing_ext_cols].median(axis=1)
        X['EXT_SOURCES_STD'] = X[existing_ext_cols].std(axis=1)
        
        if 'EXT_SOURCE_1' in X.columns and 'EXT_SOURCE_2' in X.columns and 'EXT_SOURCE_3' in X.columns:
            X['EXT_SOURCES_WEIGHTED_SUM'] = X['EXT_SOURCE_3'] * 5 + X['EXT_SOURCE_1'] * 3 + X['EXT_SOURCE_2'] * 1
            X['EXT_SOURCES_WEIGHTED_AVG'] = (X['EXT_SOURCE_3'] * 5 + X['EXT_SOURCE_1'] * 3 + X['EXT_SOURCE_2'] * 1) / 3

    # Differences
    if 'AMT_CREDIT' in X.columns and 'AMT_GOODS_PRICE' in X.columns:
        X['AMT_CREDIT'] = pd.to_numeric(X['AMT_CREDIT'], errors='coerce')
        X['AMT_GOODS_PRICE'] = pd.to_numeric(X['AMT_GOODS_PRICE'], errors='coerce')
        X['DIFF_AMT_CREDIT_AMT_GOODS_PRICE'] = X['AMT_CREDIT'] - X['AMT_GOODS_PRICE']
    if 'AMT_ANNUITY' in X.columns and 'AMT_GOODS_PRICE' in X.columns:
        X['AMT_ANNUITY'] = pd.to_numeric(X['AMT_ANNUITY'], errors='coerce')
        X['DIFF_AMT_ANNUITY_AMT_GOODS_PRICE'] = X['AMT_ANNUITY'] - X['AMT_GOODS_PRICE']
    if 'AMT_INCOME_TOTAL' in X.columns and 'AMT_ANNUITY' in X.columns:
        X['DIFF_AMT_INCOME_TOTAL_AMT_ANNUITY'] = X['AMT_INCOME_TOTAL'] - X['AMT_ANNUITY']
    if 'CNT_FAM_MEMBERS' in X.columns and 'CNT_CHILDREN' in X.columns:
        X['CNT_FAM_MEMBERS'] = pd.to_numeric(X['CNT_FAM_MEMBERS'], errors='coerce')
        X['CNT_CHILDREN'] = pd.to_numeric(X['CNT_CHILDREN'], errors='coerce')
        X['CNT_ADULT_FAM_MEMBER'] = X['CNT_FAM_MEMBERS'] - X['CNT_CHILDREN']
    if 'OBS_30_CNT_SOCIAL_CIRCLE' in X.columns and 'OBS_60_CNT_SOCIAL_CIRCLE' in X.columns:
        X['OBS_30_CNT_SOCIAL_CIRCLE'] = pd.to_numeric(X['OBS_30_CNT_SOCIAL_CIRCLE'], errors='coerce')
        X['OBS_60_CNT_SOCIAL_CIRCLE'] = pd.to_numeric(X['OBS_60_CNT_SOCIAL_CIRCLE'], errors='coerce')
        X['DIFF_OBS_30_CNT_SOCIAL_CIRCLE_OBS_60_CNT_SOCIAL_CIRCLE'] = X['OBS_30_CNT_SOCIAL_CIRCLE'] - X['OBS_60_CNT_SOCIAL_CIRCLE']
    if 'DEF_30_CNT_SOCIAL_CIRCLE' in X.columns and 'DEF_60_CNT_SOCIAL_CIRCLE' in X.columns:
        X['DEF_30_CNT_SOCIAL_CIRCLE'] = pd.to_numeric(X['DEF_30_CNT_SOCIAL_CIRCLE'], errors='coerce')
        X['DEF_60_CNT_SOCIAL_CIRCLE'] = pd.to_numeric(X['DEF_60_CNT_SOCIAL_CIRCLE'], errors='coerce')
        X['DIFF_DEF_30_CNT_SOCIAL_CIRCLE_DEF_60_CNT_SOCIAL_CIRCLE'] = X['DEF_30_CNT_SOCIAL_CIRCLE'] - X['DEF_60_CNT_SOCIAL_CIRCLE']

    # Categorical features
    cat_feat = [
        'NAME_CONTRACT_TYPE', 
        'CODE_GENDER', 
        'FLAG_OWN_CAR', 
        'FLAG_OWN_REALTY', 
        'NAME_INCOME_TYPE', 
        'NAME_EDUCATION_TYPE', 
        'NAME_FAMILY_STATUS', 
        'NAME_HOUSING_TYPE', 
        'REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY', 
        'WEEKDAY_APPR_PROCESS_START', 
        'ORGANIZATION_TYPE',
        'NAME_TYPE_SUITE', 
        'OCCUPATION_TYPE', 
        'WALLSMATERIAL_MODE', 
        'FONDKAPREMONT_MODE',
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
        'OWN_CAR_AGE',
    ]

    # Make copies of each multi-categorical feature, and set aside 
    # those copies to be target mean encoded. 
    #
    # (Some of the original categorical features will later be 
    #  used as numerical features.)
    for feature in cat_feat:
        new_name = feature + '_CAT'
        orig_idx_of_feature = X.columns.get_loc(feature)
        X.insert(orig_idx_of_feature, new_name, X[feature])
        X[[new_name]] = X[[new_name]].apply(lambda x: x.astype('category'))

    # Set aside the categorical features that will also be used as 
    # numerical features.
    cat_feat_to_keep = [
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
        'OWN_CAR_AGE',
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
    ]


    # Drop the original categorical features that won't also be used as 
    # numerical features:
    cat_feat_to_drop = list(set(cat_feat) - set(cat_feat_to_keep))
    X.drop(cat_feat_to_drop, axis=1, inplace=True)
    
    
    # Encode categorical features (using one-hot encoding)
    print("Applying one-hot encoding using pd.get_dummies...")
    original_cols = X.columns
    X = pd.get_dummies(X, dummy_na=False, dtype=np.int8) # Use int8 for dummies
    cat_cols = [col for col in X.columns if col not in original_cols]
    print(f"Created {len(cat_cols)} new dummy columns (using int8).")

    # Replace infinite values with NaN after potential division by zero during feature engineering
    X = X.replace([np.inf, -np.inf], np.nan)

    print("Main application features engineered.")
    gc.collect()
    return X
