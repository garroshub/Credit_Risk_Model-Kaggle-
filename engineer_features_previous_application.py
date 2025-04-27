# feature_engineering_previous_app.py
import pandas as pd
import numpy as np 
import gc # Add import for garbage collection

from preprocessing import preprocess_previous_application # Import the preprocessing function
from utils import convert_dtypes # Import convert_dtypes from utils


def engineer_previous_application_features(previous_application):
    """
    Engineers features from the previous_application DataFrame.

    Args:
        previous_application (pd.DataFrame): The previous_application DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered previous application features, indexed by SK_ID_CURR.
    """
    # Preprocess previous application data
    previous_application = preprocess_previous_application(previous_application)
    # Convert dtypes (including object to category)
    prev_app_to_agg = convert_dtypes(previous_application, use_float16=False)

    # ----- START: Add Notebook Pre-calculated Features -----
    # Add a new numerical feature: the percentage of the amount of 
    # money applied for in the loan application, that was actually received.
    prev_app_to_agg['RATIO_AMT_APPLICATION_TO_AMT_CREDIT'] = (prev_app_to_agg['AMT_APPLICATION'] / prev_app_to_agg['AMT_CREDIT'])
    
    # New aggregations based on notebook
    prev_app_to_agg['RATIO_AMT_CREDIT_TO_AMT_ANNUITY'] = prev_app_to_agg['AMT_CREDIT'] / prev_app_to_agg['AMT_ANNUITY']
    prev_app_to_agg['RATIO_AMT_APPLICATION_TO_AMT_ANNUITY'] = prev_app_to_agg['AMT_APPLICATION'] / prev_app_to_agg['AMT_ANNUITY']
    prev_app_to_agg['DIFF_AMT_DOWN_PAYMENT_AMT_ANNUITY'] = prev_app_to_agg['AMT_DOWN_PAYMENT'] - prev_app_to_agg['AMT_ANNUITY']
    prev_app_to_agg['RATIO_AMT_CREDIT_TO_AMT_DOWN_PAYMENT'] = prev_app_to_agg['AMT_CREDIT'] / prev_app_to_agg['AMT_DOWN_PAYMENT']
    prev_app_to_agg['DIFF_AMT_CREDIT_AMT_GOODS_PRICE'] = prev_app_to_agg['AMT_CREDIT'] - prev_app_to_agg['AMT_GOODS_PRICE']
    prev_app_to_agg['DIFF_AMT_APPLICATION_AMT_GOODS_PRICE'] = prev_app_to_agg['AMT_APPLICATION'] - prev_app_to_agg['AMT_GOODS_PRICE']
    prev_app_to_agg['DIFF_RATE_DOWN_PAYMENT_RATE_INTEREST_PRIMARY'] = prev_app_to_agg['RATE_DOWN_PAYMENT'] - prev_app_to_agg['RATE_INTEREST_PRIMARY']
    prev_app_to_agg['DIFF_RATE_INTEREST_PRIVILEGED_RATE_INTEREST_PRIMARY'] = prev_app_to_agg['RATE_INTEREST_PRIVILEGED'] - prev_app_to_agg['RATE_INTEREST_PRIMARY']
    prev_app_to_agg['DIFF_DAYS_LAST_DUE_DAYS_FIRST_DUE'] = prev_app_to_agg['DAYS_LAST_DUE'] - prev_app_to_agg['DAYS_FIRST_DUE']
    prev_app_to_agg['DIFF_DAYS_TERMINATION_DAYS_DECISION'] = prev_app_to_agg['DAYS_TERMINATION'] - prev_app_to_agg['DAYS_DECISION']
    prev_app_to_agg['RATIO_DAYS_LAST_DUE_TO_DAYS_LAST_DUE_1ST_VERSION'] = prev_app_to_agg['DAYS_LAST_DUE'] / prev_app_to_agg['DAYS_LAST_DUE_1ST_VERSION']
    prev_app_to_agg['RATIO_DAYS_DECISION_TO_DAYS_TERMINATION'] = prev_app_to_agg['DAYS_DECISION'] / prev_app_to_agg['DAYS_TERMINATION']
    # ----- END: Add Notebook Pre-calculated Features -----

    # ----- Numerical Aggregations (Expanded based on Notebook) -----
    # Use prev_app_to_agg for calculations from now on
    PREVIOUS_APP_NUM_AGG = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'], 
        'AMT_CREDIT': ['min', 'max', 'mean'], 
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'], 
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'], 
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'], 
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'], 
        'RATE_INTEREST_PRIMARY': ['min', 'max', 'mean'],   
        'RATE_INTEREST_PRIVILEGED': ['min', 'max', 'mean'], 
        'DAYS_DECISION': ['min', 'max', 'mean'], 
        'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'], 
        'DAYS_FIRST_DUE': ['min', 'max', 'mean'], 
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'], 
        'DAYS_LAST_DUE': ['min', 'max', 'mean'], 
        'DAYS_TERMINATION': ['min', 'max', 'mean'], 
        'CNT_PAYMENT': ['mean', 'sum'],
        # Aggregations for pre-calculated features
        'RATIO_AMT_APPLICATION_TO_AMT_CREDIT': ['min', 'max', 'mean', 'var'],
        'RATIO_AMT_CREDIT_TO_AMT_ANNUITY': ['min', 'max', 'mean', 'var'],
        'RATIO_AMT_APPLICATION_TO_AMT_ANNUITY': ['min', 'max', 'mean', 'var'],
        'DIFF_AMT_DOWN_PAYMENT_AMT_ANNUITY': ['min', 'max', 'mean', 'var'],
        'RATIO_AMT_CREDIT_TO_AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'var'],
        'DIFF_AMT_CREDIT_AMT_GOODS_PRICE': ['min', 'max', 'mean', 'var'],
        'DIFF_AMT_APPLICATION_AMT_GOODS_PRICE': ['min', 'max', 'mean', 'var'],
        'DIFF_RATE_DOWN_PAYMENT_RATE_INTEREST_PRIMARY': ['min', 'max', 'mean', 'var'],
        'DIFF_RATE_INTEREST_PRIVILEGED_RATE_INTEREST_PRIMARY': ['min', 'max', 'mean', 'var'],
        'DIFF_DAYS_LAST_DUE_DAYS_FIRST_DUE': ['min', 'max', 'mean', 'var'],
        'DIFF_DAYS_TERMINATION_DAYS_DECISION': ['min', 'max', 'mean', 'var'],
        'RATIO_DAYS_LAST_DUE_TO_DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean', 'var'],
        'RATIO_DAYS_DECISION_TO_DAYS_TERMINATION': ['min', 'max', 'mean', 'var'],
    }

    # Aggregate numerical features
    previous_app_agg = prev_app_to_agg.groupby('SK_ID_CURR').agg(PREVIOUS_APP_NUM_AGG)
    previous_app_agg.columns = pd.Index([f'PREV_{e[0]}_{e[1].upper()}' for e in previous_app_agg.columns.tolist()])
    previous_app_agg = previous_app_agg.reset_index() # <<< RESET INDEX EARLY
    print(f"DEBUG: Shape after initial numeric agg & reset_index: {previous_app_agg.shape}")
    gc.collect()

    # --- Aggregation for Categorical Features (using dummies/codes)
    prev_app_cat_agg = prev_app_to_agg.select_dtypes('category').copy()
    prev_app_cat_agg['SK_ID_CURR'] = prev_app_to_agg['SK_ID_CURR']

    # Convert categories to codes for aggregation
    for col in prev_app_cat_agg.select_dtypes('category').columns:
        prev_app_cat_agg[col] = prev_app_cat_agg[col].cat.codes
        # Handle potential missing values introduced by coding if necessary
        # (Assuming -1 or similar might represent NaN/None)
        # prev_app_cat_agg[col] = prev_app_cat_agg[col].replace(-1, np.nan) 

    # Group by SK_ID_CURR and aggregate codes (using mean)
    prev_app_cat_agg = prev_app_cat_agg.groupby('SK_ID_CURR').agg(['mean'])
    prev_app_cat_agg.columns = pd.Index(['PREV_' + e[0] + "_CODE_MEAN" for e in prev_app_cat_agg.columns.tolist()])
    prev_app_cat_agg = prev_app_cat_agg.reset_index() # <<< RESET INDEX for merge

    # Merge numerical and new categorical aggregations
    previous_app_agg = pd.merge(previous_app_agg, prev_app_cat_agg, on='SK_ID_CURR', how='left') # <<< MERGE ON COLUMN
    print(f"DEBUG: Shape after merging category code means: {previous_app_agg.shape}")

    del prev_app_cat_agg
    gc.collect()

    # Aggregate approved applications separately (Uses updated PREVIOUS_APP_NUM_AGG)
    approved = prev_app_to_agg[prev_app_to_agg['NAME_CONTRACT_STATUS'] == 'Approved']
    approved_agg = approved.groupby('SK_ID_CURR').agg(PREVIOUS_APP_NUM_AGG)
    approved_agg.columns = pd.Index([f'PREV_{e[0]}_{e[1].upper()}_APPROVED' for e in approved_agg.columns.tolist()])
    approved_agg = approved_agg.reset_index() # <<< RESET INDEX for merge
    previous_app_agg = pd.merge(previous_app_agg, approved_agg, on='SK_ID_CURR', how='left') # <<< MERGE ON COLUMN
    print(f"DEBUG: Shape after merging approved agg: {previous_app_agg.shape}")
    del approved, approved_agg
    gc.collect()

    # Aggregate refused applications separately
    refused = prev_app_to_agg[prev_app_to_agg['NAME_CONTRACT_STATUS'] == 'Refused']
    refused_agg = refused.groupby('SK_ID_CURR').agg(PREVIOUS_APP_NUM_AGG)
    refused_agg.columns = pd.Index([f'PREV_{e[0]}_{e[1].upper()}_REFUSED' for e in refused_agg.columns.tolist()])
    refused_agg = refused_agg.reset_index() # <<< RESET INDEX for merge
    previous_app_agg = pd.merge(previous_app_agg, refused_agg, on='SK_ID_CURR', how='left') # <<< MERGE ON COLUMN
    print(f"DEBUG: Shape after merging refused agg: {previous_app_agg.shape}")

    del refused, refused_agg
    gc.collect()

    # ---------------------------------------
    # ----- START: Add Loan Counts (Notebook Method) -----
    # Count total previous loans
    count_prev_loans_df = prev_app_to_agg[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
    count_prev_loans_df = count_prev_loans_df.groupby(['SK_ID_CURR'], as_index=False).count()
    count_prev_loans_df = count_prev_loans_df.rename(index=str, columns = {'SK_ID_PREV': 'PREV_COUNT_TOTAL_PREV_APP_LOANS'})
    previous_app_agg = pd.merge(previous_app_agg, count_prev_loans_df, on='SK_ID_CURR', how='left') # <<< MERGE ON COLUMN
    del count_prev_loans_df
    gc.collect()

    # Count approved previous loans
    count_appr_prev_loans_df = prev_app_to_agg[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
    count_appr_prev_loans_df = count_appr_prev_loans_df[prev_app_to_agg['NAME_CONTRACT_STATUS'] == 'Approved']
    count_appr_prev_loans_df = count_appr_prev_loans_df.groupby(['SK_ID_CURR'], as_index=False).count()
    count_appr_prev_loans_df = count_appr_prev_loans_df.rename(index=str, columns = {'SK_ID_PREV': 'PREV_COUNT_APPROVED_PREV_APP_LOANS'})

    # Merge approved count
    print(f"DEBUG: Merging approved count. Left key 'SK_ID_CURR' dtype: {previous_app_agg['SK_ID_CURR'].dtype}, Right key 'SK_ID_CURR' dtype: {count_appr_prev_loans_df['SK_ID_CURR'].dtype}")

    # Ensure consistent key types before merging
    try:
        # No need to convert index anymore
        previous_app_agg['SK_ID_CURR'] = previous_app_agg['SK_ID_CURR'].astype(int)
        count_appr_prev_loans_df['SK_ID_CURR'] = count_appr_prev_loans_df['SK_ID_CURR'].astype(int)
        print("DEBUG: Coerced merge keys for approved count to int.")
    except Exception as e:
        print(f"WARN: Could not coerce merge keys for approved count to int: {e}")

    previous_app_agg = pd.merge(previous_app_agg, count_appr_prev_loans_df, on='SK_ID_CURR', how='left') # <<< MERGE ON COLUMN
    # No need to drop SK_ID_CURR from index merge anymore
    # if 'SK_ID_CURR_y' in previous_app_agg.columns: previous_app_agg.drop(columns=['SK_ID_CURR_y'], inplace=True)
    # if 'SK_ID_CURR_x' in previous_app_agg.columns: previous_app_agg.rename(columns={'SK_ID_CURR_x': 'SK_ID_CURR'}, inplace=True)

    del count_appr_prev_loans_df
    gc.collect()

    # Count refused previous loans
    refused_prev_loans_df = prev_app_to_agg[prev_app_to_agg['NAME_CONTRACT_STATUS'] == 'Refused'][['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
    refused_prev_loans_df = refused_prev_loans_df.groupby(['SK_ID_CURR'], as_index=False).count()
    refused_prev_loans_df = refused_prev_loans_df.rename(index=str, columns = {'SK_ID_PREV': 'PREV_COUNT_REFUSED_PREV_APP_LOANS'})
    previous_app_agg = pd.merge(previous_app_agg, refused_prev_loans_df, on='SK_ID_CURR', how='left') # <<< MERGE ON COLUMN
    del refused_prev_loans_df
    gc.collect()

    # Count canceled previous loans
    canceled_prev_loans_df = prev_app_to_agg[prev_app_to_agg['NAME_CONTRACT_STATUS'] == 'Canceled'][['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
    canceled_prev_loans_df = canceled_prev_loans_df.groupby(['SK_ID_CURR'], as_index=False).count()
    canceled_prev_loans_df = canceled_prev_loans_df.rename(index=str, columns = {'SK_ID_PREV': 'PREV_COUNT_CANCELED_PREV_APP_LOANS'})
    previous_app_agg = pd.merge(previous_app_agg, canceled_prev_loans_df, on='SK_ID_CURR', how='left') # <<< MERGE ON COLUMN
    del canceled_prev_loans_df
    gc.collect()
    # ----- END: Add Loan Counts (Notebook Method) -----
    # ---------------------------------------

    # Create categorical loan count features
    previous_app_agg['PREV_COUNT_TOTAL_PREV_APP_LOANS_CAT'] = pd.cut(previous_app_agg['PREV_COUNT_TOTAL_PREV_APP_LOANS'], bins = [0, 1, 3, 5, 10, 1000], labels = ['1', '2-3', '4-5', '6-10', '>10']).astype('object')
    previous_app_agg['PREV_COUNT_APPROVED_PREV_APP_LOANS_CAT'] = pd.cut(previous_app_agg['PREV_COUNT_APPROVED_PREV_APP_LOANS'], bins = [0, 1, 3, 5, 10, 1000], labels = ['1', '2-3', '4-5', '6-10', '>10']).astype('object')

    # Fill NaN values generated during aggregation & merges
    num_cols_before_nan_fill = len(previous_app_agg.columns)
    # Get numeric columns before filling NaN
    numeric_cols = previous_app_agg.select_dtypes(include=np.number).columns.tolist()
    previous_app_agg[numeric_cols] = previous_app_agg[numeric_cols].fillna(0) # Using 0 for numeric NaNs
    print(f"DEBUG: Filling NaNs in {len(numeric_cols)} aggregated PREV_ numeric columns before return.")

    # Check for and handle duplicate columns before returning
    duplicates = previous_app_agg.columns[previous_app_agg.columns.duplicated()].tolist()
    if duplicates:
       print(f"WARNING: Duplicate columns found in PREV features before returning: {duplicates}")
       previous_app_agg = previous_app_agg.loc[:,~previous_app_agg.columns.duplicated()] # Drop duplicates

    # Check if SK_ID_CURR is actually in columns AFTER reset_index
    if 'SK_ID_CURR' in previous_app_agg.columns:
        print("DEBUG: 'SK_ID_CURR' IS present as a column before return.")
        # Ensure SK_ID_CURR is int type for merging consistency
        try:
            previous_app_agg['SK_ID_CURR'] = previous_app_agg['SK_ID_CURR'].astype(int)
            print("DEBUG: Ensured 'SK_ID_CURR' column is int type.")
        except Exception as e:
             print(f"WARN: Could not ensure SK_ID_CURR is int: {e}")
    else:
        print("DEBUG: 'SK_ID_CURR' IS NOT present as a column before return. THIS IS UNEXPECTED.")
        print(f"DEBUG: Columns before return: {previous_app_agg.columns.tolist()}")


    return previous_app_agg
