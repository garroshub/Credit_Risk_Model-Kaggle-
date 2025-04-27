# feature_engineering_credit_card.py
import pandas as pd
import numpy as np
import gc # Import garbage collector

def engineer_credit_card_features(X, credit_card_balance):
    """
    Engineers features from the credit_card_balance DataFrame, joining them to X.

    Args:
        X (pd.DataFrame): The main application DataFrame.
        credit_card_balance (pd.DataFrame): The credit_card_balance DataFrame.

    Returns:
        pd.DataFrame: DataFrame X with engineered credit card features added.
    """
    print("\n--- Engineering Credit Card Features (Notebook Alignment) ---")
    if credit_card_balance is None or credit_card_balance.empty:
        print("DEBUG: Input credit_card_balance DataFrame is empty or None. Skipping Credit Card features.")
        # Add placeholder columns if needed
        return X

    cc_to_agg = credit_card_balance.copy()

    # Engineer new features (from notebook)
    cc_to_agg['RATIO_AMT_BALANCE_TO_AMT_CREDIT_LIMIT_ACTUAL'] = cc_to_agg['AMT_BALANCE'] / cc_to_agg['AMT_CREDIT_LIMIT_ACTUAL']
    cc_to_agg['SUM_ALL_AMT_DRAWINGS'] = cc_to_agg[['AMT_DRAWINGS_ATM_CURRENT', 
                                                   'AMT_DRAWINGS_CURRENT', 
                                                   'AMT_DRAWINGS_OTHER_CURRENT', 
                                                   'AMT_DRAWINGS_POS_CURRENT']].sum(axis=1)
    cc_to_agg['RATIO_AMT_PAYMENT_TOTAL_CURRENT_TO_AMT_TOTAL_RECEIVABLE'] = cc_to_agg['AMT_PAYMENT_TOTAL_CURRENT'] / cc_to_agg['AMT_TOTAL_RECEIVABLE']
    cc_to_agg['RATIO_AMT_PAYMENT_CURRENT_TO_AMT_RECIVABLE'] = cc_to_agg['AMT_PAYMENT_CURRENT'] / cc_to_agg['AMT_RECIVABLE']
    cc_to_agg['SUM_ALL_CNT_DRAWINGS'] = cc_to_agg[['CNT_DRAWINGS_ATM_CURRENT', 
                                                   'CNT_DRAWINGS_CURRENT', 
                                                   'CNT_DRAWINGS_OTHER_CURRENT', 
                                                   'CNT_DRAWINGS_POS_CURRENT']].sum(axis=1)
    cc_to_agg['RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS'] = cc_to_agg['SUM_ALL_AMT_DRAWINGS'] / cc_to_agg['SUM_ALL_CNT_DRAWINGS']
    # Note: Notebook had division here, typo? Assuming difference based on name.
    # cc_to_agg['DIFF_AMT_TOTAL_RECEIVABLE_AMT_PAYMENT_TOTAL_CURRENT'] = cc_to_agg['AMT_TOTAL_RECEIVABLE'] / cc_to_agg['AMT_PAYMENT_TOTAL_CURRENT'] 
    cc_to_agg['DIFF_AMT_TOTAL_RECEIVABLE_AMT_PAYMENT_TOTAL_CURRENT'] = cc_to_agg['AMT_TOTAL_RECEIVABLE'] - cc_to_agg['AMT_PAYMENT_TOTAL_CURRENT'] 
    cc_to_agg['RATIO_AMT_PAYMENT_CURRENT_TO_AMT_PAYMENT_TOTAL_CURRENT'] = cc_to_agg['AMT_PAYMENT_CURRENT'] / cc_to_agg['AMT_PAYMENT_TOTAL_CURRENT']
    cc_to_agg['RATIO_AMT_RECEIVABLE_PRINCIPAL_TO_AMT_RECIVABLE'] = cc_to_agg['AMT_RECEIVABLE_PRINCIPAL'] / cc_to_agg['AMT_RECIVABLE']

    # Replace potential INF values after calculations
    cc_to_agg.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle NAME_CONTRACT_STATUS (from notebook)
    # Check if column exists and has non-NA values before converting
    if 'NAME_CONTRACT_STATUS' in cc_to_agg.columns and cc_to_agg['NAME_CONTRACT_STATUS'].notna().any():
        cc_to_agg['NAME_CONTRACT_STATUS'] = cc_to_agg['NAME_CONTRACT_STATUS'].astype('category')
        cc_to_agg['NAME_CONTRACT_STATUS_CODE'] = cc_to_agg['NAME_CONTRACT_STATUS'].cat.codes
    else:
        # Handle cases where the column might be missing or all NaN
        cc_to_agg['NAME_CONTRACT_STATUS_CODE'] = np.nan 

    # Aggregation dictionary (from notebook)
    aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum', 'var'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max'],
        'AMT_DRAWINGS_CURRENT': ['max'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['max'],
        'AMT_DRAWINGS_POS_CURRENT': ['max'],
        'AMT_INST_MIN_REGULARITY': ['max'],
        'AMT_PAYMENT_CURRENT': ['max'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max'],
        'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'sum'],
        'AMT_RECIVABLE': ['mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max'], 
        'CNT_DRAWINGS_CURRENT': ['max'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['max'],
        'CNT_DRAWINGS_POS_CURRENT': ['max'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean', 'sum'],
        'SK_DPD': ['max', 'sum'],
        'SK_DPD_DEF': ['max', 'sum'],
        'NAME_CONTRACT_STATUS_CODE': ['mean'], # Use the numeric code
        
        #Newly engineered feats
        'RATIO_AMT_BALANCE_TO_AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
        'SUM_ALL_AMT_DRAWINGS': ['min', 'max', 'mean'],
        'RATIO_AMT_PAYMENT_TOTAL_CURRENT_TO_AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean'],
        'RATIO_AMT_PAYMENT_CURRENT_TO_AMT_RECIVABLE': ['min', 'max', 'mean'],
        'SUM_ALL_CNT_DRAWINGS': ['min', 'max', 'mean'],
        'RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS': ['min', 'max', 'mean'],
        'DIFF_AMT_TOTAL_RECEIVABLE_AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean'],
        'RATIO_AMT_PAYMENT_CURRENT_TO_AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean'],
        'RATIO_AMT_RECEIVABLE_PRINCIPAL_TO_AMT_RECIVABLE': ['min', 'max', 'mean'],
    }
    
    # Aggregate features
    # Drop original categorical column before aggregation if it exists
    cols_to_agg = list(aggregations.keys())
    if 'NAME_CONTRACT_STATUS' in cc_to_agg.columns:
         cols_to_aggregate = ['SK_ID_CURR'] + cols_to_agg
    else:
         cols_to_aggregate = ['SK_ID_CURR'] + cols_to_agg

    # Ensure all columns in aggregations exist in cc_to_agg before grouping
    valid_cols_for_agg = [col for col in cols_to_agg if col in cc_to_agg.columns]
    final_aggregations = {k: v for k, v in aggregations.items() if k in valid_cols_for_agg}
    
    if not final_aggregations:
        print("DEBUG: No valid columns found for CC aggregation. Skipping aggregation.")
        # Handle count separately if needed
    else:
        cc_agg_df = cc_to_agg.groupby('SK_ID_CURR').agg(final_aggregations)
        # Update column naming (from notebook)
        cc_agg_df.columns = pd.Index([f'{e[0]}_{e[1].upper()}_(CREDIT_CARD)' for e in cc_agg_df.columns.tolist()])
        # Join to main dataframe
        X = X.join(cc_agg_df, on='SK_ID_CURR', how='left')
        print(f"DEBUG: Shape after joining CC aggregations: {X.shape}")
        del cc_agg_df
        gc.collect()

    # Loan count (from notebook)
    count_credit_card_loans_df = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']]
    # Check if SK_ID_PREV exists before attempting nunique
    if 'SK_ID_PREV' in count_credit_card_loans_df.columns:
        count_credit_card_loans_df = pd.DataFrame(data=count_credit_card_loans_df.groupby(['SK_ID_CURR'])['SK_ID_PREV'].nunique()).reset_index()
        count_credit_card_loans_df = count_credit_card_loans_df.rename(columns={'SK_ID_PREV': 'COUNT_CREDIT_CARD_LOANS_(CREDIT_CARD)'})
        # Join to main dataframe
        X = X.join(count_credit_card_loans_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
        print(f"DEBUG: Shape after joining CC loan count: {X.shape}")
        del count_credit_card_loans_df
        gc.collect()
    else:
        print("DEBUG: SK_ID_PREV not found in credit_card_balance, skipping CC loan count.")


    # Remove fillna(0) - rely on left joins
    # print(f"DEBUG: Filling NaNs in {cc_agg.shape[1]} CC_ columns before return.")
    # cc_agg = cc_agg.fillna(0)

    del cc_to_agg # Clean up the initial dataframe copy
    gc.collect()

    print("Credit card features engineered (Notebook alignment).")
    return X # Return the modified main DataFrame
