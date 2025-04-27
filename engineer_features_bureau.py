# feature_engineering_bureau.py
import pandas as pd
import numpy as np

from preprocessing import replace_XNA_XAP
from utils import one_hot_encoder, reduce_mem_usage, group, convert_dtypes

# Define aggregations - updated to include pre-calculated features
BUREAU_AGGREGATIONS = {
    'SK_ID_BUREAU': ('count',),
    # Original Features
    'DAYS_CREDIT': ('min', 'max', 'mean', 'var'),
    'DAYS_CREDIT_ENDDATE': ('min', 'max', 'mean'),
    'DAYS_CREDIT_UPDATE': ('mean',),
    'CREDIT_DAY_OVERDUE': ('max', 'mean'),
    'AMT_CREDIT_MAX_OVERDUE': ('max', 'mean'),
    'AMT_CREDIT_SUM': ('max', 'mean', 'sum'),
    'AMT_CREDIT_SUM_DEBT': ('max', 'mean', 'sum'),
    'AMT_CREDIT_SUM_LIMIT': ('mean', 'sum'),
    'AMT_CREDIT_SUM_OVERDUE': ('mean', 'sum'),
    'AMT_ANNUITY': ('max', 'mean'),
    'CNT_CREDIT_PROLONG': ('sum',),
    'MONTHS_BALANCE_SIZE': ('mean', 'sum'), # Aggregated from bureau_balance
    # Notebook Pre-calculated Features
    'DIFF_DAYS_CREDIT_ENDDATE_DAYS_CREDIT': ('mean', 'max'),
    'RATIO_AMT_CREDIT_SUM_TO_AMT_CREDIT_SUM_DEBT': ('mean', 'max', 'min'),
    'RATIO_AMT_CREDIT_SUM_TO_AMT_ANNUITY': ('mean', 'max', 'min'),
    'DIFF_AMT_CREDIT_SUM_DEBT_AMT_ANNUITY': ('mean', 'max', 'min'),
    'RATIO_AMT_CREDIT_SUM_DEBT_AMT_CREDIT_SUM': ('mean', 'max', 'min'),
}

# Define numerical aggregations for active/closed loans (re-use for ratios)
BUREAU_NUM_AGG = {
    'DAYS_CREDIT': ['min', 'max', 'mean'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
    'AMT_ANNUITY': ['max', 'mean'],
    'CNT_CREDIT_PROLONG': ['sum'],
    'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
    # Include notebook pre-calculated features here too
    'DIFF_DAYS_CREDIT_ENDDATE_DAYS_CREDIT': ['mean', 'max'],
    'RATIO_AMT_CREDIT_SUM_TO_AMT_CREDIT_SUM_DEBT': ['mean', 'max', 'min'],
    'RATIO_AMT_CREDIT_SUM_TO_AMT_ANNUITY': ['mean', 'max', 'min'],
    'DIFF_AMT_CREDIT_SUM_DEBT_AMT_ANNUITY': ['mean', 'max', 'min'],
    'RATIO_AMT_CREDIT_SUM_DEBT_AMT_CREDIT_SUM': ['mean', 'max', 'min'],
}

# Updated Bureau Balance aggregations for the two-step process
BB_NUM_AGG = {
    'MONTHS_BALANCE': ['min', 'max', 'size', 'mean', 'std', 'sum'],
    'DPD_COUNT': ['sum'] # Specific count of DPD statuses
}

BB_CAT_AGG = {
    'STATUS': ['nunique'] # Add other cat aggregations if needed
}

def _engineer_bureau_balance_features(bureau_balance):
    """Engineer features from bureau_balance data using two-step aggregation."""
    
    # Add DPD count feature based on STATUS
    # As per notebook: ['2','3','4','5'] are DPD statuses
    bureau_balance['DPD_COUNT'] = bureau_balance['STATUS'].apply(lambda x: 1 if x in ['2', '3', '4', '5'] else 0)
    
    # Categorical aggregations (by loan - SK_ID_BUREAU)
    bb_cat, bb_cat_cols = one_hot_encoder(bureau_balance, ['STATUS'], nan_as_category=False)
    # Convert column names to avoid issues later
    bb_cat.columns = ['BBalance_' + col for col in bb_cat.columns]
    bureau_balance = pd.concat([bureau_balance, bb_cat], axis=1)
    
    # Aggregate by SK_ID_BUREAU (Loan Level)
    aggregations = {}
    for agg in BB_NUM_AGG:
        aggregations[agg] = BB_NUM_AGG[agg]
    # Add categorical aggregations
    for agg in BB_CAT_AGG:
        aggregations[agg] = BB_CAT_AGG[agg]
    # Add one-hot encoded category means
    for col in bb_cat_cols:
         aggregations['BBalance_' + col] = ['mean']

    bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(aggregations)
    
    # Flatten column names
    bureau_balance_agg.columns = pd.Index([
        e[0] + "_BY_LOAN_" + e[1].upper() + '_(BB)' for e in bureau_balance_agg.columns.tolist()
    ])

    # Rename size column for clarity before merging
    bureau_balance_agg.rename(columns={'MONTHS_BALANCE_BY_LOAN_SIZE_(BB)': 'MONTHS_BALANCE_SIZE'}, inplace=True)

    return bureau_balance_agg

def engineer_bureau_features(bureau, bureau_balance): 
    """Engineer features for bureau and bureau_balance data.

    Args:
        bureau (pd.DataFrame): The bureau DataFrame.
        bureau_balance (pd.DataFrame): The bureau_balance DataFrame.

    Returns:
        pd.DataFrame: Aggregated bureau features indexed by SK_ID_CURR.
    """
    
    # Convert dtypes for memory efficiency
    bureau = convert_dtypes(bureau, use_float16=False)
    bureau_balance = convert_dtypes(bureau_balance, use_float16=False)
    
    # --- Pre-calculate features (from notebook) --- 
    # Convert relevant columns to numeric, coercing errors
    numeric_cols = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'AMT_CREDIT_SUM', 
                    'AMT_CREDIT_SUM_DEBT', 'AMT_ANNUITY']
    for col in numeric_cols:
        bureau[col] = pd.to_numeric(bureau[col], errors='coerce')

    # Differences
    bureau['DIFF_DAYS_CREDIT_ENDDATE_DAYS_CREDIT'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
    bureau['DIFF_AMT_CREDIT_SUM_DEBT_AMT_ANNUITY'] = bureau['AMT_CREDIT_SUM_DEBT'] - bureau['AMT_ANNUITY']

    # Ratios (handle division by zero)
    bureau['RATIO_AMT_CREDIT_SUM_TO_AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan)
    bureau['RATIO_AMT_CREDIT_SUM_TO_AMT_ANNUITY'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY'].replace(0, np.nan)
    bureau['RATIO_AMT_CREDIT_SUM_DEBT_AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM'].replace(0, np.nan)
    # --- End Pre-calculation --- 

    # Process bureau_balance (Loan Level Aggregation)
    bureau_balance_agg = _engineer_bureau_balance_features(bureau_balance)

    # Merge bureau_balance aggregations into bureau
    bureau = bureau.merge(bureau_balance_agg, how='left', on='SK_ID_BUREAU')
    # Fill NaNs created by the merge (if a bureau loan had no balance info)
    bb_agg_cols = bureau_balance_agg.columns
    bureau[bb_agg_cols] = bureau[bb_agg_cols].fillna(0)

    # --- Bureau Aggregations (Borrower Level) --- 
    
    # 1. Count Bureau Loans
    count_bureau_loans = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    bureau['COUNT_BUREAU_LOANS_(BUREAU)'] = bureau['SK_ID_CURR'].map(count_bureau_loans['SK_ID_BUREAU'])

    # 2. Main Numerical Aggregation (including pre-calculated features)
    bureau_agg = group(bureau, 'BUREAU', BUREAU_AGGREGATIONS)
    
    # 3. Active/Closed Loan Aggregations & Ratios
    active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    active_agg = group(active, 'BUREAU_ACTIVE', BUREAU_NUM_AGG)
    closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
    closed_agg = group(closed, 'BUREAU_CLOSED', BUREAU_NUM_AGG)
    
    bureau_agg = bureau_agg.merge(active_agg, how='left', on='SK_ID_CURR')
    bureau_agg = bureau_agg.merge(closed_agg, how='left', on='SK_ID_CURR')
    del active, active_agg, closed, closed_agg
    
    # Calculate ratios of active/closed aggregates
    for num_col in BUREAU_NUM_AGG:
        for stat in BUREAU_NUM_AGG[num_col]:
            active_col = f'BUREAU_ACTIVE_{num_col}_{stat.upper()}'
            closed_col = f'BUREAU_CLOSED_{num_col}_{stat.upper()}'
            if active_col in bureau_agg.columns and closed_col in bureau_agg.columns:
                bureau_agg[f'RATIO_{active_col}_TO_{closed_col}'] = bureau_agg[active_col] / bureau_agg[closed_col].replace(0, np.nan)

    # 4. Categorical Aggregations (Mean of Codes)
    categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
    for col in categorical_cols:
        if col in bureau.columns:
            bureau[col] = bureau[col].astype('category')
            bureau[f'{col}_CODE'] = bureau[col].cat.codes
            cat_agg_dict = {f'{col}_CODE': ['mean']}
            cat_agg = group(bureau, f'BUREAU_CAT_{col}', cat_agg_dict)
            bureau_agg = bureau_agg.merge(cat_agg, how='left', on='SK_ID_CURR')
            del cat_agg
            
    # 5. Latest Categorical Features
    latest_loan_idx = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].idxmax()
    latest_loan_info = bureau.loc[latest_loan_idx, ['SK_ID_CURR'] + categorical_cols].set_index('SK_ID_CURR')
    for col in categorical_cols:
         if col in latest_loan_info.columns:
            bureau_agg[f'LATEST_{col}_CAT_(BUREAU)'] = latest_loan_info[col]
    del latest_loan_idx, latest_loan_info
            
    # 6. Overdue Features
    bureau['IS_CREDIT_BUREAU_LOANS_OVERDUE'] = (bureau['CREDIT_DAY_OVERDUE'] > 0).astype(int)
    overdue_agg = bureau.groupby('SK_ID_CURR')['IS_CREDIT_BUREAU_LOANS_OVERDUE'].sum().reset_index()
    overdue_agg.rename(columns={'IS_CREDIT_BUREAU_LOANS_OVERDUE': 'COUNT_CREDIT_BUREAU_LOANS_OVERDUE_(BUREAU)'}, inplace=True)
    bureau_agg = bureau_agg.merge(overdue_agg, on='SK_ID_CURR', how='left')
    bureau_agg['HAS_CREDIT_BUREAU_LOANS_OVERDUE_(BUREAU)'] = (bureau_agg['COUNT_CREDIT_BUREAU_LOANS_OVERDUE_(BUREAU)'] > 0).astype(int)
    del overdue_agg

    # 7. One-Hot Encoding of Categorical Features (Mean)
    # Using modified one_hot_encoder which groups and takes mean/sum
    bureau_cat_ohe_agg, bureau_cat_ohe_cols = one_hot_encoder(bureau, 
                                                              ['CREDIT_ACTIVE', 'CREDIT_TYPE'], 
                                                              nan_as_category=False, 
                                                              group_by_sk_id=True, 
                                                              agg_method='mean') # Notebook used sum and clipped, let's use mean for now
    # Adjust column names to match notebook pattern (even though agg method differs slightly)
    bureau_cat_ohe_agg.columns = ['CAT_OHE_MEAN_' + col + '_(BUREAU)' for col in bureau_cat_ohe_cols]
    bureau_agg = bureau_agg.merge(bureau_cat_ohe_agg, how='left', on='SK_ID_CURR')
    del bureau_cat_ohe_agg

    # Add quantile features
    print("Calculating quantile features for bureau data...")
    quantile_features = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': lambda x: np.quantile(x, 0.25),
        'AMT_CREDIT_SUM': lambda x: np.quantile(x, 0.25),
        'AMT_CREDIT_SUM_DEBT': lambda x: np.quantile(x, 0.75)
    })
    
    quantile_features.columns = ['BUREAU_' + col + '_QUANTILE_(BUREAU)' for col in quantile_features.columns]
    bureau_agg = bureau_agg.merge(quantile_features, how='left', on='SK_ID_CURR')
    del quantile_features
    
    # Reduce memory
    bureau_agg = reduce_mem_usage(bureau_agg)

    print(f"Bureau features engineered: {bureau_agg.shape}")
    # Return the aggregated dataframe directly, ensuring SK_ID_CURR is index
    if 'SK_ID_CURR' in bureau_agg.columns:
         bureau_agg = bureau_agg.set_index('SK_ID_CURR')

    return bureau_agg

# --- Keep original function for reference or potential rollback ---
# def engineer_bureau_features_original(X, bureau, bureau_balance):
def engineer_bureau_features_original(X, bureau, bureau_balance):
    """
    Engineers features from the bureau and bureau_balance DataFrames.

    Args:
        X (pd.DataFrame): The main DataFrame containing SK_ID_CURR.
        bureau (pd.DataFrame): The bureau DataFrame.
        bureau_balance (pd.DataFrame): The bureau_balance DataFrame.

    Returns:
        pd.DataFrame: DataFrame X merged with engineered bureau features.
    """
    # Preprocess bureau data
    bureau = replace_XNA_XAP(bureau)
    
    # ----- Bureau Aggregations -----
    # Basic counts
    count_bureau_loans_df = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby(['SK_ID_CURR'], as_index=False).count()
    count_bureau_loans_df.rename(columns={'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'}, inplace=True)
    X = pd.merge(X, count_bureau_loans_df, on=['SK_ID_CURR'], how='left')
    X['BUREAU_LOAN_COUNT'] = X['BUREAU_LOAN_COUNT'].fillna(0)

    # Count loan types
    loan_types_df = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(['SK_ID_CURR'])['CREDIT_TYPE'].nunique()
    loan_types_df = loan_types_df.reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    X = pd.merge(X, loan_types_df, on=['SK_ID_CURR'], how='left')
    X['BUREAU_LOAN_TYPES'] = X['BUREAU_LOAN_TYPES'].fillna(0)
    
    # Average loans per type
    X['BUREAU_AVERAGE_LOAN_TYPE'] = X['BUREAU_LOAN_COUNT'] / (X['BUREAU_LOAN_TYPES'] + 1e-8) # Avoid division by zero

    # Debt-Credit ratio
    bureau['BUREAU_DEBT_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM'] + 1e-8)
    bureau['BUREAU_OVERDUE_DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / (bureau['AMT_CREDIT_SUM_DEBT'] + 1e-8)

    # ----- Bureau Balance Aggregations -----
    bureau_balance_agg = preprocess_bureau_balance(bureau_balance)
    bureau = pd.merge(bureau, bureau_balance_agg, left_on='SK_ID_BUREAU', right_index=True, how='left')
    
    # Clean up potentially very large values from merge artifacts if any
    numeric_cols_bb = [col for col in bureau.columns if col.startswith('bureau_balance_')]
    for col in numeric_cols_bb:
        bureau[col] = bureau[col].replace([np.inf, -np.inf], np.nan) # Replace infs if any

    # ----- Aggregate Bureau Features per SK_ID_CURR -----
    # Define aggregations
    BUREAU_AGGREGATIONS = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'BUREAU_DEBT_CREDIT_RATIO': ['mean', 'max'],
        'BUREAU_OVERDUE_DEBT_RATIO': ['mean', 'max']
    }
    
    # Add bureau balance aggregations
    for col in numeric_cols_bb:
         BUREAU_AGGREGATIONS[col] = ['mean', 'sum', 'min', 'max'] # Aggregate balance features

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(BUREAU_AGGREGATIONS)
    bureau_agg.columns = pd.Index([f'BUREAU_{e[0]}_{e[1].upper()}' for e in bureau_agg.columns.tolist()])
    
    # Aggregate active loans separately
    active_bureau_agg = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').agg(BUREAU_AGGREGATIONS)
    active_bureau_agg.columns = pd.Index([f'BUREAU_ACTIVE_{e[0]}_{e[1].upper()}' for e in active_bureau_agg.columns.tolist()])
    bureau_agg = pd.merge(bureau_agg, active_bureau_agg, left_index=True, right_index=True, how='left')

    # Aggregate closed loans separately
    closed_bureau_agg = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').agg(BUREAU_AGGREGATIONS)
    closed_bureau_agg.columns = pd.Index([f'BUREAU_CLOSED_{e[0]}_{e[1].upper()}' for e in closed_bureau_agg.columns.tolist()])
    bureau_agg = pd.merge(bureau_agg, closed_bureau_agg, left_index=True, right_index=True, how='left')

    # ----- Categorical Features -----
    # One-hot encode CREDIT_ACTIVE and CREDIT_TYPE
    bureau_cat_features = pd.get_dummies(bureau.select_dtypes('category'), prefix_sep='_', dtype=np.int8)
    bureau_cat_features['SK_ID_CURR'] = bureau['SK_ID_CURR']
    bureau_cat_agg = bureau_cat_features.groupby('SK_ID_CURR').agg(['mean', 'sum']) # Use mean for ratios, sum for counts
    
    new_cat_colnames = []
    for var in bureau_cat_agg.columns.levels[0]:
         for stat in bureau_cat_agg.columns.levels[1]:
              new_cat_colnames.append(f'BUREAU_{var}_{stat.upper()}')
    bureau_cat_agg.columns = new_cat_colnames

    bureau_agg = pd.merge(bureau_agg, bureau_cat_agg, left_index=True, right_index=True, how='left')
    
    # Merge bureau aggregations back to X
    X = pd.merge(X, bureau_agg, left_on='SK_ID_CURR', right_index=True, how='left')

    # Fill NaNs potentially created during merges or aggregations
    # Identify bureau columns added to X
    bureau_cols_in_X = [col for col in X.columns if col.startswith('BUREAU_')]
    X[bureau_cols_in_X] = X[bureau_cols_in_X].fillna(0) # Example: fill with 0, adjust if needed

    print("Bureau features engineered.")
    return X

def preprocess_bureau_balance(bureau_balance):
    """Preprocesses the bureau_balance DataFrame."""
    bureau_balance = replace_XNA_XAP(bureau_balance) # Assuming replace_XNA_XAP handles potential XNA/XAP here if needed

    # Define aggregations using named aggregation syntax
    aggregations = {
        'bureau_balance_MONTHS_BALANCE_min': ('MONTHS_BALANCE', 'min'),
        'bureau_balance_MONTHS_BALANCE_max': ('MONTHS_BALANCE', 'max'),
        'bureau_balance_MONTHS_BALANCE_size': ('MONTHS_BALANCE', 'size'),
    }

    # Add status aggregations dynamically
    status_values = ['0', '1', '2', '3', '4', '5', 'C', 'X']
    for status_val in status_values:
        aggregations[f'bureau_balance_STATUS_{status_val}_count'] = ('STATUS', lambda x, s=status_val: (x == s).sum())
        aggregations[f'bureau_balance_STATUS_{status_val}_ratio'] = ('STATUS', lambda x, s=status_val: ((x == s).sum()) / x.size if x.size > 0 else 0)

    # Perform aggregation
    bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(**aggregations) # Use ** to unpack the dict

    # Column names are already set correctly by named aggregation
    
    return bureau_balance_agg
