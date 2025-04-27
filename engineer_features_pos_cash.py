# engineer_features_pos_cash.py
import pandas as pd
import numpy as np
import gc

# Placeholder for preprocessing if needed - Align with notebook (none needed here yet)
def preprocess_pos_cash(pos_cash):
    """Placeholder for potential preprocessing steps on POS CASH data."""
    # pos_cash.replace(SOME_VALUE, np.nan, inplace=True) # Example
    return pos_cash

def engineer_pos_cash_features(X, pos_cash_balance):
    """
    Engineers features from the POS_CASH_balance DataFrame based on notebook logic.

    Args:
        X (pd.DataFrame): The main application DataFrame to join features onto.
        pos_cash_balance (pd.DataFrame): The POS_CASH_balance DataFrame.

    Returns:
        pd.DataFrame: The main DataFrame `X` with engineered POS_CASH features joined.
    """
    print(f"DEBUG: Input pos_cash_balance shape: {pos_cash_balance.shape}")
    if pos_cash_balance.empty:
        print("DEBUG: Input pos_cash_balance DataFrame is empty. Skipping POS CASH features.")
        # Define expected columns based on full logic below to add if needed
        expected_cols = [
            # Aggregations
            'MONTHS_BALANCE_MAX_(POS_CASH)', 'MONTHS_BALANCE_MEAN_(POS_CASH)', 'MONTHS_BALANCE_SIZE_(POS_CASH)',
            'SK_DPD_MAX_(POS_CASH)', 'SK_DPD_MEAN_(POS_CASH)', 'SK_DPD_SUM_(POS_CASH)',
            'SK_DPD_DEF_MAX_(POS_CASH)', 'SK_DPD_DEF_MEAN_(POS_CASH)', 'SK_DPD_DEF_SUM_(POS_CASH)',
            'CNT_INSTALMENT_FUTURE_MEAN_(POS_CASH)', 'CNT_INSTALMENT_FUTURE_SUM_(POS_CASH)',
            'CNT_INSTALMENT_MAX_(POS_CASH)', 'CNT_INSTALMENT_MEAN_(POS_CASH)', # Added mean based on common practice
            'NAME_CONTRACT_STATUS_MEAN_(POS_CASH)', # Mean of codes
            'RATIO_SK_DPD_TO_CNT_INSTALMENT_FUTURE_MIN_(POS_CASH)', 'RATIO_SK_DPD_TO_CNT_INSTALMENT_FUTURE_MAX_(POS_CASH)', 'RATIO_SK_DPD_TO_CNT_INSTALMENT_FUTURE_MEAN_(POS_CASH)',
            'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_FUTURE_MIN_(POS_CASH)', 'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_FUTURE_MAX_(POS_CASH)', 'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_FUTURE_MEAN_(POS_CASH)',
            'RATIO_SK_DPD_TO_CNT_INSTALMENT_MIN_(POS_CASH)', 'RATIO_SK_DPD_TO_CNT_INSTALMENT_MAX_(POS_CASH)', 'RATIO_SK_DPD_TO_CNT_INSTALMENT_MEAN_(POS_CASH)',
            'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_MIN_(POS_CASH)', 'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_MAX_(POS_CASH)', 'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_MEAN_(POS_CASH)',
            'RATIO_CNT_INSTALMENT_TO_CNT_INSTALMENT_FUTURE_MIN_(POS_CASH)', 'RATIO_CNT_INSTALMENT_TO_CNT_INSTALMENT_FUTURE_MAX_(POS_CASH)', 'RATIO_CNT_INSTALMENT_TO_CNT_INSTALMENT_FUTURE_MEAN_(POS_CASH)',
            'DIFF_CNT_INSTALMENT_FUTURE_CNT_INSTALMENT_MIN_(POS_CASH)', 'DIFF_CNT_INSTALMENT_FUTURE_CNT_INSTALMENT_MAX_(POS_CASH)', 'DIFF_CNT_INSTALMENT_FUTURE_CNT_INSTALMENT_MEAN_(POS_CASH)',
            # Loan Counts
            'COUNT_POS_CASH_LOANS_(POS_CASH)', 'COUNT_POS_CASH_LOANS_CAT_(POS_CASH)',
            # Recent Status Counts
            'POS_CASH_COUNT_COMPLETED_LAST_MONTH', 'POS_CASH_COUNT_SIGNED_LAST_MONTH'
        ]
        for col in expected_cols:
            if col not in X.columns:
                 # Add column with NaN, will be filled later if needed
                 X[col] = np.nan
                 # Attempt to set dtype if possible (e.g., for category)
                 if '_CAT_' in col:
                     try:
                         X[col] = X[col].astype('category')
                     except Exception as e:
                         print(f"Warning: Could not set dtype for {col}: {e}")
        return X

    # Preprocessing (if any defined)
    pos_cash_balance = preprocess_pos_cash(pos_cash_balance.copy())

    # Copy for aggregation to preserve original values for later steps
    pos_cash_to_agg = pos_cash_balance.copy()

    # ----- Engineer New Numerical Features Before Aggregation -----
    pos_cash_to_agg['RATIO_SK_DPD_TO_CNT_INSTALMENT_FUTURE'] = pos_cash_to_agg['SK_DPD'] / pos_cash_to_agg['CNT_INSTALMENT_FUTURE']
    pos_cash_to_agg['RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_FUTURE'] = pos_cash_to_agg['SK_DPD_DEF'] / pos_cash_to_agg['CNT_INSTALMENT_FUTURE']
    pos_cash_to_agg['RATIO_SK_DPD_TO_CNT_INSTALMENT'] = pos_cash_to_agg['SK_DPD'] / pos_cash_to_agg['CNT_INSTALMENT']
    pos_cash_to_agg['RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT'] = pos_cash_to_agg['SK_DPD_DEF'] / pos_cash_to_agg['CNT_INSTALMENT']
    pos_cash_to_agg['RATIO_CNT_INSTALMENT_TO_CNT_INSTALMENT_FUTURE'] = pos_cash_to_agg['CNT_INSTALMENT'] / pos_cash_to_agg['CNT_INSTALMENT_FUTURE']
    pos_cash_to_agg['DIFF_CNT_INSTALMENT_FUTURE_CNT_INSTALMENT'] = pos_cash_to_agg['CNT_INSTALMENT_FUTURE'] - pos_cash_to_agg['CNT_INSTALMENT']

    # Replace infinite values that might result from division by zero
    pos_cash_to_agg.replace([np.inf, -np.inf], np.nan, inplace=True)


    # ----- Aggregation Dictionary -----
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'sum'],
        'CNT_INSTALMENT': ['max', 'mean'], # Added 'mean' based on common practice, notebook had 'max'
        'NAME_CONTRACT_STATUS': ['mean'], # Will aggregate the category codes

        # New features
        'RATIO_SK_DPD_TO_CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'RATIO_SK_DPD_TO_CNT_INSTALMENT': ['min', 'max', 'mean'],
        'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT': ['min', 'max', 'mean'],
        'RATIO_CNT_INSTALMENT_TO_CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'DIFF_CNT_INSTALMENT_FUTURE_CNT_INSTALMENT': ['min', 'max', 'mean'],
    }

    # Handle NAME_CONTRACT_STATUS: Convert to category codes for mean aggregation
    # Check if dtype is already category (due to optimization in main.py)
    if pos_cash_to_agg['NAME_CONTRACT_STATUS'].dtype.name != 'category':
        print("DEBUG: Converting NAME_CONTRACT_STATUS to category for POS CASH.")
        pos_cash_to_agg['NAME_CONTRACT_STATUS'] = pos_cash_to_agg['NAME_CONTRACT_STATUS'].astype('category')

    # Store original column reference before overwriting with codes
    name_contract_status_col = pos_cash_to_agg['NAME_CONTRACT_STATUS']
    pos_cash_to_agg['NAME_CONTRACT_STATUS'] = name_contract_status_col.cat.codes # Aggregate codes


    # ----- Perform Aggregation -----
    print("DEBUG: Performing POS CASH aggregation...")
    pos_cash_agg_df = pos_cash_to_agg.groupby('SK_ID_CURR').agg(aggregations)
    pos_cash_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() + '_(POS_CASH)' for e in pos_cash_agg_df.columns.tolist()])
    print(f"DEBUG: POS CASH aggregation complete. Shape: {pos_cash_agg_df.shape}")

    # Join aggregated features to main dataframe X
    print(f"DEBUG: Merging pos_cash_agg_df (shape {pos_cash_agg_df.shape}) onto X (shape {X.shape})")
    X = X.join(pos_cash_agg_df, how='left', on='SK_ID_CURR')
    print(f"DEBUG: X shape after pos_cash_agg merge: {X.shape}")
    # Clean up intermediate aggregation dataframe
    del pos_cash_to_agg, pos_cash_agg_df
    gc.collect()


    # ----- Calculate Loan Counts -----
    print("DEBUG: Calculating POS CASH loan counts...")
    # Use original pos_cash_balance for counts
    count_pos_cash_loans_df = pos_cash_balance[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
    count_pos_cash_loans_df = count_pos_cash_loans_df.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count()
    count_pos_cash_loans_df = count_pos_cash_loans_df.rename(columns={'SK_ID_PREV': 'COUNT_POS_CASH_LOANS_(POS_CASH)'})

    # Join count to main dataframe X
    print(f"DEBUG: Merging count_pos_cash_loans_df (shape {count_pos_cash_loans_df.shape}) onto X (shape {X.shape})")
    X = pd.merge(X, count_pos_cash_loans_df, on='SK_ID_CURR', how='left')
    # Fill NaN for borrowers with no POS loans and ensure integer type
    X['COUNT_POS_CASH_LOANS_(POS_CASH)'] = X['COUNT_POS_CASH_LOANS_(POS_CASH)'].fillna(0).astype(int)
    print(f"DEBUG: X shape after loan count merge: {X.shape}")
    del count_pos_cash_loans_df
    gc.collect()


    # ----- Calculate Categorical Loan Count -----
    print("DEBUG: Creating categorical POS loan count...")
    # 'COUNT_POS_CASH_LOANS_CAT_(POS_CASH)': Same as above, but as category type
    X['COUNT_POS_CASH_LOANS_CAT_(POS_CASH)'] = X['COUNT_POS_CASH_LOANS_(POS_CASH)'].astype('category')


    # ----- Calculate Recent Status Counts -----
    print("DEBUG: Calculating POS CASH recent status counts...")
    # Features indicating the count of loans with specific statuses in the most recent month per loan.
    recent_statuses = ['Completed', 'Signed'] # Only these are used in the notebook code provided

    # Get the most recent record for each loan (SK_ID_PREV)
    pos_cash_recent = pos_cash_balance.loc[pos_cash_balance.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()]

    print(f"DEBUG: pos_cash_recent shape: {pos_cash_recent.shape}")

    # One-hot encode the NAME_CONTRACT_STATUS for these recent records
    # Use the original pos_cash_recent which still has the categorical column
    if pos_cash_recent['NAME_CONTRACT_STATUS'].dtype.name != 'category':
         print("DEBUG: Converting NAME_CONTRACT_STATUS to category for recent POS records.")
         pos_cash_recent['NAME_CONTRACT_STATUS'] = pos_cash_recent['NAME_CONTRACT_STATUS'].astype('category')

    recent_status_dummies = pd.get_dummies(pos_cash_recent[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']],
                                           columns=['NAME_CONTRACT_STATUS'],
                                           prefix='POS_CASH_STATUS',
                                           dtype=np.int8)

    # Aggregate counts per SK_ID_CURR
    recent_status_counts = recent_status_dummies.groupby('SK_ID_CURR').sum()

    # Rename columns to match notebook (only for the selected statuses)
    cols_to_keep_rename = {}
    final_cols = []
    for status in recent_statuses:
        original_col_name = f'POS_CASH_STATUS_{status}'
        new_col_name = f'POS_CASH_COUNT_{status.upper()}_LAST_MONTH' # Use uppercase like notebook
        final_cols.append(new_col_name)
        if original_col_name in recent_status_counts.columns:
            cols_to_keep_rename[original_col_name] = new_col_name
        else:
            # If a status never appears, ensure the column exists in X and is filled with 0
            print(f"DEBUG: Status '{status}' not found in recent POS records, ensuring column '{new_col_name}' exists in X and is filled with 0.")
            if new_col_name not in X.columns:
                 X[new_col_name] = 0 # Create and initialize
            else:
                 X[new_col_name] = X[new_col_name].fillna(0) # Fill if already exists (e.g., from previous empty run)

    # Select and rename the columns that were found
    if cols_to_keep_rename: # Check if there are any columns to rename
        recent_status_counts = recent_status_counts[list(cols_to_keep_rename.keys())].rename(columns=cols_to_keep_rename)
        print(f"DEBUG: Renamed recent status counts shape: {recent_status_counts.shape}")

        # Join recent status counts to main dataframe X
        print(f"DEBUG: Merging recent_status_counts (shape {recent_status_counts.shape}) onto X (shape {X.shape})")
        X = X.join(recent_status_counts, how='left', on='SK_ID_CURR')
        print(f"DEBUG: X shape after recent status merge: {X.shape}")

        # Fill NaNs for borrowers who had no POS loans at all or none with these recent statuses
        # Only fill columns that were actually joined
        for col in recent_status_counts.columns:
            if col in X.columns: # Check if column exists before filling
                 X[col] = X[col].fillna(0).astype(int)
    else:
        print("DEBUG: No recent statuses ('Completed', 'Signed') found to aggregate.")

    # Clean up intermediate dataframes for recent status
    del pos_cash_recent, recent_status_dummies, recent_status_counts
    gc.collect()

    # Final check on NaNs for columns added by aggregation join
    # Only fill if the column definitely came from the aggregation step
    agg_cols_added = [col for col in X.columns if col.endswith('_(POS_CASH)')]
    print(f"DEBUG: Filling NaNs potentially introduced by POS CASH aggregation join for {len(agg_cols_added)} columns.")
    for col in agg_cols_added:
        if col in X.columns: # Ensure column exists
            # Fill with 0 for counts/sums/size, potentially NaN or 0 for means/mins/maxs depending on desired behavior
            # Using 0 as a general fill based on notebook's implicit behavior
            X[col] = X[col].fillna(0)

    print("POS CASH features engineered and merged.")
    print(f"DEBUG: Final X shape before returning from POS CASH: {X.shape}")
    # Return the updated main DataFrame
    return X
