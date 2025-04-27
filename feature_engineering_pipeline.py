# main.py
import pandas as pd
import numpy as np
import time
import gc # Garbage collector
import os

# Import preprocessing and feature engineering functions
from preprocessing import replace_XNA_XAP, preprocess_previous_application # Note: preprocess_previous_application is called within engineer_previous_app
from engineer_features_application import engineer_main_features, reduce_mem_usage 
from engineer_features_bureau import engineer_bureau_features
from engineer_features_previous_application import engineer_previous_application_features
from engineer_features_pos_cash import engineer_pos_cash_features
from engineer_features_installments import engineer_installments_features
from engineer_features_credit_card import engineer_credit_card_features

# Define data paths (Adjust paths as necessary)
DATA_DIR = 'd:/Apps/Fun Apps/Credit_Risk_Model/Data/' # Make sure this path is correct
OUTPUT_DIR = 'd:/Apps/Fun Apps/Credit_Risk_Model/Output/' # Optional: Define output directory

# Function to load data
def load_data(data_dir):
    print("Loading data...")
    app_train = pd.read_csv(data_dir + 'application_train.csv')
    app_test = pd.read_csv(data_dir + 'application_test.csv')
    bureau = pd.read_csv(data_dir + 'bureau.csv')
    bureau_balance = pd.read_csv(data_dir + 'bureau_balance.csv')
    previous_application = pd.read_csv(data_dir + 'previous_application.csv')
    pos_cash = pd.read_csv(data_dir + 'POS_CASH_balance.csv')
    installments_payments = pd.read_csv(data_dir + 'installments_payments.csv')
    credit_card_balance = pd.read_csv(data_dir + 'credit_card_balance.csv')

    # Define categorical columns for application data
    app_categorical_cols = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
        'NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'WALLSMATERIAL_MODE', 'FONDKAPREMONT_MODE'
    ]

    # Optimize app_train
    print("\nOptimizing app_train...")
    for col in app_categorical_cols:
        if col in app_train.columns:
            app_train[col] = app_train[col].astype('category')
    app_train = reduce_mem_usage(app_train)
    gc.collect()

    # Optimize app_test
    print("\nOptimizing app_test...")
    for col in app_categorical_cols:
        if col in app_test.columns:
            app_test[col] = app_test[col].astype('category')
    app_test = reduce_mem_usage(app_test)
    gc.collect()

    # Optimize bureau
    print("\nOptimizing bureau...")
    bureau_categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_TYPE', 'CREDIT_CURRENCY']
    for col in bureau_categorical_cols:
        if col in bureau.columns:
            bureau[col] = bureau[col].astype('category')
    bureau = reduce_mem_usage(bureau)
    gc.collect()

    # Optimize bureau_balance
    print("\nOptimizing bureau_balance...")
    if 'STATUS' in bureau_balance.columns:
        bureau_balance['STATUS'] = bureau_balance['STATUS'].astype('category')
    bureau_balance = reduce_mem_usage(bureau_balance)
    gc.collect()

    # Optimize previous_application
    print("\nOptimizing previous_application...")
    prev_app_categorical_cols = [
        'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT',
        'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
        'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
        'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
        'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION'
    ]
    for col in prev_app_categorical_cols:
        if col in previous_application.columns:
            if previous_application[col].dtype == 'object':
                previous_application[col] = previous_application[col].astype('category')
    previous_application = reduce_mem_usage(previous_application)
    gc.collect()

    # Optimize pos_cash
    print("\nOptimizing pos_cash...")
    if 'NAME_CONTRACT_STATUS' in pos_cash.columns:
        # Check if column actually exists and is object type before converting
        if pos_cash['NAME_CONTRACT_STATUS'].dtype == 'object':
            pos_cash['NAME_CONTRACT_STATUS'] = pos_cash['NAME_CONTRACT_STATUS'].astype('category')
    pos_cash = reduce_mem_usage(pos_cash)
    gc.collect()

    # Optimize installments_payments
    print("\nOptimizing installments_payments...")
    installments_payments = reduce_mem_usage(installments_payments)
    gc.collect()

    # Optimize credit_card_balance
    print("\nOptimizing credit_card_balance...")
    # TODO: Identify and convert credit_card_balance categorical columns if needed
    credit_card_balance = reduce_mem_usage(credit_card_balance)
    gc.collect()

    print("\nData loaded and optimized successfully.")
    return app_train, app_test, bureau, bureau_balance, previous_application, pos_cash, installments_payments, credit_card_balance

# Main processing function
def process_data(debug=False):
    num_rows = 10000 if debug else None # Limit rows for debugging if needed

    start_time = time.time()

    # Load data
    app_train, app_test, bureau, bureau_balance, previous_application, pos_cash, installments_payments, credit_card_balance = load_data(DATA_DIR)

    # Combine train and test for consistent processing
    df = pd.concat([app_train, app_test], ignore_index=True, sort=False)
    print(f"Combined train/test shape: {df.shape}")

    # Apply initial preprocessing (optional, if needed before feature engineering)
    # df = replace_XNA_XAP(df) # Example if needed on the combined df first

    # --- Feature Engineering ---
    df = engineer_main_features(df)
    gc.collect()
    print("Shape after main features:", df.shape)

    print("\n--- Engineering Bureau Features ---")
    # Corrected call: function returns aggregated features, merge them after
    bureau_features = engineer_bureau_features(bureau, bureau_balance)
    df = pd.merge(df, bureau_features, on='SK_ID_CURR', how='left')
    del bureau, bureau_balance, bureau_features
    gc.collect()
    print("Shape after bureau features:", df.shape)

    # Get features from previous application and merge them to df
    prev_app_features = engineer_previous_application_features(previous_application)
    df = pd.merge(df, prev_app_features, on='SK_ID_CURR', how='left')
    del previous_application, prev_app_features; gc.collect()
    print(f"Shape after previous app features: {df.shape}")

    df = engineer_pos_cash_features(df, pos_cash)
    del pos_cash; gc.collect()
    print(f"Shape after POS CASH features: {df.shape}")

    df = engineer_installments_features(df, installments_payments)
    del installments_payments; gc.collect()
    print(f"Shape after installments features: {df.shape}")

    df = engineer_credit_card_features(df, credit_card_balance)
    del credit_card_balance; gc.collect()
    print(f"Shape after credit card features: {df.shape}")

    # --- Final Memory Reduction ---
    print("\n--- Applying Final Memory Reduction ---")
    df = reduce_mem_usage(df)
    gc.collect()

    # Final processing (e.g., handling infinities, renaming columns if needed)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Consider filling remaining NaNs if appropriate before modeling
    # df.fillna(0, inplace=True) # Example

    # Rename columns to be compatible with LightGBM (removes special chars)
    df = df.rename(columns = lambda x: ''.join(c if c.isalnum() or c=='_' else '' for c in str(x)))

    # Separate train and test
    final_train_df = df[df['TARGET'].notna()]
    final_test_df = df[df['TARGET'].isna()]
    del df; gc.collect()

    print(f"Final Train shape: {final_train_df.shape}")
    print(f"Final Test shape: {final_test_df.shape}")

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    # --- Save Final DataFrames ---
    print(f"\nSaving dataframes to {OUTPUT_DIR}...")
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Save dataframes as parquet
    final_train_df.to_parquet(os.path.join(OUTPUT_DIR, 'train_processed.parquet'))
    final_test_df.to_parquet(os.path.join(OUTPUT_DIR, 'test_processed.parquet'))
    print("Dataframes saved successfully.")

    return final_train_df, final_test_df

# Run the process
if __name__ == "__main__":
    # Set debug=True for faster testing on a subset of data
    train_df, test_df = process_data(debug=False)
    # Further steps (like model training) would go here
