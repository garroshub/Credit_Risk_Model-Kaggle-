"""
train_model.py - Credit Risk Model Training

This script implements the LightGBM model training approach from the notebook
'kernel_home_credit_putting_all_the_steps_together_v10.ipynb'.

It reads the processed data files created by main.py and trains a LightGBM model
using cross-validation and target mean encoding.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import os
import csv
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Define directories
OUTPUT_DIR = 'd:/Apps/Fun Apps/Credit_Risk_Model/Output/'
MODEL_DIR = 'd:/Apps/Fun Apps/Credit_Risk_Model/Models/'

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Target Mean Encoding function from the notebook
def target_encode(train_series, test_series, target, min_samples_leaf=1, smoothing=1, noise_level=0):
    """
    Target mean encoding implementation.
    
    Args:
        train_series: Series with training data
        test_series: Series with test data
        target: Series with target values for training data
        min_samples_leaf: Minimum samples to take category average into account
        smoothing: Smoothing effect to balance category average vs prior
        noise_level: Noise level to add for regularization
        
    Returns:
        train_encoded, test_encoded: Series with encoded values
    """
    # Calculate global mean
    mean = target.mean()
    
    # Convert categorical series to string to avoid category dtype issues
    if train_series.dtype.name == 'category':
        train_series = train_series.astype(str)
    if test_series.dtype.name == 'category':
        test_series = test_series.astype(str)
    
    # Create a temporary dataframe for aggregation
    temp_df = pd.DataFrame({'category': train_series, 'target': target})
    
    # Group by category and calculate mean and count
    agg_df = temp_df.groupby('category').agg({'target': ['mean', 'count']})
    
    # Extract means and counts
    averages = agg_df[('target', 'mean')]
    counts = agg_df[('target', 'count')]
    
    # Calculate smoothed means
    smoothed_means = (counts * averages + smoothing * mean) / (counts + smoothing)
    
    # Apply means to the train series
    train_encoded = train_series.map(smoothed_means)
    
    # Add noise if specified
    if noise_level > 0:
        train_encoded = train_encoded * (1 + noise_level * np.random.randn(len(train_encoded)))
    
    # Apply means to the test series
    test_encoded = test_series.map(smoothed_means)
    
    # Handle previously unseen values
    train_encoded.fillna(mean, inplace=True)
    test_encoded.fillna(mean, inplace=True)
    
    # Return the encoded series
    return train_encoded, test_encoded

# LightGBM training function
def train_lgbm(params, X_train, y_train, X_val=None, y_val=None, for_cv=True, 
               max_boost_rounds=500, early_stop_rounds=50):
    """
    Train a LightGBM model.
    
    Args:
        params: LightGBM parameters
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        for_cv: Whether this is for cross-validation
        max_boost_rounds: Maximum number of boosting rounds
        early_stop_rounds: Early stopping rounds
        
    Returns:
        Trained LightGBM model
    """
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Set up callbacks for early stopping and verbose evaluation
    callbacks = []
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val)
        # Create early stopping callback
        callbacks.append(lgb.callback.early_stopping(stopping_rounds=early_stop_rounds, verbose=True))
        # Create log evaluation callback
        callbacks.append(lgb.callback.log_evaluation(period=50))
        
        # Train with validation data and callbacks
        model = lgb.train(
            params,
            train_data,
            num_boost_round=max_boost_rounds,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
    else:
        # If no validation data, train for fixed number of rounds with just logging
        callbacks.append(lgb.callback.log_evaluation(period=50))
        model = lgb.train(
            params,
            train_data,
            num_boost_round=max_boost_rounds,
            callbacks=callbacks
        )
    
    return model

# Function to display feature importances
def display_feature_importances(feat_importance_df, for_cv=False, top_n=50):
    """
    Display feature importances.
    
    Args:
        feat_importance_df: DataFrame with feature importances
        for_cv: Whether this is for cross-validation
        top_n: Number of top features to display
    """
    # Sort by importance
    sorted_importance = dict(sorted(feat_importance_df.items(), key=lambda x: x[1], reverse=True))
    
    # Print top N features
    print(f"\nTop {top_n} Feature Importances:")
    for i, (feature, importance) in enumerate(list(sorted_importance.items())[:top_n], 1):
        print(f"{i:3d}. {feature:50s}: {importance:.6f}")
    
    # Save feature importances to file
    if not for_cv:
        with open(os.path.join('Models', 'feature_importances.txt'), 'w') as f:
            for feature, importance in sorted_importance.items():
                f.write(f"{feature}: {importance:.6f}\n")

# Cross-validation function
def cross_validate(lgbm_params, X_train, y_train, n_folds=5, max_boost_rounds=500, 
                   early_stop_rounds=50, tme_min_samples_leaf=1, tme_smoothing=1, 
                   tme_noise_level=0, cv_random_seed=42):
    """
    Perform cross-validation with LightGBM.
    
    Args:
        lgbm_params: LightGBM parameters
        X_train: Training features
        y_train: Training target
        n_folds: Number of folds
        max_boost_rounds: Maximum number of boosting rounds
        early_stop_rounds: Early stopping rounds
        tme_*: Target mean encoding parameters
        cv_random_seed: Random seed for cross-validation
    """
    # Handle problematic columns before splitting
    # Convert object dtypes to numeric or categorical as appropriate
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            print(f"Converting object column to category: {col}")
            X_train[col] = X_train[col].astype('category')
    # Stratified K-fold cross val
    cv = StratifiedKFold(n_splits=n_folds, random_state=cv_random_seed, shuffle=True)

    # Keep track of best validation score and best round in each fold
    cv_scores = []
    best_rounds = []
    fold_count = 0
    
    # Store feature importances
    feat_importance_df = pd.DataFrame()

    for train_index, val_index in cv.split(X_train, y_train):
        fold_count += 1
        print(f"\n--- Fold {fold_count} of {n_folds} ---")
        
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Conduct target mean encoding for categorical features
        cat_feats = X_train_fold.select_dtypes(['category']).columns.tolist()
        print(f"Found {len(cat_feats)} categorical features")
        
        # Create copies to avoid modifying the original dataframes
        X_train_fold_processed = X_train_fold.copy()
        X_val_fold_processed = X_val_fold.copy()
        
        for feature in cat_feats:
            try:
                # Get target mean encoded series
                encoded_train, encoded_val = target_encode(
                    train_series=X_train_fold[feature], 
                    test_series=X_val_fold[feature], 
                    target=y_train_fold, 
                    min_samples_leaf=tme_min_samples_leaf,
                    smoothing=tme_smoothing,
                    noise_level=tme_noise_level
                )

                # Add encoded features with new names to avoid conflicts
                encoded_feature_name = f"{feature}_ENCODED"
                X_train_fold_processed[encoded_feature_name] = encoded_train
                X_val_fold_processed[encoded_feature_name] = encoded_val
                
                # Drop the original categorical feature
                X_train_fold_processed = X_train_fold_processed.drop(feature, axis=1)
                X_val_fold_processed = X_val_fold_processed.drop(feature, axis=1)
                
                print(f"Encoded feature: {feature}")
            except Exception as e:
                print(f"Error encoding feature {feature}: {e}")
                # If encoding fails, just drop the feature
                if feature in X_train_fold_processed.columns:
                    X_train_fold_processed = X_train_fold_processed.drop(feature, axis=1)
                if feature in X_val_fold_processed.columns:
                    X_val_fold_processed = X_val_fold_processed.drop(feature, axis=1)
        
        # Update references
        X_train_fold = X_train_fold_processed
        X_val_fold = X_val_fold_processed
        
        # Build and train the LightGBM classifier
        clf_lgb = train_lgbm(
            lgbm_params, 
            X_train_fold, 
            y_train_fold, 
            X_val_fold, 
            y_val_fold, 
            for_cv=True, 
            max_boost_rounds=max_boost_rounds, 
            early_stop_rounds=early_stop_rounds
        )
        
        # Save feature importances
        fold_feat_importance_df = pd.DataFrame()
        fold_feat_importance_df['Feature Name'] = clf_lgb.feature_name()
        fold_feat_importance_df['Importance Value'] = clf_lgb.feature_importance(importance_type='split', iteration=-1)
        fold_feat_importance_df['Fold'] = fold_count
        feat_importance_df = pd.concat([feat_importance_df, fold_feat_importance_df], axis=0)
        
        # Get best scores and rounds
        best_round = clf_lgb.best_iteration
        best_val_roc_auc_score = clf_lgb.best_score['val']['auc']
        best_train_roc_auc_score = clf_lgb.best_score['train']['auc']
        
        # Track scores and rounds
        cv_scores.append(best_val_roc_auc_score)
        best_rounds.append(best_round)
        
        print(f"Fold: {fold_count}  Validation AUC: {best_val_roc_auc_score:.6f}  Train AUC: {best_train_roc_auc_score:.6f}  Diff Train-Val: {best_train_roc_auc_score - best_val_roc_auc_score:.4f}  Best Round: {best_round}")
    
    print(f"\nAverage CV Score across {n_folds} folds: {np.mean(cv_scores):.6f}    Average Best Round: {np.mean(best_rounds):.1f}")
    
    # Calculate mean feature importance across folds
    mean_feat_importance = feat_importance_df.groupby('Feature Name')['Importance Value'].mean().to_dict()
    
    # Display feature importances
    display_feature_importances(mean_feat_importance, for_cv=True)
    
    return cv_scores, best_rounds, mean_feat_importance

# Function to blend predictions using mean ranking
def mean_rank_predictions(y_scores):
    """
    Blend predictions using mean ranking.
    
    Args:
        y_scores: DataFrame with multiple sets of predictions
        
    Returns:
        Blended predictions
    """
    # Rank each set of predictions
    pred_ranks = y_scores.rank(axis=0, method='average')
    
    # Calculate mean of ranks
    mean_ranks = pred_ranks.mean(axis=1)
    
    # Scale to [0,1]
    mean_ranks_scaler = MinMaxScaler()
    mean_ranked_predictions = mean_ranks_scaler.fit_transform(mean_ranks.values.reshape(-1,1))[:,0]
    
    return mean_ranked_predictions

# Function to write predictions to CSV
def write_pred_to_csv(borrower_IDs, predictions, predictions_version):
    """
    Write predictions to CSV file.
    
    Args:
        borrower_IDs: Series with borrower IDs
        predictions: Series with predictions
        predictions_version: Version string for the filename
    """
    # Create the CSV file
    file_output = f'kaggle_home_credit_submission_{predictions_version}.csv'
    output_path = os.path.join(OUTPUT_DIR, file_output)
    
    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['SK_ID_CURR','TARGET'])
        # Write predictions
        for index, value in enumerate(borrower_IDs):
            writer.writerow([value, predictions[index]])
    
    print(f"Predictions saved to {output_path}")

def main():
    """Main function to run the model training pipeline."""
    print("Starting Credit Risk Model Training...")
    start_time = time.time()
    
    # Check if processed data exists
    train_path = os.path.join(OUTPUT_DIR, 'train_processed.parquet')
    test_path = os.path.join(OUTPUT_DIR, 'test_processed.parquet')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Processed data not found. Please run feature_engineering_pipeline.py first to generate the processed data.")
        return
    
    # Load processed data
    print("Loading processed data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Separate features and target
    X_train = train_df.drop('TARGET', axis=1)
    y_train = train_df['TARGET']
    X_test = test_df.drop('TARGET', axis=1)
    test_ids = test_df['SK_ID_CURR'].values
    
    # Convert object dtypes to appropriate types for LightGBM
    # First, identify columns with object dtype
    object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"Found {len(object_cols)} columns with object dtype")
    
    # Convert object columns to categorical
    for col in object_cols:
        X_train[col] = X_train[col].astype('category')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')
    
    # LightGBM parameters (optimized)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'learning_rate': 0.03,         # Lower learning rate for better stability
        'num_leaves': 31,              # Fewer leaves to reduce overfitting
        'max_depth': 6,                # Slightly reduced tree depth
        'min_data_in_leaf': 100,
        'max_bin': 100,
        'bagging_fraction': 0.7,       # Reduced sample ratio
        'bagging_freq': 2,             # Increased bagging frequency
        'feature_fraction': 0.7,       # Reduced feature ratio
        'lambda_l1': 0.7,              # Increased L1 regularization
        'lambda_l2': 0.7,              # Increased L2 regularization
        'min_gain_to_split': 0.00,
        'min_child_weight': 10,        # Controls minimum sample weight in leaf nodes
        'cat_smooth': 20,              # Smoothing parameter for categorical features
        'cat_l2': 10,                  # L2 regularization for categorical features
        'verbose': -1,
        'seed': 42
    }
    
    # Cross-validation parameters
    N_FOLDS = 8                        # Increased folds for better stability
    MAX_BOOST_ROUNDS = 10000
    EARLY_STOP_ROUNDS = 50             # Reduced early stopping rounds to prevent overfitting
    
    # Target Mean Encoding parameters
    TME_MIN_SAMPLES_LEAF = 5           # Increased minimum samples for more stable encoding
    TME_SMOOTHING = 5                  # Increased smoothing effect to reduce noise
    TME_NOISE_LEVEL = 0.01             # Added small noise to prevent overfitting
    
    # Run cross-validation
    print("\nRunning cross-validation...")
    cv_scores, best_rounds, feat_importance_df = cross_validate(
        params, 
        X_train, 
        y_train, 
        n_folds=N_FOLDS, 
        max_boost_rounds=MAX_BOOST_ROUNDS, 
        early_stop_rounds=EARLY_STOP_ROUNDS,
        tme_min_samples_leaf=TME_MIN_SAMPLES_LEAF,
        tme_smoothing=TME_SMOOTHING,
        tme_noise_level=TME_NOISE_LEVEL
    )
    
    # Calculate optimal number of boosting rounds (105% of average best round)
    # Reduced coefficient to prevent overfitting
    optimal_rounds = int(np.mean(best_rounds) * 1.05)
    print(f"\nOptimal number of boosting rounds: {optimal_rounds}")
    
    # Training parameters for final model
    NUM_TRAIN_ROUNDS = 5
    SEED_VALUES = [42, 10, 2018, 38, 28]
    VERSION = 'v1'
    
    # Ensure there is a seed value for each training round
    assert NUM_TRAIN_ROUNDS == len(SEED_VALUES)
    
    # Apply target mean encoding for final training
    print('\nApplying target mean encoding for final training...')
    cat_feats = X_train.select_dtypes(['category']).columns.tolist()
    print(f"Found {len(cat_feats)} categorical features for final training")
    
    # Create encoded feature names for tracking
    encoded_features = {}
    
    # Apply target encoding to all categorical features
    for feature in cat_feats:
        try:
            encoded_train, encoded_test = target_encode(
                X_train[feature], 
                X_test[feature], 
                y_train,
                min_samples_leaf=tme_min_samples_leaf,
                smoothing=tme_smoothing,
                noise_level=0  # No noise for final training
            )
            
            # Create new feature name
            encoded_feature_name = f"{feature}_TARGET_ENCODED"
            encoded_features[feature] = encoded_feature_name
            
            # Add encoded features
            X_train[encoded_feature_name] = encoded_train
            X_test[encoded_feature_name] = encoded_test
            
            print(f"Encoded feature: {feature}")
        except Exception as e:
            print(f"Error encoding feature {feature}: {e}")
    
    print("Target encoding completed.")
    
    # Drop original categorical features
    X_train = X_train.drop(cat_feats, axis=1)
    X_test = X_test.drop(cat_feats, axis=1)
    
    # Train 7 models with different parameter variants and blend predictions
    models = []
    test_preds = []
    final_feat_importance = {}
    
    for i in range(1, 8):
        print(f"\nBegin Training Round {i} of 7.")
        
        # Set random seed and slightly different parameters for each model
        params_i = params.copy()
        params_i['seed'] = i
        
        # Slightly adjust parameters for each model
        if i > 1:  # Keep first model with original parameters as baseline
            # Adjust feature sampling ratio
            params_i['feature_fraction'] = max(0.5, params['feature_fraction'] - (i * 0.02))
            # Adjust sample sampling ratio
            params_i['bagging_fraction'] = max(0.5, params['bagging_fraction'] - (i * 0.02))
            # Alternately adjust regularization parameters
            if i % 2 == 0:
                params_i['lambda_l1'] = params['lambda_l1'] * 1.2
            else:
                params_i['lambda_l2'] = params['lambda_l2'] * 1.2
            # Alternately adjust learning rate
            if i % 3 == 0:
                params_i['learning_rate'] = params['learning_rate'] * 0.9
            elif i % 3 == 1:
                params_i['learning_rate'] = params['learning_rate'] * 1.1
        
        print(f"Model {i} parameters: learning_rate={params_i['learning_rate']}, feature_fraction={params_i['feature_fraction']}, lambda_l1={params_i['lambda_l1']}, lambda_l2={params_i['lambda_l2']}")
        
        # Train model
        model = train_lgbm(
            params_i,
            X_train,
            y_train,
            max_boost_rounds=optimal_rounds,
            early_stop_rounds=50
        )
        
        # Save model
        model_path = os.path.join('Models', f'lgbm_model_round_{i}.txt')
        model.save_model(model_path)
        print(f"Model saved to {os.path.abspath(model_path)}")
        
        # Save feature importances
        feature_names = model.feature_name()
        feature_importances = model.feature_importance(importance_type='split')
        
        # Update the feature importance dictionary
        for feat_name, importance in zip(feature_names, feature_importances):
            if feat_name in final_feat_importance:
                final_feat_importance[feat_name] += importance
            else:
                final_feat_importance[feat_name] = importance
        
        # Make predictions
        test_pred = model.predict(X_test)
        test_preds.append(test_pred)
        
        # Add to models list
        models.append(model)
        
        print(f"Training Round {i} of 7 complete.")
    
    # Average the feature importances
    for feat_name in final_feat_importance:
        final_feat_importance[feat_name] /= 5.0
    
    # Display feature importances
    display_feature_importances(final_feat_importance, for_cv=False)
    
    # Blend predictions
    y_scores = pd.concat([pd.Series(pred, name=f'TRAINING_ROUND_{i+1}') for i, pred in enumerate(test_preds)], axis=1)
    final_predictions = mean_rank_predictions(y_scores)
    
    # Save predictions
    write_pred_to_csv(test_ids, final_predictions, VERSION)
    
    end_time = time.time()
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()
