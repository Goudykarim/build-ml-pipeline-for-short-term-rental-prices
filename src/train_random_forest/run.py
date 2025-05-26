#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer, StandardScaler

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline # Explicitly import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging.getLogger()


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    if not isinstance(dates, pd.DataFrame):
        dates_df = pd.DataFrame(dates)
    else:
        dates_df = dates
    
    date_sanitized = dates_df.apply(pd.to_datetime, errors='coerce')
    if isinstance(date_sanitized, pd.Series):
        max_date = date_sanitized.max()
        delta = (max_date - date_sanitized).dt.days
    else: 
        delta = date_sanitized.apply(lambda d_col: (d_col.max() - d_col).dt.days, axis=0)
    
    return delta.to_numpy().reshape(-1, 1)


def go(args):

    run = wandb.init(job_type="train_random_forest", group="training") 
    logger.info(f"W&B run initiated with id: {run.id}")
    run.config.update(vars(args)) 

    logger.info(f"Loading Random Forest configuration from: {args.rf_config_path}")
    with open(args.rf_config_path) as fp:
        rf_config = json.load(fp)
    run.config.update({"random_forest_specific_config": rf_config}) 
    logger.info(f"Random Forest config: {rf_config}")

    rf_config['random_state'] = args.random_seed

    logger.info(f"Fetching train/validation artifact: {args.trainval_artifact}")
    try:
        trainval_artifact_obj = run.use_artifact(args.trainval_artifact)
        trainval_local_path = trainval_artifact_obj.file()
        logger.info(f"Train/validation artifact downloaded to: {trainval_local_path}")
    except Exception as e:
        logger.error(f"Failed to download train/validation artifact {args.trainval_artifact}: {e}")
        run.finish()
        raise

    logger.info("Reading train/validation data...")
    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")  

    logger.info(f"Data shape: X-{X.shape}, y-{y.shape}")
    logger.info(f"Price stats - Min: {y.min()}, Max: {y.max()}, Mean: {y.mean():.2f}")

    logger.info(f"Splitting data into train and validation sets (val_size: {args.val_size}). Stratifying by: {args.stratify_by}")
    stratify_column = X[args.stratify_by] if args.stratify_by != "none" and args.stratify_by in X.columns else None
    if stratify_column is None and args.stratify_by != "none":
        logger.warning(f"Stratification column '{args.stratify_by}' not found or set to 'none'. Proceeding without stratification.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=stratify_column, random_state=args.random_seed
    )
    logger.info(f"Train set shape: X_train-{X_train.shape}, y_train-{y_train.shape}")
    logger.info(f"Validation set shape: X_val-{X_val.shape}, y_val-{y_val.shape}")


    logger.info("Preparing sklearn pipeline...")
    numerical_features_to_scale = [
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365",
        "longitude", "latitude"
    ] 
    
    sk_pipe, processed_original_features_list = get_inference_pipeline(
        rf_config, 
        args.max_tfidf_features,
        numerical_features_to_scale 
    )

    logger.info("Fitting the pipeline...")
    try:
        sk_pipe.fit(X_train, y_train)
        logger.info("Pipeline fitting complete.")
    except Exception as e:
        logger.error(f"Error during pipeline fitting: {e}")
        # Log traceback for more details
        import traceback
        logger.error(traceback.format_exc())
        run.finish()
        raise

    logger.info("Scoring model on validation set...")
    y_pred_val = sk_pipe.predict(X_val)
    
    r_squared_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)

    logger.info(f"Validation R^2 score: {r_squared_val:.4f}")
    logger.info(f"Validation MAE: {mae_val:.4f}")

    logger.info("Logging metrics to W&B...")
    run.summary['validation_r2'] = r_squared_val
    run.summary['validation_mae'] = mae_val
    wandb.log({
        "validation_r2": r_squared_val,
        "validation_mae": mae_val
    })

    logger.info("Exporting model pipeline with MLflow...")
    model_export_dir = "random_forest_model_dir" 
    if os.path.exists(model_export_dir):
        shutil.rmtree(model_export_dir)
    os.makedirs(model_export_dir)

    mlflow.sklearn.save_model(
        sk_model=sk_pipe,
        path=os.path.join(model_export_dir), 
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )
    logger.info(f"Model saved locally to {model_export_dir}")

    logger.info(f"Uploading model artifact '{args.output_artifact}' to W&B...")
    model_artifact = wandb.Artifact(
        name=args.output_artifact, 
        type="model", 
        description="Random Forest Regressor pipeline for NYC Airbnb price prediction.",
        metadata=dict(rf_config) 
    )
    model_artifact.add_dir(model_export_dir) 
    run.log_artifact(model_artifact)
    logger.info("Model artifact uploaded to W&B.")

    if hasattr(sk_pipe.named_steps['random_forest'], 'feature_importances_'):
        logger.info("Plotting and logging feature importance...")
        try:
            fig_feat_imp = plot_feature_importance(sk_pipe, processed_original_features_list, X_train) 
            if fig_feat_imp:
                 run.log({"feature_importance_plot": wandb.Image(fig_feat_imp)})
                 plt.close(fig_feat_imp) 
                 logger.info("Feature importance plot logged.")
            else:
                logger.warning("Feature importance plot was not generated.")
        except Exception as e:
            logger.error(f"Could not plot/log feature importances: {e}")
    else:
        logger.warning("Model in pipeline does not have feature_importances_ attribute.")

    run.finish()
    logger.info("train_random_forest step finished successfully.")

def plot_feature_importance(pipe, original_feature_names, X_train_df_for_names=None):
    logger.info("Attempting to plot feature importance.")
    rf_model = pipe.named_steps['random_forest']
    importances = rf_model.feature_importances_

    preprocessor = pipe.named_steps['preprocessor']
    feature_names_out = []
    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            if X_train_df_for_names is not None:
                 feature_names_out = list(preprocessor.get_feature_names_out(X_train_df_for_names.columns))
                 logger.info(f"Successfully got {len(feature_names_out)} feature names from preprocessor.")
            else:
                logger.warning("X_train_df_for_names not provided to plot_feature_importance, cannot use get_feature_names_out.")
        except Exception as e:
            logger.warning(f"Could not get feature names from preprocessor using get_feature_names_out: {e}. Falling back.")

    if not feature_names_out: 
        logger.info("Using a simplified approach for feature names based on original columns and counts.")
        num_total_transformed_features = importances.shape[0]
        if num_total_transformed_features == len(original_feature_names) and not any(isinstance(pipe.named_steps['preprocessor'].transformers_[i][1], (OneHotEncoder, TfidfVectorizer)) for i in range(len(pipe.named_steps['preprocessor'].transformers_))):
            feature_names_out = original_feature_names
        else:
            feature_names_out = [f"feature_{i}" for i in range(num_total_transformed_features)]
            logger.warning(f"Generated generic feature names as specific names could not be determined. Count: {num_total_transformed_features}")

    if len(feature_names_out) != len(importances):
        logger.error(f"Mismatch between number of feature names ({len(feature_names_out)}) and importances ({len(importances)}). Cannot plot.")
        return None 

    feature_importance_df = pd.DataFrame({'feature': feature_names_out, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_importance_df) * 0.3)))
    ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], align='center')
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Top Feature Importances')
    fig.tight_layout()
    logger.info("Feature importance plot generated.")
    return fig

def get_inference_pipeline(rf_config_model_params, max_tfidf_features, numerical_features_to_scale):
    ordinal_categorical = ["room_type"] 
    non_ordinal_categorical = ["neighbourhood_group"]

    ordinal_categorical_preproc = OrdinalEncoder(categories=[['Shared room', 'Private room', 'Entire home/apt', 'Hotel room']])

    non_ordinal_categorical_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) 
    ])

    numerical_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value=0)), 
        ('scaler', StandardScaler()) 
    ])
    numerical_features = numerical_features_to_scale 

    date_feature_engineering = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='2010-01-01')), 
        ('delta_days', FunctionTransformer(delta_date_feature, check_inverse=False, validate=False))
    ])

    # Corrected NLP pipeline for the "name" column
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1}, validate=False)
    name_tfidf_preproc = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value="")), 
        ('reshape', reshape_to_1d), # Ensures input to TfidfVectorizer is 1D
        ('tfidf', TfidfVectorizer(
            binary=False, 
            max_features=max_tfidf_features,
            stop_words='english'
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal_cat', ordinal_categorical_preproc, ordinal_categorical),
            ('non_ordinal_cat', non_ordinal_categorical_preproc, non_ordinal_categorical),
            ('numerical', numerical_preproc, numerical_features), 
            ('date_eng', date_feature_engineering, ['last_review']),
            ('name_tfidf', name_tfidf_preproc, ['name']) 
        ],
        remainder="drop",  
    )

    processed_original_features_list = ordinal_categorical + non_ordinal_categorical + numerical_features + ["days_since_last_review_group", "name_tfidf_group"]

    random_forest_model = RandomForestRegressor(**rf_config_model_params)

    sk_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("random_forest", random_forest_model)
    ])
    
    logger.info(f"Sklearn pipeline created with steps: {sk_pipe.steps}")

    return sk_pipe, processed_original_features_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model on the NYC Airbnb dataset.")

    parser.add_argument(
        "--trainval_artifact", type=str, help="Input W&B artifact for training data.", required=True
    )
    parser.add_argument(
        "--val_size", type=float, help="Fraction for validation set.", required=True
    )
    parser.add_argument(
        "--random_seed", type=int, help="Random seed.", default=42
    )
    parser.add_argument(
        "--stratify_by", type=str, help="Column for stratification.", default="none"
    )
    parser.add_argument(
        "--rf_config_path", type=str, help="Path to RF hyperparameters JSON.", required=True
    )
    parser.add_argument(
        "--max_tfidf_features", type=int, help="Max features for TF-IDF.", default=50
    )
    parser.add_argument(
        "--output_artifact", type=str, help="Name for W&B output model artifact.", required=True
    )

    args = parser.parse_args()
    go(args)
