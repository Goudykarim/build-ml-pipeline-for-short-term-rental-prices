import argparse
import logging
import pandas as pd
import wandb
import os # For checking file existence

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging.getLogger()

def go(args):
    """
    This function executes the data cleaning process.
    1. Downloads the input artifact from W&B.
    2. Performs data cleaning (price filtering, datetime conversion, geolocation filtering).
    3. Saves the cleaned data to a CSV file.
    4. Uploads the cleaned data as a new artifact to W&B.
    """
    run = wandb.init(job_type="basic_cleaning", group="data_prep")
    logger.info(f"W&B run initiated with id: {run.id}")
    run.config.update(vars(args)) # Log script arguments

    logger.info(f"Downloading artifact: {args.input_artifact}")
    try:
        artifact_obj = run.use_artifact(args.input_artifact) # Changed variable name for clarity
        local_path = artifact_obj.file()
        logger.info(f"Artifact {args.input_artifact} downloaded to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download artifact {args.input_artifact}: {e}")
        run.finish()
        raise

    logger.info("Reading downloaded data into pandas DataFrame...")
    try:
        df = pd.read_csv(local_path)
    except Exception as e:
        logger.error(f"Failed to read CSV from {local_path}: {e}")
        run.finish()
        raise

    # --- DATA CLEANING ---
    logger.info("Starting data cleaning process...")
    df_cleaned = df.copy() # Start with a copy

    # 1. Filter price outliers
    logger.info(f"Filtering prices between {args.min_price} and {args.max_price}")
    idx_price = df_cleaned['price'].between(float(args.min_price), float(args.max_price))
    df_cleaned = df_cleaned[idx_price].copy() # Apply filter and reassign
    logger.info(f"Shape after price filtering: {df_cleaned.shape}")

    # 2. Convert 'last_review' to datetime
    if 'last_review' in df_cleaned.columns:
        logger.info("Converting 'last_review' column to datetime")
        df_cleaned['last_review'] = pd.to_datetime(df_cleaned['last_review'], errors='coerce')
    else:
        logger.warning("'last_review' column not found. Skipping datetime conversion.")
    
    # 3. Geolocation Filtering (Added for Step 5.3)
    logger.info("Filtering data based on NYC geographical boundaries...")
    # Define reasonable approximate boundaries for NYC
    # These might need adjustment based on actual data distribution expected
    # Using boundaries from project instructions for test_proper_boundaries
    min_longitude, max_longitude = -74.25, -73.70 # Adjusted based on common NYC extent
    min_latitude, max_latitude = 40.47, 40.92    # Adjusted based on common NYC extent
    # The project notes for sample2.csv mentioned: -74.25, -73.50 and 40.5, 41.2
    # Let's use the ones from the project notes to ensure the fix works as intended for sample2.csv
    # min_longitude, max_longitude = -74.25, -73.50
    # min_latitude, max_latitude = 40.5, 41.2
    # Sticking to slightly wider ones for general robustness, but test_proper_boundaries will be the judge.
    # The key is that this filter is *added*. If test_proper_boundaries has specific values, align with those.
    # For the specific fix for sample2.csv as per project notes:
    min_longitude_fix, max_longitude_fix = -74.25, -73.50
    min_latitude_fix, max_latitude_fix = 40.5, 41.2

    if 'longitude' in df_cleaned.columns and 'latitude' in df_cleaned.columns:
        idx_geo = (df_cleaned['longitude'].between(min_longitude_fix, max_longitude_fix) &
                   df_cleaned['latitude'].between(min_latitude_fix, max_latitude_fix))
        
        num_original_rows = len(df_cleaned)
        df_cleaned = df_cleaned[idx_geo].copy() # Apply filter and reassign
        num_dropped_geo = num_original_rows - len(df_cleaned)

        if num_dropped_geo > 0:
            logger.info(f"Removed {num_dropped_geo} rows outside defined NYC geographical boundaries.")
        logger.info(f"Shape after geolocation filtering: {df_cleaned.shape}")
    else:
        logger.warning("Longitude or latitude columns not found. Skipping geolocation filtering.")

    logger.info("Data cleaning finished.")
    # --- END OF DATA CLEANING ---

    output_filename = "clean_sample.csv" 
    logger.info(f"Saving cleaned data to {output_filename}")
    try:
        df_cleaned.to_csv(output_filename, index=False)
    except Exception as e:
        logger.error(f"Failed to save cleaned data to {output_filename}: {e}")
        run.finish()
        raise

    logger.info(f"Creating W&B artifact: {args.output_artifact}")
    try:
        output_artifact_obj = wandb.Artifact( # Changed variable name
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        output_artifact_obj.add_file(output_filename)
        logger.info(f"Logging artifact {args.output_artifact} to W&B...")
        run.log_artifact(output_artifact_obj)
        output_artifact_obj.wait() 
        logger.info("Artifact logged successfully.")
    except Exception as e:
        logger.error(f"Failed to create or log W&B artifact {args.output_artifact}: {e}")
        run.finish()
        raise
    finally:
        if os.path.exists(output_filename):
            os.remove(output_filename)
            logger.info(f"Removed temporary file: {output_filename}")
    
    run.finish()
    logger.info("basic_cleaning step finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download from W&B the raw dataset and apply basic data cleaning, exporting the result to a new artifact"
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact name from W&B (e.g., sample.csv:latest)",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact (e.g., clean_sample.csv)",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the output artifact (e.g., clean_data)",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float, 
        help="Minimum price to consider for filtering outliers",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float, 
        help="Maximum price to consider for filtering outliers",
        required=True,
    )

    args = parser.parse_args()
    go(args)
