import argparse
import logging
import pandas as pd
import wandb
import os # For checking file existence

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    """
    This function executes the data cleaning process.
    1. Downloads the input artifact from W&B.
    2. Performs data cleaning (price filtering, datetime conversion).
    3. Saves the cleaned data to a CSV file.
    4. Uploads the cleaned data as a new artifact to W&B.
    """
    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning", group="data_prep") # Match your W&B project
    logger.info(f"W&B run initiated with id: {run.id}")

    logger.info(f"Downloading artifact: {args.input_artifact}")
    try:
        artifact = run.use_artifact(args.input_artifact)
        local_path = artifact.file()
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

    # --- YOUR DATA CLEANING CODE HERE ---
    logger.info("Starting data cleaning process...")

    # 1. Filter price outliers
    logger.info(f"Filtering prices between {args.min_price} and {args.max_price}")
    idx = df['price'].between(float(args.min_price), float(args.max_price))
    df_cleaned = df[idx].copy()
    logger.info(f"Shape after price filtering: {df_cleaned.shape}")

    # 2. Convert 'last_review' to datetime
    # Check if 'last_review' column exists before trying to convert
    if 'last_review' in df_cleaned.columns:
        logger.info("Converting 'last_review' column to datetime")
        df_cleaned['last_review'] = pd.to_datetime(df_cleaned['last_review'], errors='coerce')
        # errors='coerce' will turn unparseable dates into NaT (Not a Time)
    else:
        logger.warning("'last_review' column not found in the DataFrame. Skipping conversion.")

    logger.info("Data cleaning finished.")
    # --- END OF YOUR DATA CLEANING CODE ---

    # Save cleaned data to a temporary CSV file
    output_filename = "clean_sample.csv" # Temporary local filename
    logger.info(f"Saving cleaned data to {output_filename}")
    try:
        df_cleaned.to_csv(output_filename, index=False)
    except Exception as e:
        logger.error(f"Failed to save cleaned data to {output_filename}: {e}")
        run.finish()
        raise

    # Upload the cleaned data as a new artifact to W&B
    logger.info(f"Creating W&B artifact: {args.output_artifact}")
    try:
        artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(output_filename) # Add the local file to the artifact
        logger.info(f"Logging artifact {args.output_artifact} to W&B...")
        run.log_artifact(artifact)
        artifact.wait() # Wait for the artifact to be uploaded
        logger.info("Artifact logged successfully.")
    except Exception as e:
        logger.error(f"Failed to create or log W&B artifact {args.output_artifact}: {e}")
        run.finish()
        raise
    finally:
        # Clean up the local temporary file
        if os.path.exists(output_filename):
            os.remove(output_filename)
            logger.info(f"Removed temporary file: {output_filename}")

    run.finish()
    logger.info("basic_cleaning step finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact"
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
        help="Type for the output artifact (e.g., clean_sample)",
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
        type=float, # Changed to float to match MLproject and usage
        help="Minimum price to consider for filtering outliers",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float, # Changed to float to match MLproject and usage
        help="Maximum price to consider for filtering outliers",
        required=True,
    )

    args = parser.parse_args()
    go(args)
