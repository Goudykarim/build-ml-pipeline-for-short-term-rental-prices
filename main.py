import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging.getLogger()

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_path=".", config_name='config', version_base=None)
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    if "download" in active_steps:
        logger.info("Running download step...")
        _ = mlflow.run(
            f"{config['main']['components_repository']}/get_data",
            "main",
            version='main',
            parameters={
                "sample": config["etl"]["sample"],
                "artifact_name": "sample.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw file as downloaded"
            },
        )
        logger.info("Download step finished.")

    if "basic_cleaning" in active_steps:
        logger.info("Running basic_cleaning step...")
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
            "main",
            parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_data",
                "output_description": "Data with price outliers filtered and last_review converted to datetime",
                "min_price": config['etl']['min_price'],
                "max_price": config['etl']['max_price']
            },
        )
        logger.info("basic_cleaning step finished.")

    if "data_check" in active_steps:
        logger.info("Running data_check step...")
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
            "main",
            parameters={
                "csv": "clean_sample.csv:latest",
                "ref": "clean_sample.csv:reference",
                "kl_threshold": config["data_check"]["kl_threshold"],
                "min_price": config['etl']['min_price'],
                "max_price": config['etl']['max_price']
            },
        )
        logger.info("data_check step finished.")

    if "data_split" in active_steps:
        logger.info("Running data_split step...")
        _ = mlflow.run(
            f"{config['main']['components_repository']}/train_val_test_split",
            "main",
            parameters={
                "input": "clean_sample.csv:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"]
            },
        )
        logger.info("data_split step finished.")

    if "train_random_forest" in active_steps:
        logger.info("Running train_random_forest step...")
        
        rf_config_filename = "rf_config.json"
        rf_config_path = os.path.join(hydra.utils.get_original_cwd(), rf_config_filename)

        logger.info(f"Creating Random Forest configuration file at {rf_config_path}")
        with open(rf_config_path, "w+") as fp:
            serializable_rf_config = OmegaConf.to_container(config["modeling"]["random_forest"], resolve=True)
            json.dump(serializable_rf_config, fp)

        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
            "main", 
            parameters={
                "trainval_artifact": "trainval_data.csv:latest", 
                "val_size": config["modeling"]["val_size"], 
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"].get("stratify_by_for_trainval_split", config["modeling"]["stratify_by"]), 
                "rf_config_path": rf_config_path, 
                "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                "output_artifact": "random_forest_export" 
            },
        )
        logger.info("train_random_forest step finished.")

    if "test_regression_model" in active_steps:
        logger.info("Running test_regression_model step...")
        _ = mlflow.run(
            f"{config['main']['components_repository']}/test_regression_model", 
            "main",
            parameters={
                "mlflow_model": "random_forest_export:prod", 
                "test_dataset": "test_data.csv:latest" # Parameter 'performance_metric' removed
                # The component's MLproject defines 'performance_metric', but its run.py does not accept it.
            },
        )
        logger.info("test_regression_model step finished.")

if __name__ == "__main__":
    go()
