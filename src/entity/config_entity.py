from dataclasses import dataclass
from src.constants import *
from src.utils.main_utils import read_yaml_file_sync
import os
from datetime import datetime
# TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
TIMESTAMP: str = "artifacts_date_time"

@dataclass
class TrainingPipeline:
    artifacts_dir:str=os.path.join(ARTIFACT_DIR,TIMESTAMP)
    pipeline_name:str=PIPELINE_NAME
    timestamp:str=TIMESTAMP


training_pipeline:TrainingPipeline=TrainingPipeline()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str=os.path.join(training_pipeline.artifacts_dir,DATA_INGESTION_DIR_NAME)
    feature_store_file_path:str=os.path.join(data_ingestion_dir,FEATURE_STORE_FILE_DIR,FEATURE_STORE_FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    data_base_url:str = DATA_BASE_URL 


@dataclass
class DataValidationConfig:
    data_validation_dir:str=os.path.join(training_pipeline.artifacts_dir,DATA_VALIDATION_DIR_NAME)
    data_validation_file_path:str=os.path.join(data_validation_dir,DATA_VALIDATION_FILE_NAME)



@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline.artifacts_dir, DATA_TRANSFORMATION_DIR
    )
    transformed_train_file_path: str = os.path.join(
        data_transformation_dir, TRANSFORMED_TRAIN_FILE_NAME
    )
    transformed_test_file_path: str = os.path.join(
        data_transformation_dir, TRANSFORMED_TEST_FILE_NAME
    )
    transformed_train_csv_path: str = os.path.join(
        data_transformation_dir, TRANSFORMED_TRAIN_CSV_NAME
    )
    transformed_test_csv_path: str = os.path.join(
        data_transformation_dir, TRANSFORMED_TEST_CSV_NAME
    )
    en_vocab_file_path: str = os.path.join(
        data_transformation_dir, EN_VOCAB_NAME
    )
    hi_vocab_file_path: str = os.path.join(
        data_transformation_dir, HI_VOCAB_NAME
    )


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline.artifacts_dir, MODEL_TRAINING_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, MODEL_TRAINED_DIR, MODEL_FILE_NAME
        )

        self._config = read_yaml_file_sync(MODEL_TRAINING_CONFIG_FILE_PATH)["model_training"]
        self.embed_size: int = self._config["embed_size"]
        self.hidden_size: int = self._config["hidden_size"]
        self.batch_size: int = self._config["batch_size_training"]
        self.epochs: int = self._config["epochs"]
        self.learning_rate: float = self._config["learning_rate"]
        self.teacher_forcing_ratio: float = self._config["teacher_forcing_ratio"]

