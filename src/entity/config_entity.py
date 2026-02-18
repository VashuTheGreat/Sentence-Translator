from dataclasses import dataclass
from src.constants import *
import os
from datetime import datetime
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

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


