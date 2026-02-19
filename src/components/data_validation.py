from src.exception import MyException
import sys
import logging
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.data_access.data_fetcher import SentenceDataFetcher
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import os
from pandas import DataFrame
import pandas as pd
from src.utils.main_utils import read_yaml_file_sync,write_yaml_file
from src.constants import DATA_YAML_SCHEMA_FILE_PATH
from typing import List

class Data_Validator(ABC):
    def __init__(self):
        super().__init__()
        logging.info("Data_Validator initialized")

    @abstractmethod
    async def initiate_data_validation(self) -> DataValidationArtifact:
        pass


class Sentence_data_validation(Data_Validator):
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,data_validation_config:DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config=data_validation_config
        logging.info("Initializing Sentence_data_validation")
        self._schema = read_yaml_file_sync(file_path=DATA_YAML_SCHEMA_FILE_PATH)
        self.validation_message:List[str]=[]
        logging.info("Schema loaded successfully")

    async def validate_no_columns(self, data: pd.DataFrame):
        try:
            logging.info("Validating number of columns")
            columns = data.columns
            logging.info(f"Expected columns: {len(self._schema['columns'])}, Found columns: {len(columns)}")
            if not len(columns) == len(self._schema['columns']):
                logging.error("Number of columns mismatched")
                
                self.validation_message.append("no of columns mismatched")
            logging.info("Column count validation passed")
        except Exception as e:
            logging.exception("Error occurred during column count validation")
            raise MyException(e, sys)

    async def validate_features(self, data: pd.DataFrame):
        try:
            logging.info("Validating feature names")
            features = self._schema['columns']
            for i in features:
                if i not in data.columns:
                    logging.error(f"Feature not found: {i}")
                    self.validation_message.append(f"Feature not found: {i}")
            logging.info("Feature validation passed")
        except Exception as e:
            logging.exception("Error occurred during feature validation")
            raise MyException(e, sys)

    async def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation process")
            logging.info(f"Reading features file from {self.data_ingestion_artifact.features_file_path}")
            data = pd.read_csv(self.data_ingestion_artifact.features_file_path)
            logging.info("Features file loaded successfully")

            await self.validate_no_columns(data=data)
            await self.validate_features(data=data)


            data_validation_artifact = DataValidationArtifact(
                validation_status=True if not self.validation_message else False,
                message=self.validation_message,
                validation_report_file_path=self.data_validation_config.data_validation_file_path
                )

            
            await write_yaml_file(file_path=self.data_validation_config.data_validation_file_path,content=data_validation_artifact)

            logging.info("Data validation completed successfully")
            return data_validation_artifact

        except Exception as e:
            logging.exception("Error occurred during data validation process")
            raise MyException(e, sys)
