from src.exception import MyException
import sys
import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.data_fetcher import SentenceDataFetcher
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import os
from pandas import DataFrame
import pandas as pd

class Data_Ingestion(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def initiate_data_ingestion(self) -> DataIngestionArtifact:
        pass

class Sentence_Data_Ingestion(Data_Ingestion):
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        super().__init__()
        self.data_ingestion_config = data_ingestion_config
    
    async def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        try:
            train_test_ratio = self.data_ingestion_config.train_test_split_ratio
            train, test = train_test_split(dataframe, random_state=42, shuffle=True, test_size=train_test_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Created directories for train and test files")

            logging.info("Exporting train and test file path.")
            train.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file path.")
            logging.info("Exited split_data_as_train_test method")
        except Exception as e:
            logging.error(f"Error in split_data_as_train_test: {str(e)}")
            raise MyException(e, sys)
    
    async def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Entered export_data_into_feature_store method")
            data_fetcher = SentenceDataFetcher(url=self.data_ingestion_config.data_base_url)
            logging.info("DataFetcher initialized successfully")
            
            data = await data_fetcher.export_data_as_df(split='train')
            logging.info("Data fetched successfully from parquet")
            
            dir_path = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Created directory for feature store")

            data.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            logging.info(f"Data exported to feature store. Shape: {data.shape}")
            logging.info("Exited export_data_into_feature_store method")
            return data
        except Exception as e:
            logging.error(f"Error in export_data_into_feature_store: {str(e)}")
            raise MyException(e, sys)

    async def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entered initiate_data_ingestion method")
            data = await self.export_data_into_feature_store()
            logging.info("Feature store export completed")
            
            await self.split_data_as_train_test(dataframe=data)
            logging.info("Train-test split completed")
            
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                features_file_path=self.data_ingestion_config.feature_store_file_path
            )
            logging.info("DataIngestionArtifact created successfully")
            logging.info("Data ingestion pipeline completed successfully")
            logging.info("Exited initiate_data_ingestion method")
            return data_ingestion_artifact
        except Exception as e:
            logging.error(f"Error in initiate_data_ingestion: {str(e)}")
            raise MyException(e, sys)
