import sys
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.components.data_ingestion import Sentence_Data_Ingestion
from src.entity.config_entity import DataIngestionConfig
from src.exception import MyException

class DataIngestionPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    async def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entered the initiate_data_ingestion method of DataIngestionPipeline")
            self.data_ingestion = Sentence_Data_Ingestion(data_ingestion_config=self.data_ingestion_config)
            self.data_ingestion_artifacts = await self.data_ingestion.initiate_data_ingestion()
            logging.info("Exited the initiate_data_ingestion method of DataIngestionPipeline")
            return self.data_ingestion_artifacts
        except Exception as e:
            raise MyException(e, sys)