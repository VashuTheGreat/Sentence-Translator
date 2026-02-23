import sys
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.logger import logging
from src.components.data_validation import Sentence_data_validation
from src.entity.config_entity import DataValidationConfig
from src.exception import MyException

class DataValidationPipeline:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()

    async def initiate_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Entered the initiate_data_validation method of DataValidationPipeline")
            self.data_validation = Sentence_data_validation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            self.data_validation_artifacts = await self.data_validation.initiate_data_validation()
            logging.info("Exited the initiate_data_validation method of DataValidationPipeline")
            return self.data_validation_artifacts
        except Exception as e:
            raise MyException(e, sys)