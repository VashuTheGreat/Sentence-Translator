import sys
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.logger import logging
from src.components.data_transformation import Sentence_data_transformer
from src.entity.config_entity import DataTransformationConfig
from src.exception import MyException

class DataTransformationPipeline:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    async def initiate_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Entered the initiate_data_transformation method of DataTransformationPipeline")
            self.data_transformation = Sentence_data_transformer(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            self.data_transformation_artifacts = await self.data_transformation.initiate_data_transformation()
            logging.info("Exited the initiate_data_transformation method of DataTransformationPipeline")
            return self.data_transformation_artifacts
        except Exception as e:
            raise MyException(e, sys)