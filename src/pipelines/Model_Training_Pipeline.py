import sys
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.logger import logging
from src.components.model_training import Sentence_model_trainer
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException

class ModelTrainingPipeline:
    def __init__(self):
        self.model_training_config = ModelTrainerConfig()

    async def initiate_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Entered the initiate_model_training method of ModelTrainingPipeline")
            self.model_training = Sentence_model_trainer(
                model_trainer_config=self.model_training_config,
                data_transformation_artifact=data_transformation_artifact
            )
            self.model_training_artifacts = await self.model_training.initiate_model_training()
            logging.info("Exited the initiate_model_training method of ModelTrainingPipeline")
            return self.model_training_artifacts
        except Exception as e:
            raise MyException(e, sys)