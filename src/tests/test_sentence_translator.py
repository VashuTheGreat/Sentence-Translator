from numpy.testing import print_assert_equal
from src.logger import *
from src.components.data_ingestion import Sentence_Data_Ingestion
from src.components.data_validation import Sentence_data_validation
from src.components.data_transformation import Sentence_data_transformer
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
import asyncio
from src.components.model_training import Sentence_model_trainer
# data ingestion intiation config
data_ingestion_config=DataIngestionConfig()

# data ingestion
data_ingestion=Sentence_Data_Ingestion(data_ingestion_config=data_ingestion_config)
data_ingestion_artifacts=asyncio.run(data_ingestion.initiate_data_ingestion())


# data validation config
data_validation_config=DataValidationConfig()

# data validation
data_validation=Sentence_data_validation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifacts)
data_validation_artifacts=asyncio.run(data_validation.initiate_data_validation())


# data transformation config
data_transformation_config=DataTransformationConfig()

# data  Transformation
data_transformation=Sentence_data_transformer(
    data_transformation_config=data_transformation_config,
    data_ingestion_artifact=data_ingestion_artifacts
)
data_transformation_artifact=asyncio.run(data_transformation.initiate_data_transformation())



# data training config
data_trainer=Sentence_model_trainer(
    data_transformation_artifact=data_transformation_artifact,
    model_trainer_config=data_trainer_config
)
data_trainer_artifact=asyncio.run(data_trainer.initiate_model_training())
print(data_trainer_artifact)