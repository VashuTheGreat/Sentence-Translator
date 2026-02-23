import asyncio
import sys
import os

# Add project root to path such that no error world come src not found
sys.path.append(os.getcwd())

from src.logger import logging
from src.exception import MyException
from src.pipelines.Data_Ingestion_Pipeline import DataIngestionPipeline
from src.pipelines.Data_Validation_Pipeline import DataValidationPipeline
from src.pipelines.Data_Transformation_Pipeline import DataTransformationPipeline
from src.pipelines.Model_Training_Pipeline import ModelTrainingPipeline

async def run_pipeline():
    try:
        logging.info("Starting the full training pipeline test")

        # Data Ingestion
        logging.info("--- Data Ingestion Phase ---")
        ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_artifact = await ingestion_pipeline.initiate_data_ingestion()
        logging.info(f"Data ingestion completed: {data_ingestion_artifact}")

        # Data Validation
        logging.info("--- Data Validation Phase ---")
        validation_pipeline = DataValidationPipeline()
        data_validation_artifact = await validation_pipeline.initiate_data_validation(
            data_ingestion_artifact=data_ingestion_artifact
        )
        logging.info(f"Data validation completed: {data_validation_artifact}")

        # Data Transformation
        logging.info("--- Data Transformation Phase ---")
        transformation_pipeline = DataTransformationPipeline()
        data_transformation_artifact = await transformation_pipeline.initiate_data_transformation(
            data_ingestion_artifact=data_ingestion_artifact
        )
        logging.info(f"Data transformation completed: {data_transformation_artifact}")

        # Model Training
        logging.info("--- Model Training Phase ---")
        training_pipeline = ModelTrainingPipeline()
        model_trainer_artifact = await training_pipeline.initiate_model_training(
            data_transformation_artifact=data_transformation_artifact
        )
        logging.info(f"Model training completed: {model_trainer_artifact}")

        logging.info("Full training pipeline test completed successfully")
        print("\nPipeline execution successful!")
        print(f"Final Artifact: {model_trainer_artifact}")

    except Exception as e:
        logging.exception("Error occurred in pipeline test")
        raise MyException(e, sys)

if __name__ == "__main__":
    asyncio.run(run_pipeline())