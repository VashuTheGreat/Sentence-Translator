from src.logger import *
from src.components.data_ingestion import Sentence_Data_Ingestion
from src.entity.config_entity import DataIngestionConfig
import asyncio
data_ingestion_config=DataIngestionConfig()
data_ingestion=Sentence_Data_Ingestion(data_ingestion_config=data_ingestion_config)
data_ingestion_artifacts=asyncio.run(data_ingestion.initiate_data_ingestion())
print(f"Data Ingestion Artifacts: {data_ingestion_artifacts}")