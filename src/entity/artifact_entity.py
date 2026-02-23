from dataclasses import dataclass
from src.constants import *
import os
from typing import Optional

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str
    features_file_path:str
     
    



@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:Optional[str]
    validation_report_file_path:str    

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    en_vocab_file_path:str
    hi_vocab_file_path:str


@dataclass
class ModelTrainerArtifact:
    model_file_path:str
    loss_history:Optional[str]
    en_dat_path:str
    hi_dat_path:str
    en_vocab_path:str
    hi_vocab_path:str
    