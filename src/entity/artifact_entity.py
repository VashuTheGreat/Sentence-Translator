from dataclasses import dataclass
from src.constants import *
import os

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str
     
    

