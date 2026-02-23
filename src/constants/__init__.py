import os
import torch

ARTIFACT_DIR="artifacts"
PIPELINE_NAME="Sentence_translator"
DATA_INGESTION_DIR_NAME="data_ingestion"
FEATURE_STORE_FILE_DIR="features"
FEATURE_STORE_FILE_NAME="data.csv"
DATA_INGESTION_INGESTED_DIR="ingestion"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO=0.2
TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"


# Data Fetcher

TRAIN_SPLIT='data/train-00000-of-00001.parquet'
VALIDATE_SPLIT='data/validation-00000-of-00001.parquet'
TEST_SPLIT='data/test-00000-of-00001.parquet'
DATA_BASE_URL="hf://datasets/cfilt/iitb-english-hindi/"


# Data Validation
DATA_YAML_SCHEMA_FILE_PATH=os.path.join("config","data_validation.yaml")
DATA_VALIDATION_DIR_NAME="validation"
DATA_VALIDATION_FILE_NAME="validation_result.yaml"
MODEL_TRAINING_CONFIG_FILE_PATH=os.path.join("config","model_training.yaml")


# Data Transformation

DATA_TRANSFORMATION_DIR="transformed_data"
TRANSFORMED_TRAIN_FILE_NAME="en.dat"
TRANSFORMED_TEST_FILE_NAME="hi.dat"
TRANSFORMED_TRAIN_CSV_NAME="train.csv"
TRANSFORMED_TEST_CSV_NAME="test.csv"
EN_VOCAB_NAME = "en_vocab.pth"
HI_VOCAB_NAME = "hi_vocab.pth"
MAX_SEQ_LEN = 100


# Model Training

MODEL_TRAINING_DIR_NAME="model_training"
MODEL_FILE_NAME="model.pth"
MODEL_TRAINED_DIR="trained_model"



# Model Config
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'




# Prediction Config

model_path=os.path.join("saved_model","model.pth")
en_vocab_path=os.path.join("saved_model","en_vocab.pth")
hi_vocab_path=os.path.join("saved_model","hi_vocab.pth")
en_dat=os.path.join("saved_model","en.dat")
hi_dat=os.path.join("saved_model","hi.dat")
