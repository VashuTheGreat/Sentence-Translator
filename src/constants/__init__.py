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