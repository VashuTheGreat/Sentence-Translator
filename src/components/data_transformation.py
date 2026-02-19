from abc import ABC, abstractmethod
import logging
import sys
import os
import pandas as pd
import nltk
from tqdm import tqdm
tqdm.pandas()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact
)
from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils import read_yaml_file_sync, save_numpy_array_data
from src.constants import DATA_YAML_SCHEMA_FILE_PATH
from src.exception import MyException
from collections import Counter
from src.utils.main_utils import save_object


def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


download_nltk_resources()


def build_vocab(sentences):
    """
    Builds a vocabulary from a list of sentences.
    """
    counter = Counter()
    for sentence in sentences:
        for word in str(sentence).split():
            counter[word] += 1

    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}

    current_index = 3
    for word, _ in counter.items():
        if word not in vocab:
            vocab[word] = current_index
            current_index += 1

    return vocab




class Data_Transformer(ABC):

    def __init__(self):
        super().__init__()
        logging.info("Data_Transformer base class initialized")

    @abstractmethod
    async def initiate_data_transformation(self) -> DataTransformationArtifact:
        pass



class Sentence_data_transformer(Data_Transformer):

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        super().__init__()

        logging.info("Initializing Sentence_data_transformer")

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config
        self._schema = read_yaml_file_sync(file_path=DATA_YAML_SCHEMA_FILE_PATH)

        self.stop_words = set(stopwords.words("english"))

        logging.info("Schema loaded successfully for transformation")

    
    def clean_text(self, text: str, lang: str = "english") -> str:
        try:
            if pd.isna(text):
                return ""

            text = str(text).lower()
            tokens = word_tokenize(text)

            tokens = [
                token for token in tokens
                if token not in self.stop_words and token.isalpha()
            ]

            return " ".join(tokens)

        except Exception as e:
            logging.exception("Error during text cleaning")
            raise e


    async def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            logging.info("Starting data transformation process")

            # Read train & test data
            logging.info("Reading train dataset")
            data_train = pd.read_csv(
                self.data_ingestion_artifact.train_file_path
            )

            logging.info("Reading test dataset")
            data_test = pd.read_csv(
                self.data_ingestion_artifact.test_file_path
            )

            # Apply cleaning on schema columns
            logging.info("Applying text cleaning on schema columns")

            for col in tqdm(self._schema["columns"]):
                if col not in data_train.columns:
                    raise Exception(f"Column '{col}' not found in training data")

                logging.info(f"Cleaning column: {col}")

                data_train[col] = data_train[col].progress_apply(self.clean_text)
                data_test[col] = data_test[col].progress_apply(self.clean_text)

            # Create output directory
            os.makedirs(
                self.data_transformation_config.data_transformation_dir,
                exist_ok=True
            )
            os.makedirs(
                os.path.dirname(self.data_transformation_config.transformed_object_file_path),
                exist_ok=True
            )

            # Build vocabulary
            vocabs={}
            for col in self._schema["columns"]:
                logging.info("Building vocabulary")
                vocab = build_vocab(data_train[col])
                vocabs[col] = vocab

            # Save vocabulary
            logging.info("Saving vocabulary")
            await save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=vocabs
            )


            # Save train & test data
            logging.info("Saving transformed train CSV data")
            data_train.to_csv(
                self.data_transformation_config.transformed_train_csv_path,
                index=False
            )

            logging.info("Saving transformed test CSV data")
            data_test.to_csv(
                self.data_transformation_config.transformed_test_csv_path,
                index=False
            )


            # Save numpy arrays
            logging.info("Saving transformed train numpy array")
            await save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=data_train.values
            )

            logging.info("Saving transformed test numpy array")
            await save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=data_test.values
            )

            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

            logging.info("Data transformation completed successfully")

            return data_transformation_artifact

        except Exception as e:
            logging.exception("Error occurred during data transformation")
            raise MyException(e, sys)
