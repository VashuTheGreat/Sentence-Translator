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
import torch
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
    Builds a vocabulary from a list of sentences as a dictionary mapping words to IDs.
    """
    counter = Counter()
    for sentence in sentences:
        for word in str(sentence).split():
            counter[word] += 1

    # Special tokens
    vocab = {"<pad>": 0, "<pos>": 1, "<eos>": 2}

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


import numpy as np
from src.constants import MAX_SEQ_LEN

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

    def tokenize_and_pad(self, sentence, vocab, max_len=MAX_SEQ_LEN):
        tokens = [vocab.get('<pos>', 1)]
        for word in str(sentence).split():
            tokens.append(vocab.get(word, vocab.get('<unk>', 0)))
        tokens.append(vocab.get('<eos>', 2))
        
        # Pad or truncate
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([vocab.get('<pad>', 0)] * (max_len - len(tokens)))
        return tokens

    async def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            logging.info("Starting data transformation process")

            logging.info("Reading train dataset")
            data_train = pd.read_csv(
                self.data_ingestion_artifact.train_file_path
            )

            logging.info("Reading test dataset")
            data_test = pd.read_csv(
                self.data_ingestion_artifact.test_file_path
            )

            logging.info("Applying text cleaning on schema columns")

            for col in tqdm(self._schema["columns"]):
                if col not in data_train.columns:
                    raise Exception(f"Column '{col}' not found in training data")

                logging.info(f"Cleaning column: {col}")

                data_train[col] = data_train[col].progress_apply(self.clean_text)
                data_test[col] = data_test[col].progress_apply(self.clean_text)

            # Build vocabularies from English and Hindi cleaned columns
            logging.info("Building vocabularies")
            vocab_en = build_vocab(data_train["English"])
            vocab_hi = build_vocab(data_train["Hindi"])

            # Create output directory
            os.makedirs(
                self.data_transformation_config.data_transformation_dir,
                exist_ok=True
            )

            # Save vocabularies as .pth artifacts
            logging.info("Saving en_vocab.pth and hi_vocab.pth")
            torch.save(vocab_en, self.data_transformation_config.en_vocab_file_path)
            torch.save(vocab_hi, self.data_transformation_config.hi_vocab_file_path)

            # Tokenize and save using memmap
            logging.info("Creating memory-mapped data files")
            num_train = len(data_train)
            
            # Create English training memmap
            en_train_mmap = np.memmap(
                self.data_transformation_config.transformed_train_file_path,
                dtype='int32', mode='w+', shape=(num_train, MAX_SEQ_LEN)
            )
            # Create Hindi training memmap
            hi_train_mmap = np.memmap(
                self.data_transformation_config.transformed_test_file_path,
                dtype='int32', mode='w+', shape=(num_train, MAX_SEQ_LEN)
            )

            logging.info("Populating memory-mapped arrays")
            for i, row in enumerate(data_train.itertuples()):
                en_train_mmap[i] = self.tokenize_and_pad(row.English, vocab_en)
                hi_train_mmap[i] = self.tokenize_and_pad(row.Hindi, vocab_hi)
            
            # Flush and close
            en_train_mmap.flush()
            hi_train_mmap.flush()
            del en_train_mmap
            del hi_train_mmap

            logging.info("Saving transformed CSV for reference")
            data_train.to_csv(self.data_transformation_config.transformed_train_csv_path, index=False)
            data_test.to_csv(self.data_transformation_config.transformed_test_csv_path, index=False)

            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                en_vocab_file_path=self.data_transformation_config.en_vocab_file_path,
                hi_vocab_file_path=self.data_transformation_config.hi_vocab_file_path
            )

            logging.info("Data transformation completed successfully")

            return data_transformation_artifact

        except Exception as e:
            logging.exception("Error occurred during data transformation")
            raise MyException(e, sys)
