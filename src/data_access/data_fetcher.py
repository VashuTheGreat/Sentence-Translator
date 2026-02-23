from abc import ABC, abstractmethod
import pandas as pd
from src.constants import TRAIN_SPLIT, TEST_SPLIT, VALIDATE_SPLIT, DATA_BASE_URL
from src.exception import MyException
from typing import Optional, Literal
import sys
import logging

class DataFetcher(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    async def export_data_as_df(self) -> pd.DataFrame:
        pass

class SentenceDataFetcher(DataFetcher):
    def __init__(self, url: str = DATA_BASE_URL):
        super().__init__()
        self.url = url
    
    @staticmethod
    async def recompile_data(data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['English'] = data['translation'].apply(lambda x: x['en'])
            data['Hindi'] = data['translation'].apply(lambda x: x['hi'])
            data = data.drop('translation', axis=1)
            return data
        except Exception as e:
            raise MyException(e, sys)
    
    async def export_data_as_df(self, split: Literal['train', 'validation', 'test'] = "train") -> pd.DataFrame:
        try:
            logging.info("Exporting data from export_data_as_df method")
            splits = {
                "train": TRAIN_SPLIT,
                "validation": VALIDATE_SPLIT,
                "test": TEST_SPLIT
            }
            data: pd.DataFrame = pd.read_parquet(self.url + splits[split])
            # data=data[:100]
            data = await SentenceDataFetcher.recompile_data(data=data)
            return data
        except Exception as e:
            raise MyException(e, sys)
