import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import load_object, load_numpy_array_data
from src.exception import MyException
from src.logger import logging
from src.constants import DEVICE, MAX_SEQ_LEN
from src.entity.estimator import Encoder, Decoder, seq2seq

class Sentence_model_trainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        logging.info("Initializing Sentence_model_trainer")
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    class DatasetLoader(Dataset):
        def __init__(self, data_en, data_hi):
            self.data_en = data_en
            self.data_hi = data_hi

        def __len__(self):
            return len(self.data_en)

        def __getitem__(self, idx):
            en_toks = self.data_en[idx]
            hi_toks = self.data_hi[idx]
            return torch.tensor(en_toks, dtype=torch.long), torch.tensor(hi_toks, dtype=torch.long)

    def collate_fn(self, batch, pad_idx):
        en_batch, hi_batch = [], []
        for en_tokens, hi_tokens in batch:
            en_batch.append(en_tokens)
            hi_batch.append(hi_tokens)
        
        en_padded = nn.utils.rnn.pad_sequence(en_batch, batch_first=True, padding_value=pad_idx)
        hi_padded = nn.utils.rnn.pad_sequence(hi_batch, batch_first=True, padding_value=pad_idx)
        
        return en_padded, hi_padded

    async def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training process")

            logging.info("Loading vocabularies from transformation artifacts")
            vocab_en = torch.load(self.data_transformation_artifact.en_vocab_file_path, weights_only=False)
            vocab_hi = torch.load(self.data_transformation_artifact.hi_vocab_file_path, weights_only=False)

            logging.info("Loading memory-mapped tokenized data")
            
            
            en_dat_path = self.data_transformation_artifact.transformed_train_file_path
            file_size = os.path.getsize(en_dat_path)
            num_train = file_size // (MAX_SEQ_LEN * 4) # file size
            shape = (num_train, MAX_SEQ_LEN)

            data_train_en = await load_numpy_array_data(
                en_dat_path,
                mmap_mode='r',
                shape=shape
            )
            data_train_hi = await load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path,
                mmap_mode='r',
                shape=shape
            )

            dataset = self.DatasetLoader(data_train_en, data_train_hi)
            pad_idx = vocab_hi.get('<pad>', 0)
            
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self.model_trainer_config.batch_size,
                shuffle=True,
                collate_fn=lambda x: self.collate_fn(x, pad_idx)
            )

            input_size_en = len(vocab_en)
            output_size_hi = len(vocab_hi)

            encoder = Encoder(
                input_size=input_size_en,
                embed_size=self.model_trainer_config.embed_size,
                hidden_size=self.model_trainer_config.hidden_size,
            )
            decoder = Decoder(
                output_size=output_size_hi,
                embed_size=self.model_trainer_config.embed_size,
                hidden_size=self.model_trainer_config.hidden_size,
            )

            model = seq2seq(encoder, decoder).to(DEVICE)

            criterion = nn.CrossEntropyLoss(ignore_index=vocab_hi["<pad>"])
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.model_trainer_config.learning_rate
            )

            logging.info(f"Training on device: {DEVICE}")

            for epoch in range(self.model_trainer_config.epochs):
                model.train()
                total_loss = 0

                for src_tensor, trg_tensor in data_loader:
                    src_tensor = src_tensor.to(DEVICE)
                    trg_tensor = trg_tensor.to(DEVICE)

                    optimizer.zero_grad()
                    output = model(src_tensor, trg_tensor, self.model_trainer_config.teacher_forcing_ratio)

                    trg_target = trg_tensor[:, 1:]
                    output_dim = output.shape[-1]
                    
                    loss = criterion(
                        output.reshape(-1, output_dim),
                        trg_target.reshape(-1),
                    )

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(data_loader)
                logging.info(f"Epoch {epoch+1} completed, Loss: {avg_loss:.4f}")

            logging.info("Saving training artifacts")
            model_trainer_dir = self.model_trainer_config.model_trainer_dir
            model_path = self.model_trainer_config.trained_model_file_path
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

            logging.info(f"Model saved in {model_trainer_dir}")

            model_trainer_artifact = ModelTrainerArtifact(
                model_file_path=model_path,
                loss_history=str(avg_loss),
                en_dat_path=self.data_transformation_artifact.transformed_train_file_path,
                hi_dat_path=self.data_transformation_artifact.transformed_test_file_path,
                en_vocab_path=self.data_transformation_artifact.en_vocab_file_path,
                hi_vocab_path=self.data_transformation_artifact.hi_vocab_file_path
            )

            return model_trainer_artifact

        except Exception as e:
            logging.exception("Error during model training")
            raise MyException(e, sys)


            
