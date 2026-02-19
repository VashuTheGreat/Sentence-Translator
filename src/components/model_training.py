from abc import ABC, abstractmethod
import logging
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from src.entity.artifact_entity import DataTransformationArtifact
from src.exception import MyException



from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import load_object, load_numpy_array_data, save_object


class Model(ABC):
    def __init__(self):
        super().__init__()
        logging.info("Model base class initialized")

    @abstractmethod
    async def returm_model_architecture(self):
        pass

    @abstractmethod
    async def initiate_model_training(self):
        pass


class Sentence_model_trainer(Model):
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        super().__init__()
        logging.info("Initializing Sentence_model_trainer")
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def encode_sentence(self, sentence, vocab):
        logging.debug("Encoding sentence")
        tokens = [vocab["<sos>"]]
        tokens += [vocab.get(word, vocab.get("<unk>")) for word in sentence.split()]
        tokens.append(vocab["<eos>"])
        return tokens

    class DatasetLoader(Dataset):
        def __init__(self, df):
            logging.info("DatasetLoader initialized")
            self.df = df

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            return {
                "English": self.df.iloc[idx]["English"],
                "Hindi": self.df.iloc[idx]["Hindi"],
            }

    def collate_batch(self, batch, vocab_en, vocab_hi):
        logging.debug("Collating batch")
        english_sentences = [item["English"] for item in batch]
        hindi_sentences = [item["Hindi"] for item in batch]

        encoded_en = [
            torch.tensor(self.encode_sentence(s, vocab_en), dtype=torch.long)
            for s in english_sentences
        ]
        encoded_hi = [
            torch.tensor(self.encode_sentence(s, vocab_hi), dtype=torch.long)
            for s in hindi_sentences
        ]

        padded_en = rnn_utils.pad_sequence(
            encoded_en, batch_first=False, padding_value=vocab_en["<pad>"]
        )
        padded_hi = rnn_utils.pad_sequence(
            encoded_hi, batch_first=False, padding_value=vocab_hi["<pad>"]
        )

        return {"English": padded_en, "Hindi": padded_hi}

    class Encoder(nn.Module):
        def __init__(self, input_size, embed_size, hidden_size):
            super().__init__()
            logging.info("Encoder initialized")
            self.embedding = nn.Embedding(input_size, embed_size)
            self.rnn = nn.GRU(embed_size, hidden_size)

        def forward(self, x):
            embedding = self.embedding(x)
            outputs, hidden = self.rnn(embedding)
            return hidden

    class Decoder(nn.Module):
        def __init__(self, output_size, embed_size, hidden_size):
            super().__init__()
            logging.info("Decoder initialized")
            self.embedding = nn.Embedding(output_size, embed_size)
            self.rnn = nn.GRU(embed_size, hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x, hidden):
            x = x.unsqueeze(0)
            embedded = self.embedding(x)
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc(output.squeeze(0))
            return prediction, hidden

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            logging.info("Seq2Seq model initialized")
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            hidden = self.encoder(src)
            input_token = trg[0]
            outputs = []

            for t in range(1, len(trg)):
                output, hidden = self.decoder(input_token, hidden)
                outputs.append(output)
                top1 = output.argmax(1)
                input_token = (
                    trg[t]
                    if torch.rand(1).item() < teacher_forcing_ratio
                    else top1
                )

            return torch.stack(outputs)

    async def returm_model_architecture(self):
        logging.info("Returning model architecture placeholder")
        return "Seq2Seq GRU Architecture"

    async def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training process")

            # Load vocabulary
            logging.info("Loading vocabulary")
            vocabs = await load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )
            vocab_en = vocabs["English"]
            vocab_hi = vocabs["Hindi"]

            # Load data
            logging.info("Loading train data")
            data_train = await load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            df_train = pd.DataFrame(data_train, columns=["English", "Hindi"])

            datasetloader = self.DatasetLoader(df_train)

            data_loader = DataLoader(
                dataset=datasetloader,
                batch_size=self.model_trainer_config.batch_size,
                shuffle=True,
                collate_fn=lambda b: self.collate_batch(b, vocab_en, vocab_hi),
            )

            logging.info("DataLoader created successfully")

            input_size_en = len(vocab_en)
            output_size_hi = len(vocab_hi)

            encoder = self.Encoder(
                input_size=input_size_en,
                embed_size=self.model_trainer_config.embed_size,
                hidden_size=self.model_trainer_config.hidden_size,
            )
            decoder = self.Decoder(
                output_size=output_size_hi,
                embed_size=self.model_trainer_config.embed_size,
                hidden_size=self.model_trainer_config.hidden_size,
            )

            model = self.Seq2Seq(encoder, decoder)

            criterion = nn.CrossEntropyLoss(ignore_index=vocab_hi["<pad>"])
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.model_trainer_config.learning_rate
            )

            logging.info("Training loop started")

            for epoch in range(self.model_trainer_config.epochs):
                model.train()
                total_loss = 0

                for batch in data_loader:
                    src_tensor = batch["English"]
                    trg_tensor = batch["Hindi"]

                    optimizer.zero_grad()
                    output = model(src_tensor, trg_tensor)

                    # output: [trg_len - 1, batch_size, output_size]
                    # trg_tensor[1:]: [trg_len - 1, batch_size]
                    loss = criterion(
                        output.view(-1, output.shape[-1]),
                        trg_tensor[1:].reshape(-1),
                    )

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(data_loader)
                logging.info(f"Epoch {epoch+1} completed, Loss: {avg_loss:.4f}")

            # Save the model
            logging.info("Saving trained model")
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )
            torch.save(
                model.state_dict(), self.model_trainer_config.trained_model_file_path
            )

            model_trainer_artifact = ModelTrainerArtifact(
                model_file_path=self.model_trainer_config.trained_model_file_path,
                loss_history=avg_loss
            )

            logging.info("Model training completed successfully")

            return model_trainer_artifact

        except Exception as e:
            logging.exception("Error during model training")
            raise MyException(e, sys)

    async def predict(self, model_trainer_artifact: ModelTrainerArtifact, input_text: str):
        try:
            logging.info("Starting model prediction")

            # Load vocabulary (needed for input size and encoding)
            logging.info("Loading vocabulary for prediction")
            vocabs = await load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )
            vocab_en = vocabs["English"]
            vocab_hi = vocabs["Hindi"]

            input_size_en = len(vocab_en)
            output_size_hi = len(vocab_hi)

            # Reconstruct model architecture
            encoder = self.Encoder(
                input_size=input_size_en,
                embed_size=self.model_trainer_config.embed_size,
                hidden_size=self.model_trainer_config.hidden_size,
            )
            decoder = self.Decoder(
                output_size=output_size_hi,
                embed_size=self.model_trainer_config.embed_size,
                hidden_size=self.model_trainer_config.hidden_size,
            )
            model = self.Seq2Seq(encoder, decoder)

            # Load weights
            model.load_state_dict(torch.load(model_trainer_artifact.model_file_path))
            model.eval()

           
            
            tokens = self.encode_sentence(input_text.lower(), vocab_en)
            input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)
            
            logging.warning("Seq2Seq model currently requires target sequence for teacher forcing logic.")
            
            with torch.no_grad():
                output = "Prediction logic requires Seq2Seq inference mode implementation."
                logging.info(output)

            return output

        except Exception as e:
            logging.exception("Error during model prediction")
            raise MyException(e, sys)
