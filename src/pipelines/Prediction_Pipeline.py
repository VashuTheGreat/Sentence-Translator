import sys
from src.entity.estimator import MyModel, Encoder, Decoder, seq2seq
from src.constants import model_path, en_vocab_path, hi_vocab_path, DEVICE
import torch
from src.logger import logging
from src.utils.asyncHandler import asyncHandler

class PredictionPipeline:
    def __init__(self):
        try:
            logging.info("Initializing PredictionPipeline and loading artifacts")
            
            en_vocab = torch.load(en_vocab_path, weights_only=False)
            hi_vocab = torch.load(hi_vocab_path, weights_only=False)
            
            # reconstructing model archtecture
            input_size_en = len(en_vocab)
            output_size_hi = len(hi_vocab)
            embed_size = 256
            hidden_size = 512
            
            encoder = Encoder(input_size_en, embed_size, hidden_size)
            decoder = Decoder(output_size_hi, embed_size, hidden_size)
            model_instance = seq2seq(encoder, decoder).to(DEVICE)
            
            # loading model in the architecture
            model_instance.load_state_dict(torch.load(model_path, map_location=DEVICE))
            
            self.model = MyModel(hi_vocab, en_vocab, model_instance)
            logging.info("PredictionPipeline initialized successfully")
            
        except Exception as e:
            logging.exception("Failed to initialize PredictionPipeline")
            from src.exception import MyException
            raise MyException(e, sys)

    @asyncHandler
    async def predict(self, sentence: str):
        logging.info(f"Received prediction request for: {sentence}")
        return await self.model.predict(sentence)

    @asyncHandler
    async def initiate_prediction_pipeline(self, sentence: str):
        return await self.predict(sentence)