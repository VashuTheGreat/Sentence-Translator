import torch
from src.utils.asyncHandler import asyncHandler
from src.utils.model_utils import sent_tokens, clean_text
from src.logger import logging
from src.constants import DEVICE

class Prediction:
    def __init__(self, model, en_vocab, hi_vocab, device=DEVICE):
        self.model = model
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.device = device

    @asyncHandler
    async def predict(self, sentence: str, max_len=50):
        logging.info(f"Prediction component started for: {sentence}")
        self.model.eval()

        # Clean text as done in notebook before training
        cleaned_sentence = clean_text(sentence)

        hi_idx_to_word = {v: k for k, v in self.hi_vocab.items()}
        tokenized_en_sent = sent_tokens(cleaned_sentence, self.en_vocab)
        src_tensor = torch.tensor(tokenized_en_sent, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoder_hidden = self.model.encoder(src_tensor)
            decoder_input = torch.tensor([self.hi_vocab['<pos>']], dtype=torch.long).to(self.device)
            predicted_hi_tokens = []

            for _ in range(max_len):
                output, encoder_hidden = self.model.decoder(decoder_input, encoder_hidden)
                predicted_token_id = output.argmax(1).item()
                predicted_hi_tokens.append(predicted_token_id)

                if predicted_token_id == self.hi_vocab['<eos>']:
                    break

                decoder_input = torch.tensor([predicted_token_id], dtype=torch.long).to(self.device)

        if predicted_hi_tokens and predicted_hi_tokens[0] == self.hi_vocab['<pos>']:
            predicted_hi_tokens = predicted_hi_tokens[1:]
        if predicted_hi_tokens and predicted_hi_tokens[-1] == self.hi_vocab['<eos>']:
            predicted_hi_tokens = predicted_hi_tokens[:-1]

        predicted_sentence = ' '.join([hi_idx_to_word[token_id] for token_id in predicted_hi_tokens if token_id in hi_idx_to_word])
        logging.info(f"Prediction component result: {predicted_sentence}")
        return predicted_sentence

