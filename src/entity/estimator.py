import torch
import torch.nn as nn
from src.utils.asyncHandler import asyncHandler
from src.utils.model_utils import sent_tokens, clean_text
from src.logger import logging
from src.constants import DEVICE

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedding = self.embedding(x)
        outputs, hidden = self.rnn(embedding)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.ff = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x).unsqueeze(1)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.ff(output.squeeze(1))
        return prediction, hidden

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.ff.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(trg.device)
        encoder_hidden = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, encoder_hidden = self.decoder(input, encoder_hidden)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs[:, 1:, :]

class MyModel:
    def __init__(self, hi_vocab, en_vocab, trained_model_object):
        self.hi_vocab = hi_vocab
        self.en_vocab = en_vocab
        self.trained_model_object = trained_model_object

    @asyncHandler
    async def predict(self, sentence: str, max_len=50, device=DEVICE):
        logging.info(f"Starting prediction for sentence: {sentence}")
        self.trained_model_object.eval()
        
        cleaned_sentence = clean_text(sentence)
        
        hi_idx_to_word = {v: k for k, v in self.hi_vocab.items()}
        tokenized_en_sent = sent_tokens(cleaned_sentence, self.en_vocab)
        src_tensor = torch.tensor(tokenized_en_sent, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            encoder_hidden = self.trained_model_object.encoder(src_tensor)
            decoder_input = torch.tensor([self.hi_vocab['<pos>']], dtype=torch.long).to(device)
            predicted_hi_tokens = []

            for _ in range(max_len):
                output, encoder_hidden = self.trained_model_object.decoder(decoder_input, encoder_hidden)
                predicted_token_id = output.argmax(1).item()
                predicted_hi_tokens.append(predicted_token_id)
                if predicted_token_id == self.hi_vocab['<eos>']:
                    break
                decoder_input = torch.tensor([predicted_token_id], dtype=torch.long).to(device)

        if predicted_hi_tokens and predicted_hi_tokens[0] == self.hi_vocab['<pos>']:
            predicted_hi_tokens = predicted_hi_tokens[1:]
        if predicted_hi_tokens and predicted_hi_tokens[-1] == self.hi_vocab['<eos>']:
            predicted_hi_tokens = predicted_hi_tokens[:-1]

        predicted_sentence = ' '.join([hi_idx_to_word[token_id] for token_id in predicted_hi_tokens if token_id in hi_idx_to_word])
        logging.info(f"Prediction completed. Result: {predicted_sentence}")
        return predicted_sentence


          
          
              
      