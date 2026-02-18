import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import pickle
import os


class Create:
    def __init__(self,data,inputCol,outputCol,epochs,batch_size):
        self.data=data
        self.inputCol=inputCol
        self.outputCol=outputCol
        self.epochs=epochs
        self.batch_size=batch_size

    
    @ staticmethod
    def prep_text(s): 
        return " ".join(s.strip().lower().split())
    def dataClean(self):
        print("Cleaning the data")
        filters = '"!#$%&()*+,-./:;=?@[\\]^`{|}~\t\n'
        self.data[self.inputCol] = self.data[self.inputCol].apply(Create.prep_text)
        self.data[self.outputCol]  = self.data[self.outputCol].apply(lambda s: f"start_ {Create.prep_text(s)} _end")
         
        self.tokenizer_e = Tokenizer(filters=filters, lower=True, oov_token=None)
        self.tokenizer_f = Tokenizer(filters=filters, lower=True, oov_token=None)


        # creating Two Tokenizers
        self.tokenizer_e.fit_on_texts(self.data[self.inputCol])
        self.tokenizer_f.fit_on_texts(self.data[self.outputCol])

                
        # creating source and target vectors 
        self.src_seq = self.tokenizer_e.texts_to_sequences(self.data[self.inputCol])
        self.tgt_seq = self.tokenizer_f.texts_to_sequences(self.data[self.outputCol])

        # storing the max length of the sequences
        self.max_len_src = max(len(s) for s in self.src_seq)
        self.max_len_tgt = max(len(s) for s in self.tgt_seq)


        # applying post padding to it
        self.src_seq = pad_sequences(self.src_seq, maxlen=self.max_len_src, padding='post')
        self.tgt_seq = pad_sequences(self.tgt_seq, maxlen=self.max_len_tgt, padding='post')


        self.tgt_input  = self.tgt_seq[:, :-1]            # encoder input _end ko hate hue up to last -1
        self.tgt_output = self.tgt_seq[:, 1:]             # start_ ko hatate hue upto _end

        self.vocab_src = len(self.tokenizer_e.word_index) + 1 # kyoki 1 se start hota h isiliye + 1 numbers of vocab size
        self.vocab_tgt = len(self.tokenizer_f.word_index) + 1 # kyoki 1 se start hota h isiliye + 1 numbers of vocab size


    def compileModel(self):
        print("Compiling the model")
        # Model

        latent_dim = 256 # lstm nodes 

        # encoder
        enc_inputs = Input(shape=(self.max_len_src,))
        enc_emb = Embedding(self.vocab_src, 128, mask_zero=True)(enc_inputs)
        _, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)


        enc_states = [state_h, state_c] # context vector


        # decoder

        dec_inputs = Input(shape=(self.max_len_tgt-1,)) # kyoki ek kam de rahe h na ham isiliye -1
        dec_emb = Embedding(self.vocab_tgt, 128, mask_zero=True)(dec_inputs)
        dec_outputs = LSTM(latent_dim, return_sequences=True, return_state=False)(dec_emb, initial_state=enc_states)


        dec_logits = Dense(self.vocab_tgt, activation='softmax')(dec_outputs) # har output par ek probability return hogi har vocab ke liye 

        # creating model

        self.model = Model([enc_inputs, dec_inputs], dec_logits)

        # compiling it it is mandatory to use sparse_categorical_crossentropy as jo next word h wo categorical h regration nahi 
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


    def fitModel(self):
        print("Training the model")
                
        # targets must be (batch, timesteps, 1) for sparse loss
        y_sparse = np.expand_dims(self.tgt_output, -1)


        # training the model
        self.model.fit([self.src_seq, self.tgt_input], y_sparse, batch_size=self.batch_size, epochs=self.epochs, verbose=1)    

    


    def saveModel(self,Modelname,InputTokenizer,OutputTokenizer,Input_max_len_src,Output_max_len_tgt):
        print("Saving the model")
        folder = Modelname
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.model.save(f'{Modelname}/{Modelname}.keras')
        with open(f'{Modelname}/{InputTokenizer}.pkl','wb') as f:
            pickle.dump(self.tokenizer_e,f)

        with open(f'{Modelname}/{OutputTokenizer}.pkl','wb') as f:
            pickle.dump(self.tokenizer_f,f) 

        with open(f'{Modelname}/{Input_max_len_src}.pkl','wb') as f:
            pickle.dump(self.max_len_src,f)

        with open(f'{Modelname}/{Output_max_len_tgt}.pkl','wb') as f:
            pickle.dump(self.max_len_tgt,f)       


def CreateModel(data,firstCol,SecondCol,name,InputTokenizer,OutputTokenizer,Input_max_len_src,Output_max_len_tgt,epochs=30,batch_size=64):
    model=Create(data,firstCol,SecondCol,epochs,batch_size)
    model.dataClean()
    model.compileModel()
    model.fitModel()
    model.saveModel(name,InputTokenizer,OutputTokenizer,Input_max_len_src,Output_max_len_tgt)

