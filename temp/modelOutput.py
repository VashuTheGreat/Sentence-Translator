import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model, load_model
import pickle



def load_model_files(modelToLoad):
    global tokenizer_e, tokenizer_f, max_len_tgt, max_len_src, model, START_ID, END_ID, index2word_f, word2index_f
    
    with open(f'{modelToLoad}/{modelToLoad}_it.pkl','rb') as f:
        tokenizer_e = pickle.load(f)

    with open(f'{modelToLoad}/{modelToLoad}_ot.pkl','rb') as f:
        tokenizer_f = pickle.load(f)     

    with open(f'{modelToLoad}/{modelToLoad}_omlt.pkl','rb') as f:
        max_len_tgt = pickle.load(f)       

    with open(f'{modelToLoad}/{modelToLoad}_imls.pkl','rb') as f:
        max_len_src = pickle.load(f)  

    model = load_model(f'{modelToLoad}/{modelToLoad}.keras')  

    # Set word mappings
    index2word_f = tokenizer_f.index_word
    word2index_f = tokenizer_f.word_index
    START_ID = word2index_f.get("start_")
    END_ID   = word2index_f.get("_end")

def prep_text(s): 
    return " ".join(s.strip().lower().split())

def translate(en_text, max_len=50):
    global START_ID, END_ID, index2word_f
    en_text = prep_text(en_text)
    x1 = tokenizer_e.texts_to_sequences([en_text])
    x1 = pad_sequences(x1, maxlen=max_len_src, padding='post')
    dec = [START_ID]
    for _ in range(min(max_len, max_len_tgt-1)):
        x2 = pad_sequences([dec], maxlen=max_len_tgt-1, padding='post')
        p  = model.predict([x1, x2], verbose=0)
        next_id = int(np.argmax(p[0, len(dec)-1, :]))
        if next_id == 0: break
        if next_id == END_ID: break
        dec.append(next_id)
    words = [index2word_f.get(i, "") for i in dec[1:]]
    return " ".join([w for w in words if w])

# Load model files
load_model_files('Hindi')


