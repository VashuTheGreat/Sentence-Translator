from src.constants import DEVICE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List
from collections import Counter

def clean_text(text: str, lang: str = 'english') -> str:
    if not text:
        return ""
    # lowercase
    text = text.lower()
    # tokenize
    tokens = word_tokenize(text)
    # get stopword list for language
    stop_words = set(stopwords.words(lang))
    # filter stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # join back to string
    return " ".join(tokens)

def build_vocabs(sentences: List[str]):
    vocab = Counter(' '.join(sentences).split())
    vocab = {k: i + 3 for i, (k, v) in enumerate(vocab.items())}
    vocab['<pad>'] = 0
    vocab['<pos>'] = 1
    vocab['<eos>'] = 2
    return vocab

def sent_tokens(sentence: str, vocab):
    tokens = [vocab.get('<pos>', 1)]
    for w in sentence.split():
        if w in vocab:
            tokens.append(vocab[w])
        elif '<unk>' in vocab:
            tokens.append(vocab['<unk>'])
        # If no <unk> and not in vocab, we skip it to avoid KeyError
    tokens.append(vocab.get('<eos>', 2))
    return tokens