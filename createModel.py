import pandas as pd
from modelCreator import CreateModel


# data = pd.read_csv('fra.txt', sep='\t', header=None, names=['English', 'Hindi'])
data = pd.read_csv('data/Dataset_English_Hindi.csv')


data.dropna(inplace=True)
data=data[:141]

CreateModel(data=data,firstCol='English',SecondCol='Hindi',name='Hindi',InputTokenizer='Hindi_it',OutputTokenizer='Hindi_ot',Input_max_len_src='Hindi_imls',Output_max_len_tgt='Hindi_omlt',epochs=100,batch_size=2)