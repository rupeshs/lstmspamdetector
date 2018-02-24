import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model
import keras.preprocessing.text as kpt
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import numpy as np
import json

dictionary = json.load(open('wordindex.json'))

def convert_text_to_index_array(text):
     # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    wordvec=[]
    for word in kpt.text_to_word_sequence(text) :
        if word in dictionary:
            wordvec.append([dictionary[word]])
        else:
            wordvec.append([0])
    
    return wordvec


X_test=[]
data= pd.read_csv("test.csv")

data = data[['MessageType','Message']]
data['Message'] = data['Message'].apply(lambda x: x.lower())

for sentence in data['Message'].values:
    word_vec=convert_text_to_index_array(sentence)
    X_test.append(word_vec)


X_test = pad_sequences(X_test, maxlen=654)
Y_test = pd.get_dummies(data['MessageType']).values

model = load_model('spam_lstm_model.h5')
print(model.summary())

ham_cnt, spam_cnt, spam_correct, ham_correct = 0, 0, 0, 0
count=0
for x in range(len(X_test)):

     result  = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
     count=count+1 
     if np.argmax(result) == np.argmax(Y_test[x]):
         if np.argmax(Y_test[x]) == 0:
             ham_correct += 1
         else:
             spam_correct += 1
     if np.argmax(Y_test[x]) == 0:
         ham_cnt += 1
     else:
        spam_cnt += 1
    
#43 spam 57 ham
#0 ham 1 spam
if spam_cnt>0:
    print("Spam acc ->", spam_correct/spam_cnt*100, "%")
if ham_cnt>0: 
    print("Ham acc ->", ham_correct/ham_cnt*100, "%")

