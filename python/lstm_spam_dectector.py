import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Dropout,Bidirectional
from keras.callbacks import TensorBoard
from sklearn.cross_validation import train_test_split
import re
import numpy as np
import os
import json

#Read train data
data= pd.read_csv("train.csv")
data = data[['MessageType','Message']]
data['Message'] = data['Message'].apply(lambda x: x.lower())

#top 2000 words
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures)
tokenizer.fit_on_texts(data['Message'].values )

dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('wordindex.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

X = tokenizer.texts_to_sequences(data['Message'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['MessageType']).values

# LSTM model
embed_dim = 128
lstm_out = 64

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(Bidirectional(LSTM(lstm_out)))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

batch_size = 30
#tbCallBack =TensorBoard(log_dir='G:/tensorflow/text/SpamDetectionData/logs', histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size)

validation_size = 100
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print(model.metrics_names)
print("Validation loss: %f" % (score))
print("Validation acc: %f" % (acc))
model.save('spam_lstm_model.h5')

res=model.predict(X_test[10].reshape(1,X_test.shape[1]))[0]
print(str(np.argmax(Y_test[10])))

ham_cnt, spam_cnt, spam_correct, ham_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
     result = model.predict(X_validate[x].reshape(1,X_validate.shape[1]),batch_size=1,verbose = 2)[0]
     #print(str(np.argmax(result)))
     if np.argmax(result) == np.argmax(Y_validate[x]):
         if np.argmax(Y_validate[x]) == 0:
             ham_correct += 1
         else:
             spam_correct += 1
     if np.argmax(Y_validate[x]) == 0:
         ham_cnt += 1
     else:
         spam_cnt += 1
    

print("Spam acc ->", spam_correct/spam_cnt*100, "%")
print("Ham acc ->", ham_correct/ham_cnt*100, "%")
print(str(X.shape[1]))
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
