# Importing the dependencies
import pandas as pd # Dataframing
import numpy as np # matrix operations
from nltk import word_tokenize # word tokenization
# ML
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

# Getting the data
path = '~/practice/train.tsv'
d_raw = pd.read_csv(path, sep='\t')
path = '~/practice/test.tsv'
df_test = pd.read_csv(path, sep='\t')

# Training data
Phrases = d_raw['Phrase']
# we need to find the words and it's indexes and then store the sentiment value
words = []
for i in range(len(Phrases)):
    p = Phrases[i].lower()
    p = word_tokenize(p)
    for word in p:
        words.append(word)
words = sorted(list(set(words)))
# making the word2id dictionary
word2id = dict((c, i) for i,c in enumerate(words))

# training data
input_array = []
for p in Phrases:
    temp = []
    p = word_tokenize(p)
    for word in p:
        temp.append(word2id[word.lower()])
    input_array.append(temp)

# testing data
Phrases = df_test['Phrase']
test_array = []
for p in Phrases:
    temp = []
    p = word_tokenize(p)
    for word in p:
        try:
            temp.append(word2id[word.lower()])
        except:
            word2id.update({word.lower(): len(word2id)})
            temp.append(word2id[word.lower()])
    test_array.append(temp)

# maximum length of any statement, passing through both the datas
max_len = 0
for i in input_array:
    if len(i) > max_len:
        max_len = len(i)
for i in test_array:
    if len(i) > max_len:
        max_len = len(i)

# padding with 0s to make it usable for ML
# For training input
input_final = []
for i in range(len(input_array)):
    t = input_array[i]
    t = t[::-1]
    while len(t) != max_len:
        t.append(0.0)
    t = t[::-1]
    input_final.append(t)
input_final = np.array(input_final).astype(np.float32)
# For test input
test_final = []
for i in range(len(test_array)):
    t = test_array[i]
    t = t[::-1]
    while len(t) != max_len:
        t.append(0.0)
    t = t[::-1]
    test_final.append(t)

# output data
output_sent = d_raw['Sentiment'].astype(np.int32)
output_final = np.zeros(shape = (len(output_sent), 5))
for i in range(len(output_final)):
    output_final[i][output_sent[i]] = 1.0
test_final = np.array(test_final).astype(np.float32)

# doing ML now
model = Sequential()
model.add(Embedding(len(words), 64, input_length = max_len))
model.add(LSTM(20))
model.add(Dense(5, activation = 'softmax'))
print(model.summary())

# compiling and running the model now
model.compile('RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(input_final, output_final, epochs = 5, batch_size = 64, validation_split = 0.2)

# Predicting the values now
op = model.predict(test_final, batch_size = 64)
op = np.reshape(op, [-1, 1])
PhraseId = df_test['PhraseId']
PhraseId = np.reshape(PhraseId, [-1, 1])

final_ = np.array(['PhraseId','Sentiment'])
final_t = np.concatenate([PhraseId, op], axis = 1)
for f in final_t:
	final_.append(f)

# I need to save this file now for submission
sub_file = open('~/practice/submission.txt', 'w')
for item in final_:
	print(str(item[0] + ',' + item[1]), file = sub_file)
sub_file.close()