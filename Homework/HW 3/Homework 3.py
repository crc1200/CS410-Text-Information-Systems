# multivariate lstm code based on Brownlee (2020)
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequences)):
    # find the end of this pattern
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out

    # check if we are beyond the dataset
    if out_end_ix > len(sequences):
      break

  # gather input and output parts of the pattern
  seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
  X.append(seq_x)
  y.append(seq_y)
  return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
in_seq3 = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, in_seq3))

# choose a number of time steps
n_steps_in, n_steps_out = 3, 2

# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss ='mse')

# fit model
model.fit(X, y, epochs=300, verbose=0)

# demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

###

from transformers import pipeline
model = pipeline('fill-mask', model='bert-base-uncased')
pred = model("Forecasts of Presidential [MASK]")
print("Predicted next word: ")
pred[0]['token_str']

###

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model2 = AutoModelForCausalLM.from_pretrained("gpt2")

seq = "Forecasts of Presidential"

inputs = tokenizer(seq, return_tensors="pt")

input_ids = inputs["input_ids"]
for id in input_ids[0]:
  word = tokenizer.decode(id)

with torch.no_grad():
  logits = model2(**inputs).logits[:, -1, :]

pred_id = torch.argmax(logits).item()

pred_word = tokenizer.decode(pred_id)
print("Predicted next word: ")
print(pred_word)

###

# Create the object for LDA model
lda1 = gensim.models.ldamodel.LdaModel
# Train the LDA model using the document term matrix.
ldamodel = lda1(matrix_of_doc_term, num_topics=10, id2word=D1, passes=100)
