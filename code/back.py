from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2

# PRIMA PROVA, 10k iter TimeDistributed. Accuracy: 87.47% F1: 45.24%
# SECONDA PROVA, 10k iter LSTM + TimeDistributed.
# TERZA PROVA, 10k iter Masking + LSTM + TimeDistributed.
# TERZA PROVA, 10k iter LSTM + TimeDistributed + Dense.

model = Sequential()
# Questo layer peggiora l'accuratezza in media dello 0.3%

#model.add(Masking(mask_value = 0, input_shape = (max_utterance_length, len(COLUMNS) - 1)))
#model.add(Bidirectional(LSTM(17, return_sequences = True), input_shape = (max_utterance_length, len(COLUMNS) - 1)))
#model.add(Bidirectional(LSTM(512)))
#model.add(Dropout(0.5))
#1model.add(TimeDistributed(Dense(5, activation = 'softmax')))
#model.add(Dropout(0.5))
#model.add(Dropout(0.5))
#model.add(LSTM(max_utterance_length))
# model.add(Dropout(0.5))


# 8747
model.add(Bidirectional(LSTM(17, return_sequences = True), input_shape = (max_utterance_length, len(COLUMNS) - 1)))
model.add(TimeDistributed(Dense(5)))
# 8549
model.add(Dense(5, activation = 'sigmoid'))




# 8951
model.add(Bidirectional(LSTM(17, return_sequences = True), input_shape = (max_utterance_length, len(COLUMNS) - 1)))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(5)))
model.add(Dense(5, activation = 'sigmoid'))
