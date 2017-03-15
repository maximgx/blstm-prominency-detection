from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.preprocessing import sequence
from keras import backend as K

import pandas
import numpy
import sys

# from itertools import repeat, chain, islice


# Path per i relativi dataset
PATHS = ['../corpus/train.csv',
            '../corpus/validation.csv',
            '../corpus/test.csv']

# Da rivedere l'ordine delle colonne
COLUMNS = ['nucleus-duration',
            'spectral-emphasis',
            'pitch-movements',
            'overall-intensity',
            'syllable-duration',
            'prominent-syllable']


dataset = []
max_utterance_length = 0

for path in PATHS:
    total_rows = len(pandas.read_csv(path, skip_blank_lines = False)) + 1
    reader = pandas.read_csv(path, delim_whitespace = True,
                                header = None,
                                names = COLUMNS,
                                skip_blank_lines = False,
                                chunksize = 1)
    print ("Extracting", path)

    x_utterance = []
    y_utterance = []
    x = []
    y = []
    chunks = 0

    for chunk in reader:
        # Estrazione del vettore delle features di ogni sillaba
        x_syllable = chunk.loc[:, 'nucleus-duration':'syllable-duration'].values[0]
        chunks += 1

        # Se il vettore contiene NaN allora la frase è finita
        if not(numpy.isnan(x_syllable[0])) and chunks < total_rows:
            x_utterance.append(x_syllable)
            y_syllable = chunk.loc[:, 'prominent-syllable'].values[0]
            y_utterance.append(y_syllable)
        else:
            # Appendi le features e le prominenze della frase al dataset
            x.append(x_utterance)
            y.append(y_utterance)
            if max_utterance_length < len(x_utterance):
                max_utterance_length = len(x_utterance)

            # print ("Features estratte: ", x_utterance)
            # print ("Sillabe prominenti:", y_utterance, "\n")

            x_utterance = []
            y_utterance = []

    dataset.append([x, y])
    print ("... extracted ", len(x), "utterances.")

print ("\nLongest expression with", max_utterance_length, "syllables.",
        "Filling shorter expressions with zeroes...")

# Tutte le espressioni avranno la lunghezza dell'espressione più lunga
for i in range(len(dataset)):
    for j in range(len(dataset[i])):
        dataset[i][j] = sequence.pad_sequences(dataset[i][j],
                                                maxlen = max_utterance_length,
                                                dtype = 'float', padding = 'post',
                                                truncating = 'post', value = 0.)

model = Sequential() # return_sequence = True
model.add(Masking(mask_value = 0, input_shape = (max_utterance_length, len(COLUMNS) - 1)))
model.add(LSTM(max_utterance_length))
                #input_shape = (max_utterance_length, len(COLUMNS) - 1)))
model.add(Dense(max_utterance_length, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print (model.summary())
numpy.set_printoptions(threshold = sys.maxsize)

model.fit(dataset[0][0], dataset[0][1],
            validation_data = (dataset[1][0], dataset[1][1]),
            nb_epoch = 20, batch_size = 5)
scores = model.evaluate(dataset[2][0], dataset[2][1], verbose = 1)
print ("Accuracy: %.2f%%" % (scores[1]*100))

# model = Sequential() # return_sequence = True
# model.add(Masking(mask_value = 0, input_shape = (max_utterance_length, len(COLUMNS) - 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(max_utterance_length))
# model.add(Dropout(0.2))
# model.add(Dense(max_utterance_length, activation = 'sigmoid'))
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# print (model.summary())
#
# model.fit(dataset[0][0], dataset[0][1],
#             nb_epoch = 500, batch_size = 5)
# score, acc = model.evaluate(dataset[1][0], dataset[1][1], verbose = 1)
# #print ("Accuracy: %.2f%%" % (scores[1]*100))
# print('Test score:', score)
# print('Test accuracy:', acc)
#
#
# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([dataset[2][0]])[0]
numpy.set_printoptions(threshold = sys.maxsize)
print (dataset[2][0][0])
print (dataset[2][1][0])
print (layer_output[0])
