'''
    Prosodic prominence detection in Italian continuous speech using BLSTMs

    Coded by: Maxim Gaina, maxim.gaina@yandex.ru
    Datasets provided by: Fabio Tamburini, fabio.tamburini@unibo.it
'''

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge, Masking, Reshape
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import backend as K

import pandas
import numpy
import sys

#import argparse
#
# argparser = argparse.ArgumentParser(description =
#             'Extracts data from corpus and feeds the BLSTM, creating then a model.')
# argparser.add_argument('--save')

# Dataset paths
PATHS = ['../corpus/train.csv',
            '../corpus/validation.csv',
            '../corpus/test.csv']

# Features per syllable and its evaluation
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
        # Extracting feature vector per each syllable
        x_syllable = chunk.loc[:, 'nucleus-duration':'syllable-duration'].values[0]
        chunks += 1

        # If vector contiains NaN then the expression is finished
        if not(numpy.isnan(x_syllable[0])) and chunks < total_rows:
            x_utterance.append(x_syllable)
            if chunk.loc[:, 'prominent-syllable'].values[0] == 0:
                y_syllable = [1, 0, 0]
            else:
                y_syllable = [0, 1, 0]
            # y_syllable = chunk.loc[:, 'prominent-syllable'].values[0]
            y_utterance.append(y_syllable)
            # y_utterance.append(numpy.full(1, y_syllable))
        else:
            # Append expression's features and evaluations at dataset
            if path == PATHS[-1]:
                x_utterance[-1] = numpy.full(5, 0, dtype = 'float64')
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

for i in range(len(dataset)):
    for j in range(len(dataset[i])):
        if j == 0:
            dataset[i][j] = sequence.pad_sequences(dataset[i][j],
                                                maxlen = max_utterance_length,
                                                dtype = 'float', padding = 'post',
                                                truncating = 'post', value = 0.)
        else:
            dataset[i][j] = sequence.pad_sequences(dataset[i][j],
                                                maxlen = max_utterance_length,
                                                dtype = 'float', padding = 'post',
                                                truncating = 'post', value = [0., 0., 1.])

# print (dataset[2][0][0])
# print (dataset[2][1][0])

dataset = numpy.asarray(dataset)

'''
epochs      100     200     300     500
Accuracy:   90.74   91.13   90.93   91.13
F1:         86.07   86.68   86.39   86.68
Precision:  86.26   86.75   86.54   86.75
BEST
epochs: 50
Accuracy: 91.57%
F1: 87.32%
Precision: 87.57%
Recall: 87.06%
'''
model = Sequential()
# 17 memory cells seems to be the best choice
model.add(Bidirectional(LSTM(17, return_sequences = True),
                        input_shape = (max_utterance_length, len(COLUMNS) - 1)))
model.add(Dropout(0.5))
# Gives a sequence of triples (one triple per each syllable) where each element
# represents the probability of being prominent, non prominent and missing at all.
# Indeed, all utterances must have the same length and some of them where
# padded with vectors of zeroes.
model.add(TimeDistributed(Dense(len(dataset[0][1][0][0]), activation = 'softmax')))


model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                metrics = ['accuracy', 'fmeasure', 'precision', 'recall'])
print (model.summary())

# No known nb_epoch and batch_size gives better metrics than this
model.fit(dataset[0][0], dataset[0][1],
            validation_data = (dataset[1][0], dataset[1][1]),
            nb_epoch = 50, batch_size = 2)


scores = model.evaluate(dataset[2][0], dataset[2][1], verbose = 1)
print ("Accuracy: %.2f%%" % (scores[1]*100))
print ("F1: %.2f%%" % (scores[2]*100))
print ("Precision: %.2f%%" % (scores[3]*100))
print ("Recall: %.2f%%" % (scores[4]*100))

model_filename = 'blstm-prominence.h5'
print ("Saving model to file...", model_filename)
model.save(model_filename)

''' See intermediate layer output '''
# get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                     [model.layers[2].output])
# layer_output = get_layer_output([dataset[2][0], 0])[0]
# numpy.set_printoptions(threshold = sys.maxsize)
# print (layer_output.shape)
# print (layer_output[3])
# print (dataset[2][0][3])
# print (dataset[2][1][3])
