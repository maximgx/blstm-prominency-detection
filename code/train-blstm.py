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
from random import shuffle

import pandas
import numpy
import sys


# Dataset path
PATH = '../corpus/NSYLxALL_X3.csv'

# Features per syllable and its evaluation
COLUMNS = ['nucleus-duration',
            'spectral-emphasis',
            'pitch-movements',
            'overall-intensity',
            'syllable-duration',
            'prominent-syllable']

LEARNING_PHASES = 20

#TRAINSET_LEGTH = 85; VALIDATIONSET_LENGHT = 15; TESTSET_LENGTH = 19
TRAIN_INDEXES = numpy.arange(85)
VALIDATION_INDEXES = numpy.arange(85, 100)
TESTSET_INDEXES = numpy.arange(100, 119)

'''
            ERROR       NO ERROR (new)
Accuracy    91.04%      91.17%
Precision   86.59%      86.77%
Recall      86.51%      86.73%
F1:         86.55%      86.75%
'''

dataFrame = pandas.read_csv(PATH, delim_whitespace = True,
                            header = None,
                            names = COLUMNS,
                            skip_blank_lines = False)

x_dataset = []; y_dataset = []
x_utterance = []; y_utterance = []

prominent_syllable = [0., 1., 0.]
not_prominent_syllable = [1., 0., 0.]
missing_features = [0., 0., 0., 0., 0.]
missing_syllable = [0., 0., 1.]

max_utterance_length = 0

print ("\nExtracting utterances from", PATH)

for index, row in dataFrame.iterrows():
    # Extracting feature vector per syllable and its prominence evaluation
    features = row['nucleus-duration':'syllable-duration'].values
    prominence = row['prominent-syllable']

    # If vector contiains NaN then the expression is finished
    if numpy.isnan(prominence):
        if max_utterance_length < len(x_utterance):
            max_utterance_length = len(x_utterance)

        # Append expression to dataset
        x_dataset.append(x_utterance)
        y_dataset.append(y_utterance)

        x_utterance = []
        y_utterance = []
    else:
        if prominence == 0:
            prominence = not_prominent_syllable
        else:
            prominence = prominent_syllable
        # Append syllable features to current expression
        x_utterance.append(features)
        y_utterance.append(prominence)

total_expressions = len(y_dataset)

print ("... extracted ", total_expressions, "utterances. Dataset split:")
print ("\tTraining set", len(TRAIN_INDEXES), "expressions")
print ("\tValidation set", len(VALIDATION_INDEXES), "expressions")
print ("\tTest set", len(TESTSET_INDEXES), "expressions")
print ("\nLongest expression composed by", max_utterance_length, "syllables.")

print ("Filling shorter expressions with zeroes...")
x_dataset = sequence.pad_sequences(x_dataset,
                                maxlen = max_utterance_length,
                                dtype = 'float',
                                padding = 'post',
                                truncating = 'post',
                                value = missing_features)
y_dataset = sequence.pad_sequences(y_dataset,
                                maxlen = max_utterance_length,
                                dtype = 'float',
                                padding = 'post',
                                truncating = 'post',
                                value = missing_syllable)

x_dataset = numpy.asarray(x_dataset)
y_dataset = numpy.asarray(y_dataset)


indexes = numpy.arange(total_expressions)
scores = []

numpy.set_printoptions(threshold = sys.maxsize)

for learning_phase in range(LEARNING_PHASES):
    numpy.random.shuffle(indexes)
    x_dataset = x_dataset[indexes]
    y_dataset = y_dataset[indexes]

    print ("Dataset shuffled")

    # Building Model
    model = Sequential()
    # 17 memory cells seems to be the best choice
    model.add(Bidirectional(LSTM(17, return_sequences = True),
                            input_shape = (max_utterance_length, len(COLUMNS) - 1)))
    model.add(Dropout(0.5))
    # Gives a sequence of triples (one triple per each syllable) where each element
    # represents the probability of being prominent, non prominent and missing at all.
    # Indeed, all utterances must have the same length and some of them where
    # padded with vectors of zeroes.
    model.add(TimeDistributed(Dense(len(prominent_syllable), activation = 'softmax')))

    model.compile(loss = 'binary_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy', 'fmeasure', 'precision', 'recall'])
    print (model.summary())

    # No known nb_epoch and batch_size gives better metrics than this
    model.fit(x_dataset[TRAIN_INDEXES], y_dataset[TRAIN_INDEXES],
                validation_data = (x_dataset[VALIDATION_INDEXES], y_dataset[VALIDATION_INDEXES]),
                nb_epoch = 50, batch_size = 2)

    metrics = model.evaluate(x_dataset[TESTSET_INDEXES],
                            y_dataset[TESTSET_INDEXES],
                            verbose = 1)

    print ("Learning phase", learning_phase + 1, "scores:")
    print ("\tAccuracy %.2f%%" % (metrics[1] * 100))
    print ("\tPrecision %.2f%%" % (metrics[3] * 100))
    print ("\tRecall %.2f%%" % (metrics[4] * 100))
    print ("\tF1: %.2f%%" % (metrics[2] * 100))

    scores.append(metrics)

    del model

overall_scores = numpy.mean(scores, axis = 0)
print ("Overall scores:")
print ("\tAccuracy %.2f%%" % (overall_scores[1] * 100))
print ("\tPrecision %.2f%%" % (overall_scores[3] * 100))
print ("\tRecall %.2f%%" % (overall_scores[4] * 100))
print ("\tF1: %.2f%%" % (overall_scores[2] * 100))

numpy.set_printoptions(threshold = sys.maxsize)
print (x_dataset[0])
print (y_dataset[0])

model_filename = 'blstm-prominence.h5'
print ("Saving model to file...", model_filename)
model.save(model_filename)
