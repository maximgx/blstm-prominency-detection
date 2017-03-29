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


# Dataset path
PATH = '../corpus/NSYLxALL_X3.csv'

# Features per syllable and its evaluation
COLUMNS = ['nucleus-duration',
            'spectral-emphasis',
            'pitch-movements',
            'overall-intensity',
            'syllable-duration',
            'prominent-syllable']


dataFrame = pandas.read_csv(PATH, delim_whitespace = True,
                            header = None,
                            names = COLUMNS,
                            skip_blank_lines = False)

x_dataset = []
y_dataset = []
x_utterance = []
y_utterance = []

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

print ("... extracted ", total_expressions, "utterances.")
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

numpy.set_printoptions(threshold = sys.maxsize)
print (x_dataset[0])
print (y_dataset[0])
