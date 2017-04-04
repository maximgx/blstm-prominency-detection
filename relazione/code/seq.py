    numpy.random.shuffle(indexes)
    x_dataset = x_dataset[indexes]
    y_dataset = y_dataset[indexes]

    # Building Model
    model = Sequential()

    model.add(Bidirectional(LSTM(max_utterance_length / 2, return_sequences = True),
                            input_shape = (max_utterance_length, len(COLUMNS) - 1)))
    model.add(Dropout(0.5))

    model.add(TimeDistributed(Dense(len(prominent_syllable), activation = 'softmax')))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                  metrics = ['accuracy', 'fmeasure', 'precision', 'recall'])

    model.fit(x_dataset[TRAIN_INDEXES], 
              y_dataset[TRAIN_INDEXES], 
              validation_data = (x_dataset[VALIDATION_INDEXES], 
                                 y_dataset[VALIDATION_INDEXES]),
              nb_epoch = 50, 
              batch_size = 2)

    metrics = model.evaluate(x_dataset[TESTSET_INDEXES], 
                             y_dataset[TESTSET_INDEXES])
