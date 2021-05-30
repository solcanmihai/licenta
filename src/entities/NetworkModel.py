import pickle
from math import floor
from typing import List

import keras
import scipy.io.wavfile as wav
import numpy as np
import sklearn
import speechpy
import os
import pydub
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.utils import to_categorical
import tensorflow as tf
from scipy.spatial import distance
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay

from src.entities.Speaker import Utterance, Speaker


class NetworkModel:
    def __init__(self):
        self.model = None

    def __init__(self, model_name, root_directory):
        model_path = os.path.join(root_directory, '..', 'trained_models', f'{model_name}.h5')
        self.model = keras.models.load_model(model_path)

    def train_model(self, speakers: List[Speaker], filterbank_nr: int, context_left_size: int, context_right_size: int):
        training_generator = DataGenerator(speakers)
        nr_hidden = 4
        nr_dropout = 2
        dropout = 0.4
        learning_rate = 0.02
        epochs = 100

        hidden_layers = Sequential([
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(256, activation='relu'),
        ])

        input_model = Input(shape=((context_left_size + context_right_size + 1) * filterbank_nr))
        coefficient_extractor = hidden_layers(input_model)

        training_model = Model(inputs=input_model, outputs=coefficient_extractor)

        # training_output_layer = Dropout(0.3)(coefficient_extractor)
        training_output_layer = Dense(len(speakers), 'softmax')(coefficient_extractor)

        model = Model(inputs=input_model, outputs=training_output_layer)

        # x_data, y_data = self.build_training_data()
        # filenames_shuffled, y_labels_shuffled = shuffle(x_data, y_data)
        # X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(np.array(filenames_shuffled),
        #                                                                       y_labels_shuffled, test_size=0.01,
        #                                                                       random_state=1)

        print(model.summary())

        sgd = SGD(lr=learning_rate)
        early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        # early_stopping_2 = EarlyStopping(monitor='loss', mode='min', baseline=0.05)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=2, min_lr=0.0005)
        # reduce_lr_exponential = ExponentialDecay(initial_learning_rate=0.3, decay_rate=0.1, decay_steps=5000000)

        csv_logger = CSVLogger('training.log')
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(training_generator, epochs=epochs, verbose=1, callbacks=[csv_logger, early_stopping, reduce_lr])

        os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'trained_models'))
        training_model.save(f'training_{epochs}_{learning_rate}.h5')
        self.model = training_model


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_speakers: List[Speaker], batch_size=1024, nr_mfcc=14, context_left=30, context_right=10):
        self.nr_context = context_left + context_right + 1
        self.dim = (nr_mfcc * self.nr_context,)
        self.batch_size = batch_size
        self.nr_mfcc = nr_mfcc

        self.list_speakers = list_speakers
        # self.total_frames_with_context = 0

        # for speaker in self.list_speakers:
        #     for utterance in speaker.utterances:
        #         self.total_frames_with_context += len(utterance.mfcc_frames) / nr_mfcc - self.nr_context + 1
        self.speaker = 0
        self.frame = context_left
        self.list_ids = []
        self.indexes = None

        for speaker_index, speaker in enumerate(list_speakers):
            for utterance_index, utterance in enumerate(speaker.utterances):
                nr = int(len(utterance.mfcc_frames) / nr_mfcc)
                for frame_start_index in range(context_left, nr - context_right):
                    slice_start = (frame_start_index - context_left) * nr_mfcc
                    slice_end = (frame_start_index + context_right + 1) * nr_mfcc
                    self.list_ids.append((speaker_index, utterance_index, slice_start, slice_end))

        if len(self.list_ids) < batch_size:
            self.batch_size = len(self.list_ids)

        print(len(self.list_ids))

        self.on_epoch_end()

    def __len__(self):
        return int(floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        self.indexes = shuffle(self.list_ids)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.nr_context * self.nr_mfcc))
        y = np.empty(self.batch_size, dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            (speaker_index, utterance_index, slice_start, slice_end) = ID
            X[i, ] = np.array(self.list_speakers[speaker_index].utterances[utterance_index].mfcc_frames[slice_start: slice_end])

            y[i] = self.list_speakers[speaker_index].id

        return X, keras.utils.to_categorical(y, num_classes=len(self.list_speakers))
