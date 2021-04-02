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

from src.entities.Speaker import Utterance, Speaker


def check_if_hifi_microphone(utterance_name: str):
    split_name = utterance_name.split('_')
    return split_name[1] == "7"

class TrainingService:

    def __init__(self, filterbank_nr: int, context_left_size: int, context_right_size: int, root_directory: str):
        self.filterbank_nr = filterbank_nr
        self.unique_speaker_number = 0
        self.speakers: List[Speaker] = []
        self.context_left_size = context_left_size
        self.context_right_size = context_right_size
        self.model = None
        self.root_directory = root_directory

    def build_training_data(self):
        frames_with_context = []
        labels = []

        for speaker in self.speakers:
            for utterance in speaker.utterances:
                for stacked_frame in utterance.mfcc_stacked_frames:
                    frames_with_context.append(stacked_frame)
                    labels.append(speaker.id)

        return frames_with_context, labels

    def train_model(self, save_name: str):
        training_generator = DataGenerator(self.speakers)

        hidden_layers = Sequential([
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
        ])

        input_model = Input(shape=((self.context_left_size + self.context_right_size + 1) * self.filterbank_nr))
        coefficient_extractor = hidden_layers(input_model)

        training_model = Model(inputs=input_model, outputs=coefficient_extractor)

        training_output_layer = Dropout(0.3)(coefficient_extractor)
        training_output_layer = Dense(len(self.speakers), 'softmax')(training_output_layer)

        model = Model(inputs=input_model, outputs=training_output_layer)

        # x_data, y_data = self.build_training_data()
        # filenames_shuffled, y_labels_shuffled = shuffle(x_data, y_data)
        # X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(np.array(filenames_shuffled),
        #                                                                       y_labels_shuffled, test_size=0.01,
        #                                                                       random_state=1)

        print(model.summary())

        sgd = SGD(lr=0.008)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=0.001)

        csv_logger = CSVLogger('training.log')
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(training_generator, epochs=512, verbose=1, callbacks=[csv_logger])

        os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'trained_models'))
        training_model.save(f'{save_name}.h5')
        self.model = training_model

    def read_raw_speaker_data(
            self,
            path_to_folder: str,
            first_n_utterances: int,
            first_n_speakers: int,
            skip_n_speakers: int,
            file_to_save: str = 'delete_me',
    ):
        os.chdir(path_to_folder)
        speaker_names = os.listdir()
        self.unique_speaker_number = first_n_speakers

        for speaker_number, speaker_name in enumerate(speaker_names[skip_n_speakers:skip_n_speakers + first_n_speakers], start=skip_n_speakers):
            utterances_names = shuffle(os.listdir(speaker_name))

            utterances_names = list(filter(check_if_hifi_microphone, utterances_names))

            speaker = Speaker(speaker_name, speaker_number, [], None)

            print(f'Reading speaker {speaker_name}...')

            for utterance_name in utterances_names[:first_n_utterances]:
                path_to_file = os.path.join(os.getcwd(), speaker_name, utterance_name)
                utterance_file = pydub.AudioSegment.from_wav(path_to_file)
                utterance_raw_data = utterance_file.get_array_of_samples()
                frame_rate = utterance_file.frame_rate
                mfcc_frames = speechpy.feature.mfcc(
                    np.asarray(utterance_raw_data),
                    frame_rate,
                    num_cepstral=self.filterbank_nr).flatten()

                speaker.utterances.append(Utterance(utterance_name, None, frame_rate, mfcc_frames, None, None))

            self.speakers.append(speaker)

        os.chdir(os.path.join(self.root_directory, '..', 'computed_mfcc'))
        p_file = open(f'{file_to_save}.pkl', 'wb')
        pickle.dump(self.speakers, p_file)
        p_file.close()

    def read_precomputed_mfcc(self, file_to_read: str):
        os.chdir(os.path.join(self.root_directory, '..', 'computed_mfcc'))
        p_file = open(f'{file_to_read}.pkl', 'rb')
        self.speakers = pickle.load(p_file)

    def extract_mfcc_from_raw_data(self):
        for speaker in self.speakers:
            for utterance in speaker.utterances:
                pass
                # Moved to raw data

                # utterance.mfcc_frames = speechpy.feature.mfcc(
                #         np.asarray(utterance.raw_data),
                #         utterance.frame_rate,
                #         num_cepstral=self.filterbank_nr)

                # FLATTENED FORMAT
                # utterance.mfcc_frames = speechpy.feature.mfcc(
                #         np.asarray(utterance.raw_data),
                #         utterance.frame_rate,
                #         num_cepstral=self.filterbank_nr).flatten()


        # for speaker in self.speakers:
        #     for utterance in speaker.utterances:
        #         frames = utterance.mfcc_frames
        #         stacked_frames = []
        #
        #         for frame_index, frame in enumerate(frames[self.context_left_size:-self.context_right_size], start=self.context_left_size):
        #             stacked = np.array(frames[frame_index - self.context_left_size:frame_index + self.context_right_size + 1]).flatten()
        #             stacked_frames.append(stacked)
        #
        #         utterance.mfcc_stacked_frames = stacked_frames


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_speakers: List[Speaker], batch_size=1024, nr_mfcc=14, context_left=30, context_right=10, n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
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
        'Denotes the number of batches per epoch'
        return int(floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
        self.indexes = shuffle(self.list_ids)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.nr_context * self.nr_mfcc))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            (speaker_index, utterance_index, slice_start, slice_end) = ID
            X[i,] = np.array(self.list_speakers[speaker_index].utterances[utterance_index].mfcc_frames[slice_start: slice_end])

            # Store class
            y[i] = self.list_speakers[speaker_index].id

        return X, keras.utils.to_categorical(y, num_classes=len(self.list_speakers))