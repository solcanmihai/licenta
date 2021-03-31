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

from src.entities.Speaker import Utterance, Speaker


class TrainingService:

    def __init__(self, filterbank_nr: int, context_left_size: int, context_right_size: int):
        self.filterbank_nr = filterbank_nr
        self.unique_speaker_number = 0
        self.speakers: List[Speaker] = []
        self.context_left_size = context_left_size
        self.context_right_size = context_right_size
        self.model = None

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
        hidden_layers = Sequential([
            Dense(256, activation='relu'),
            # Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
        ])

        input_model = Input(shape=(41 * self.filterbank_nr))
        coefficient_extractor = hidden_layers(input_model)

        training_model = Model(inputs=input_model, outputs=coefficient_extractor)

        training_output_layer = Dropout(0.3)(coefficient_extractor)
        training_output_layer = Dense(self.unique_speaker_number, 'softmax')(training_output_layer)

        model = Model(inputs=input_model, outputs=training_output_layer)

        x_data, y_data = self.build_training_data()
        filenames_shuffled, y_labels_shuffled = shuffle(x_data, y_data)
        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(np.array(filenames_shuffled),
                                                                              y_labels_shuffled, test_size=0.1,
                                                                              random_state=1)

        print(model.summary())

        sgd = SGD(lr=0.02)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.00001)

        csv_logger = CSVLogger('training.log')
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_filenames, to_categorical(y_train), batch_size=5000, epochs=1000, verbose=1,
                  validation_data=(X_val_filenames, to_categorical(y_val)), callbacks=[csv_logger])

        os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'trained_models'))
        training_model.save(f'{save_name}.h5')
        self.model = training_model

    def read_raw_speaker_data(
            self,
            path_to_folder: str,
            first_n_utterances: int,
            first_n_speakers: int,
            skip_n_speakers: int,
    ):
        os.chdir(path_to_folder)
        speaker_names = os.listdir()
        self.unique_speaker_number = first_n_speakers

        for speaker_number, speaker_name in enumerate(speaker_names[skip_n_speakers:skip_n_speakers + first_n_speakers], start=skip_n_speakers):
            utterances_names = shuffle(os.listdir(speaker_name))

            speaker = Speaker(speaker_name, speaker_number, [], None)

            for utterance_name in utterances_names[:first_n_utterances]:
                path_to_file = os.path.join(os.getcwd(), speaker_name, utterance_name)
                utterance_file = pydub.AudioSegment.from_wav(path_to_file)
                utterance_raw_data = utterance_file.get_array_of_samples()
                frame_rate = utterance_file.frame_rate

                speaker.utterances.append(Utterance(utterance_name, utterance_raw_data, frame_rate, None, None, None))

            self.speakers.append(speaker)

    def extract_mfcc_from_raw_data(self):
        for speaker in self.speakers:
            for utterance in speaker.utterances:
                utterance.mfcc_frames = speechpy.feature.mfcc(
                        np.asarray(utterance.raw_data),
                        utterance.frame_rate,
                        num_cepstral=self.filterbank_nr)

        for speaker in self.speakers:
            for utterance in speaker.utterances:
                frames = utterance.mfcc_frames
                stacked_frames = []

                for frame_index, frame in enumerate(frames[self.context_left_size:-self.context_right_size], start=self.context_left_size):
                    stacked = np.array(frames[frame_index - self.context_left_size:frame_index + self.context_right_size + 1]).flatten()
                    stacked_frames.append(stacked)

                utterance.mfcc_stacked_frames = stacked_frames

