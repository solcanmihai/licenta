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


class SpeakerMfcc:
    def __init__(self, id: int, name: str, coefficients: list):
        self.id = id
        self.name = name
        self.coefficients = coefficients
        self.d_vector = None


class Training:

    def __init__(self, filterbank_nr: int):
        self.speakers_with_coefficients: list[SpeakerMfcc] = []
        self.unique_speaker_number = 0
        self.filterbank_nr = filterbank_nr
        self.filenames = []
        self.labels = []
        self.models_directory = os.path.join(os.getcwd(), 'models')
        self.model_on_disk = os.path.join(os.getcwd(), 'models', 'without_output_layer.h5')

    def train_model(self):
        hidden_layers = Sequential([
            Dense(256, activation='relu'),
            # Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            # Dropout(0.3),
            # Dense(self.unique_speaker_number, activation='softmax'),
        ])

        input_model = Input(shape=(40 * self.filterbank_nr))
        coefficient_extractor = hidden_layers(input_model)

        training_model = Model(inputs=input_model, outputs=coefficient_extractor)

        training_output_layer = Dropout(0.3)(coefficient_extractor)
        training_output_layer = Dense(self.unique_speaker_number, 'softmax')(training_output_layer)

        model = Model(inputs=input_model, outputs=training_output_layer)

        # scaler = StandardScaler()
        # scaler.fit(coefficients)
        # coefficients = scaler.transform(coefficients)

        filenames_shuffled, y_labels_shuffled = shuffle(self.filenames, self.labels)
        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(np.array(filenames_shuffled),
                                                                              y_labels_shuffled, test_size=0.1,
                                                                              random_state=1)

        print(model.summary())
        # modelInput = Input(shape=(X_train_filenames.shape[0]))
        # modelInput = Input(shape=(X_train_filenames.shape[1], ))
        # features = model()
        #
        # spkModel = Model(inputs=modelInput, outputs=features)
        sgd = SGD(lr=0.02)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.00001)
        csv_logger = CSVLogger('training.log')
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        print(type(X_train_filenames))
        print(X_train_filenames)
        print(type(y_train))
        print(to_categorical(y_train))
        model.fit(X_train_filenames, to_categorical(y_train), batch_size=128, epochs=500, verbose=1,
                  validation_data=(X_val_filenames, to_categorical(y_val)), callbacks=[csv_logger])

        os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models'))

        model.save('with_output_layer.h5')
        training_model.save('without_output_layer.h5')

    def load_model(self):
        model = keras.models.load_model(self.model_on_disk)
        print(model.summary())

        for utterance in self.speakers_with_coefficients:
            frames = utterance.coefficients
            d_vector = []

            for frame_i, frame in enumerate(frames[30:-10], start=30):
                stacked = np.array([np.array(frames[frame_i - 30:frame_i + 10]).flatten()])

                # for frame_i_context in utterance_mfcc[frame_i - 29:frame_i + 10]:
                # stacked = [sum(x) for x in zip(stacked, frame_i_context)]

                utterance_output = model.predict(stacked)
                # print(utterance_output)
                normalized = sklearn.preprocessing.normalize(utterance_output)
                d_vector.append(normalized.flatten())

            d_vector = np.sum(np.array(d_vector), axis=0)
            utterance.d_vector = d_vector

            # if not self.speakers_d_vectors[utterance.name]:
            #     self.speakers_d_vectors[utterance.name] = []
            #
            # self.speakers_d_vectors[utterance.name].append(d_vector)

        # for utterance in self.speakers_with_coefficients:
        #     print('$$$$$$$$$$')
        #     for utterance2 in self.speakers_with_coefficients:
        #         print(distance.cosine(utterance.d_vector, utterance2.d_vector))

    def average_d_vectors(self):
        pass


    def stack_frames(self):
        self.filenames = []
        self.labels = []

        for utterance in self.speakers_with_coefficients:
            frames = utterance.coefficients
            print(len(frames))

            for frame_i, frame in enumerate(frames[30:-10], start=30):
                stacked = np.array(frames[frame_i - 30:frame_i + 10]).flatten()

                # for frame_i_context in utterance_mfcc[frame_i - 29:frame_i + 10]:
                # stacked = [sum(x) for x in zip(stacked, frame_i_context)]

                self.filenames.append(np.array(stacked))
                # labels.append(speaker_name)
                self.labels.append(utterance.id)

    def read_coefficients(self):
        pass

    def save_coefficients(self):
        pass

    def extract_coefficients(
            self,
            path_to_folder: str,
            first_n_utterances: int,
            first_n_speakers: int,
    ):
        os.chdir(path_to_folder)
        speaker_names = os.listdir()
        self.unique_speaker_number = first_n_speakers

        self.speakers_with_coefficients = []

        for speaker_name, speaker_number in zip(speaker_names[:first_n_speakers], range(first_n_speakers)):
            utterances_names = os.listdir(speaker_name)

            for utterance_name in utterances_names[:first_n_utterances]:
                path_to_file = os.path.join(os.getcwd(), speaker_name, utterance_name)
                utterance = pydub.AudioSegment.from_wav(path_to_file)
                utterance_mfcc = speechpy.feature.mfcc(np.asarray(utterance.get_array_of_samples()),
                                                       utterance.frame_rate,
                                                       num_cepstral=self.filterbank_nr)
                self.speakers_with_coefficients.append(SpeakerMfcc(speaker_number, speaker_name, utterance_mfcc))
