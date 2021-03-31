import scipy.io.wavfile as wav
import numpy as np
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

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

speaker_number = 50
first_n_utterances = 1
filterbank_nr = 14

hidden_layers = Sequential([
    Dense(256, input_shape=(40 * filterbank_nr,), activation='relu'),
    # Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
])

training_model = Model(inputs=Input(shape=(40 * filterbank_nr)), outputs=hidden_layers)
training_output_layer = Activation('relu')(hidden_layers)
training_output_layer = Dropout(0.3)(training_output_layer)
training_output_layer = Activation('relu')(training_output_layer)


# 'E:\\LICENTA\\SPEECHDATA\\wav'
# print(model.summary())


filenames = []
labels = []
coefficients_list = []




extract_coefficients()

for name, coefficients in zip(labels, filenames):
    print(name, coefficients)

# scaler = StandardScaler()
# scaler.fit(coefficients)
# coefficients = scaler.transform(coefficients)

filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)

X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(np.array(filenames_shuffled), y_labels_shuffled, test_size=0.1, random_state=1)

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
model.compile(optimizer=sgd, loss = 'categorical_crossentropy', metrics=['accuracy'])
print(type(X_train_filenames))
print(X_train_filenames)
print(type(y_train))
print(to_categorical(y_train))
model.fit(X_train_filenames, to_categorical(y_train), batch_size = 128, epochs = 1500, verbose = 1, validation_data = (X_val_filenames, to_categorical(y_val)), callbacks = [csv_logger])


# file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'sample-000000.mp3')
# print(file_name)
# samples = pydub.AudioSegment.from_mp3(file_name).get_array_of_samples()
# pydub.AudioSegment.converter = os.getcwd() + '\\ffmpeg.exe'
# audio_file = pydub.AudioSegment.from_mp3(file_name)
#
# mfe = speechpy.feature.lmfe(np.asarray(audio_file.get_array_of_samples()), audio_file.frame_rate, )
#
# print(mfe.size)

# speechpy.feature.mfcc(samples, 3)