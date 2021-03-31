import os
import keras
import sklearn

import numpy as np
from scipy.spatial import distance

from src.entities.Speaker import Speaker


class EnrollmentService:

    def __init__(self, file_name, project_root_directory):
        model_path = os.path.join(project_root_directory, '..', 'trained_models', f'{file_name}.h5')
        self.model = keras.models.load_model(model_path)
        print(self.model.summary())

    def compare_speakers(speaker1: Speaker, speaker2: Speaker):
        return distance.cosine(speaker1.d_vector, speaker2.d_vector)

    def compute_d_vector(self, speaker: Speaker):
        speaker_d_vectors = []

        for utterance in speaker.utterances:
            utterance_d_vector = []

            for stacked_frame in utterance.mfcc_stacked_frames:
                output_coefficients = self.model.predict(np.expand_dims(stacked_frame, axis=0))
                normalized_output_coefficients = sklearn.preprocessing.normalize(output_coefficients)[0]
                utterance_d_vector.append(normalized_output_coefficients)
            utterance_d_vector = np.sum(np.array(utterance_d_vector), axis=0)
            utterance.d_vector = utterance_d_vector
            speaker_d_vectors.append(utterance.d_vector)

        speaker.d_vector = np.average(np.array(speaker_d_vectors), axis=0)


            # frames = utterance.coefficients
            # d_vector = []
            #
            # for frame_i, frame in enumerate(frames[30:-10], start=30):
            #     stacked = np.array([np.array(frames[frame_i - 30:frame_i + 10]).flatten()])
            #
            #     # for frame_i_context in utterance_mfcc[frame_i - 29:frame_i + 10]:
            #     # stacked = [sum(x) for x in zip(stacked, frame_i_context)]
            #
            #     utterance_output = model.predict(stacked)
            #     # print(utterance_output)
            #     normalized = sklearn.preprocessing.normalize(utterance_output)
            #     d_vector.append(normalized.flatten())
            #
            # d_vector = np.sum(np.array(d_vector), axis=0)
            # utterance.d_vector = d_vector
            #
            # # if not self.speakers_d_vectors[utterance.name]:
            # #     self.speakers_d_vectors[utterance.name] = []
            # #
            # # self.speakers_d_vectors[utterance.name].append(d_vector)

        # for utterance in self.speakers_with_coefficients:
        #     print('$$$$$$$$$$')
        #     for utterance2 in self.speakers_with_coefficients:
        #         print(distance.cosine(utterance.d_vector, utterance2.d_vector))