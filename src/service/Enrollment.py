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

    def compare_d_vectors(d_vector_1, d_vector_2):
        return distance.cosine(d_vector_1, d_vector_2)

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