import os

from src.service.Enrollment import EnrollmentService
from src.service.Training import TrainingService
import tensorflow as tf


def keras_config():
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)
    print(tf.config.list_physical_devices("GPU"))


if __name__ == '__main__':
    keras_config()
    project_root_directory = os.getcwd()
    trainer = TrainingService(14, 30, 10)

    trainer.read_raw_speaker_data('E:\\LICENTA\\SPEECHDATA\\wav', first_n_utterances=20, first_n_speakers=200, skip_n_speakers=0)
    # trainer.extract_mfcc_from_raw_data()
    trainer.train_model('optimized_reading_20_200')

    # trainer2 = TrainingService(14, 30, 10)
    # trainer2.read_raw_speaker_data('E:\\LICENTA\\SPEECHDATA\\wav', first_n_utterances=4, first_n_speakers=10, skip_n_speakers=130)
    # trainer2.extract_mfcc_from_raw_data()
    #
    # trainer3 = TrainingService(14, 30, 10)
    # trainer3.read_raw_speaker_data('E:\\LICENTA\\SPEECHDATA\\wav', first_n_utterances=4, first_n_speakers=10, skip_n_speakers=130)
    # trainer3.extract_mfcc_from_raw_data()
    #
    # enrollment = EnrollmentService("hifi_20_100", project_root_directory)
    #
    # for speaker in trainer2.speakers:
    #     enrollment.compute_d_vector(speaker)
    #
    # for speaker in trainer3.speakers:
    #     enrollment.compute_d_vector(speaker)
    #
    # for speaker1 in trainer2.speakers:
    #     print('$$$$$$$$$$$$$$')
    #     for speaker2 in trainer3.speakers:
    #         if speaker1.name == speaker2.name:
    #             for utterance in speaker2.utterances:
    #                 print(f'{speaker1.name} {utterance.name}')
    #                 print(EnrollmentService.compare_d_vectors(speaker1.d_vector, utterance.d_vector))
