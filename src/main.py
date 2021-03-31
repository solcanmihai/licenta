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
    # trainer = TrainingService(14, 30, 10)
    #
    # trainer.read_raw_speaker_data('E:\\LICENTA\\SPEECHDATA\\wav', first_n_utterances=10, first_n_speakers=50, skip_n_speakers=0)
    # trainer.extract_mfcc_from_raw_data()
    # trainer.train_model('hifi_10_50')

    trainer2 = TrainingService(14, 30, 10)
    trainer2.read_raw_speaker_data('E:\\LICENTA\\SPEECHDATA\\wav', first_n_utterances=1, first_n_speakers=5, skip_n_speakers=50)
    trainer2.extract_mfcc_from_raw_data()

    trainer3 = TrainingService(14, 30, 10)
    trainer3.read_raw_speaker_data('E:\\LICENTA\\SPEECHDATA\\wav', first_n_utterances=1, first_n_speakers=5, skip_n_speakers=50)
    trainer3.extract_mfcc_from_raw_data()

    enrollment = EnrollmentService("hifi_10_50", project_root_directory)

    for speaker in trainer2.speakers:
        enrollment.compute_d_vector(speaker)

    for speaker in trainer3.speakers:
        enrollment.compute_d_vector(speaker)

    for speaker1 in trainer2.speakers:
        print('$$$$$$$$$$$$$$')
        for speaker2 in trainer3.speakers:
            print(f'{speaker1.utterances[0].name} {speaker2.utterances[0].name}')
            print(EnrollmentService.compare_speakers(speaker1, speaker2))
