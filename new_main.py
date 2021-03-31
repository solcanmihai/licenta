from Training import Training
import tensorflow as tf


def keras_config():
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)


if __name__ == '__main__':
    keras_config()
    trainer = Training(14)

    trainer.extract_coefficients('E:\\LICENTA\\SPEECHDATA\\wav', first_n_utterances=2, first_n_speakers=5)
    trainer.load_model()
    # trainer.stack_frames()
    # trainer.train_model()
