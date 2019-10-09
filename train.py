import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import LSTM, Flatten, Dense, TimeDistributed, Conv1D, \
    MaxPooling1D
from tensorflow.python.keras.models import Sequential

from constants import NUM_MFCC, NUM_FRAMES
from datasest import get_data, get_mfcc

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

fh = logging.FileHandler(r'/home/u1116888/log/log.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def main():
    X, Y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    x_audio = []
    for x in X_train:
        x_audio.append(np.reshape(get_mfcc(x), [NUM_MFCC, NUM_FRAMES, 1]))

    model = Sequential()

    model.add(TimeDistributed(
        Conv1D(filters=16, kernel_size=4, padding='same', activation=tf.nn.relu, data_format='channels_last'),
        input_shape=(NUM_MFCC, NUM_FRAMES, 1)))

    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, padding='same', activation=tf.nn.relu)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(10, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(units=512, activation=tf.nn.tanh))
    model.add(Dense(units=256, activation=tf.nn.tanh))
    model.add(Dense(units=Y.shape[1], activation=tf.nn.softmax, name='top_layer'))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])  # optimizer was 'Adam'

    model.summary()
    x_train = np.reshape(x_audio, [len(x_audio), NUM_MFCC, NUM_FRAMES, 1])

    model.fit(x_train, y_train, batch_size=8, epochs=10, verbose=1, validation_split=0.2)
    model.save_weights('speaker_gender_recognition.h5')

    test(X_test, y_test, model)


def test(X_test, y_test, model):
    correct_count = 0
    logger.info("Testing on {} datasets".format(len(X_test)))
    for i in range(len(X_test)):
        audio = np.reshape(get_mfcc(X_test[i]), [1, NUM_MFCC, NUM_FRAMES, 1])
        predict_index = np.argmax(model.predict(audio))
        true_index = np.argmax(y_test[i])
        if predict_index == true_index:
            correct_count += 1

    test_accuracy = (correct_count / len(X_test) * 100)
    logger.info("Test Accuracy: ", test_accuracy)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.keras.backend.clear_session()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    main()
