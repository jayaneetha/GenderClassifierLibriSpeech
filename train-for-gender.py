import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Flatten, Dense, TimeDistributed, Conv1D, \
    MaxPooling1D
from tensorflow.python.keras.models import Sequential

from constants import NUM_MFCC, NUM_FRAMES
from datasest import get_mfcc, get_dataset


def main():
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = get_dataset(class_type='gender')

    print("Training length: {}".format(len(X_train)))
    print("Testing length: {}".format(len(X_test)))
    print("Validation length: {}".format(len(X_valid)))

    x_audio_training = []

    for i in range(len(X_train)):
        if i % 100 == 0:
            print("{0:.2f} loaded in X_train".format(i / len(X_train)))
        x_audio_training.append(np.reshape(get_mfcc(X_train[i]), [NUM_MFCC, NUM_FRAMES, 1]))

    x_audio_validation = []
    for i in range(len(X_valid)):
        if i % 100 == 0:
            print("{0:.2f} loaded in X_valid".format(i / len(X_valid)))
        x_audio_validation.append(np.reshape(get_mfcc(X_valid[i]), [NUM_MFCC, NUM_FRAMES, 1]))

    model = Sequential()

    model.add(TimeDistributed(
        Conv1D(filters=16, kernel_size=4, padding='same', activation=tf.nn.relu, data_format='channels_last'),
        input_shape=(NUM_MFCC, NUM_FRAMES, 1)))

    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, padding='same', activation=tf.nn.relu)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(units=512, activation=tf.nn.tanh))
    model.add(Dense(units=256, activation=tf.nn.tanh))
    model.add(Dense(units=y_train.shape[1], activation=tf.nn.softmax, name='top_layer'))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])  # optimizer was 'Adam'

    model.summary()

    x_train = np.reshape(x_audio_training, [len(x_audio_training), NUM_MFCC, NUM_FRAMES, 1])
    x_valid = np.reshape(x_audio_validation, [len(x_audio_validation), NUM_MFCC, NUM_FRAMES, 1])

    print("Start Fitting")
    model.fit(x_train, y_train, batch_size=8, epochs=10, verbose=1, validation_data=(x_valid, y_valid))

    model_name = 'Libri_Gender_v2'
    print("Saving model as {}".format(model_name))
    model.save_weights(model_name + '.h5')

    test(X_test, y_test, model)


def test(X_test, y_test, model):
    correct_count = 0
    print("Testing on {} datasets".format(len(X_test)))
    for i in range(len(X_test)):
        audio = np.reshape(get_mfcc(X_test[i]), [1, NUM_MFCC, NUM_FRAMES, 1])
        predict_index = np.argmax(model.predict(audio))
        true_index = np.argmax(y_test[i])
        if predict_index == true_index:
            correct_count += 1

    test_accuracy = (correct_count / len(X_test) * 100)
    print("Test Accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.keras.backend.clear_session()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    main()
