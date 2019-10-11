import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Flatten, Dense, TimeDistributed, Conv1D, \
    MaxPooling1D, Dropout
from tensorflow.python.keras.models import Sequential

from constants import NUM_MFCC, NUM_FRAMES
from dataset import get_dataset, get_mfccs, save_to_pkl

feature_actions = 'load-from-pkl'  # { 'load-from-pkl', 'load-from-wav' }
feature_store = True  # save the feature pkl file


def main():
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = get_dataset(class_type='speaker')

    print("Training length: {}".format(len(X_train)))
    print("Testing length: {}".format(len(X_test)))
    print("Validation length: {}".format(len(X_valid)))

    if feature_actions == 'load-from-wav':
        x_audio_training = get_mfccs(X_train)
        x_audio_validation = get_mfccs(X_valid)

        if feature_store:
            save_to_pkl(x_audio_training, 'training.pkl')
            save_to_pkl(x_audio_validation, 'validation.pkl')

    elif feature_actions == 'load-from-pkl':
        x_audio_training = get_mfccs(pickle_file='training.pkl')
        x_audio_validation = get_mfccs(pickle_file='validation.pkl')
    else:
        print("Error in 'feature_actions'")
        return

    model = Sequential()

    model.add(TimeDistributed(
        Conv1D(filters=16, kernel_size=4, padding='same', activation=tf.nn.relu, data_format='channels_last'),
        input_shape=(NUM_MFCC, NUM_FRAMES, 1)))

    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, padding='same', activation=tf.nn.relu)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
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
    model.fit(x_train, y_train, batch_size=8, epochs=15, verbose=1, validation_data=(x_valid, y_valid))

    model_name = 'Libri_Speaker_v1'
    print("Saving model as {}".format(model_name))
    model.save_weights(model_name + '.h5')

    test(X_test, y_test, model)


def test(X_test, y_test, model):
    correct_count = 0
    print("Testing on {} datasets".format(len(X_test)))
    x_audio_testing = get_mfccs(X_test)
    for i in range(len(X_test)):
        audio = np.reshape(x_audio_testing, [1, NUM_MFCC, NUM_FRAMES, 1])
        predict_index = np.argmax(model.predict(audio))
        true_index = np.argmax(y_test[i])
        if predict_index == true_index:
            correct_count += 1

    test_accuracy = (correct_count / len(X_test) * 100)
    print("Test Accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # tf.keras.backend.clear_session()
    #
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(sess)

    main()
