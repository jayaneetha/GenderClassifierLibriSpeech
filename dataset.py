import glob
import os
import pickle
import random

import librosa
import numpy as np
from tensorflow.python.keras.utils import to_categorical

from constants import DATA_DIR, CLASSES, SPEAKER_IDX, CHAPTER_IDX, FILENAME_IDX, DURATION, NUM_MFCC, \
    SPEAKER_FILE, DATASET_STR, GENDER_CLASSES, NUM_CLASSES, NUM_FRAMES, PICKLE_FILE_PREFIX

speaker_gender_map = {}
TEMP_CLASS_INDEX = []


def init_speaker_gender_map():
    with open(SPEAKER_FILE) as f:
        content = f.readlines()

    for line in content:
        if DATASET_STR in line:
            sp = line.split('|')
            sp_id = sp[0].strip()
            gender = sp[1].strip()
            speaker_gender_map[sp_id] = gender


init_speaker_gender_map()


@DeprecationWarning
def get_data():
    speaker_gender = {}

    with open(SPEAKER_FILE) as f:
        content = f.readlines()

    for line in content:
        if DATASET_STR in line:
            sp = line.split('|')
            sp_id = sp[0].strip()
            gender = sp[1].strip()
            speaker_gender[sp_id] = gender

    FILE_DIR = DATA_DIR

    file_list = glob.glob(FILE_DIR + '*/*/*.wav')
    print("Loading Data from :", FILE_DIR)
    all_data = []

    for f in file_list:
        speaker_id = f.split("/")[SPEAKER_IDX]
        chapter_id = f.split("/")[CHAPTER_IDX]
        filename = f.split("/")[FILENAME_IDX]

        if len(CLASSES) > 0:
            if speaker_id in CLASSES:
                all_data.append({
                    "speaker": speaker_id,
                    'filename': os.path.join(FILE_DIR,
                                             os.path.join(speaker_id, os.path.join(chapter_id, os.path.join(filename))))
                })
        else:
            all_data.append({
                "speaker": speaker_id,
                'filename': os.path.join(FILE_DIR,
                                         os.path.join(speaker_id, os.path.join(chapter_id, os.path.join(filename))))
            })

    random.shuffle(all_data)

    X = []
    Y = []
    TEMP_CLASS_INDEX = []
    for d in all_data:
        X.append(d['filename'])
        if len(CLASSES) > 0:
            y = GENDER_CLASSES.index(speaker_gender.get(d['speaker']))
            # y = CLASSES.index(d['speaker'])
            Y.append(y)
        else:
            # if not d['speaker'] in TEMP_CLASS_INDEX:
            #     TEMP_CLASS_INDEX.append(d['speaker'])
            # Y.append(TEMP_CLASS_INDEX.index(d['speaker']))
            y = GENDER_CLASSES.index(speaker_gender.get(d['speaker']))
            Y.append(y)

    return X, to_categorical(Y, num_classes=len(GENDER_CLASSES))


def load_wav(filename):
    audio, sr = librosa.load(filename, duration=DURATION)
    return audio, sr


def add_missing_padding(audio, sr):
    signal_length = DURATION * sr
    audio_length = audio.shape[0]
    padding_length = signal_length - audio_length
    if padding_length > 0:
        padding = np.zeros(padding_length)
        signal = np.hstack((audio, padding))
        return signal
    return audio


def get_mfcc(filename):
    audio, sr = load_wav(filename)
    signal = add_missing_padding(audio, sr)
    return librosa.feature.mfcc(signal, sr, n_mfcc=NUM_MFCC)


def get_speaker_ids():
    speaker_ids = []
    with open(SPEAKER_FILE) as f:
        content = f.readlines()

    for line in content:
        if DATASET_STR in line:
            sp = line.split('|')
            sp_id = sp[0].strip()
            speaker_ids.append(sp_id)

    return speaker_ids


def get_mfccs(file_list=False, pickle_file=False):
    if pickle_file:
        pickle_file = PICKLE_FILE_PREFIX + pickle_file
        print("loading from pickle file : {}".format(pickle_file))
        infile = open(pickle_file, 'rb')
        x_audio = pickle.load(infile)
        infile.close()
        return x_audio
    else:
        x_audio = []
        for i in range(len(file_list)):
            if i % 100 == 0:
                print("{0:.2f} loaded ".format(i / len(file_list)))
            x_audio.append(np.reshape(get_mfcc(file_list[i]), [NUM_MFCC, NUM_FRAMES, 1]))
        return x_audio


def get_dataset(class_type='gender'):
    """@:param class_type: str - type of class needed to be in Y.
        values { 'gender' , 'speaker' }
    """
    train = []
    test = []
    valid = []
    speaker_ids = get_speaker_ids()

    for s in speaker_ids:
        file_list = glob.glob(DATA_DIR + s + '/*/*.wav')
        print("Loading Data from :", DATA_DIR + s)
        all_data = []
        for f in file_list:
            speaker_id = f.split("/")[SPEAKER_IDX]
            chapter_id = f.split("/")[CHAPTER_IDX]
            filename = f.split("/")[FILENAME_IDX]

            all_data.append(
                os.path.join(DATA_DIR, os.path.join(speaker_id, os.path.join(chapter_id, os.path.join(filename)))))

        random.shuffle(all_data)
        split_tuple = np.split(np.array(all_data), [int(0.7 * len(all_data)), int(0.9 * len(all_data))])
        train = train + split_tuple[0].tolist()
        test = test + split_tuple[1].tolist()
        valid = valid + split_tuple[2].tolist()

    if class_type == 'gender':
        x_train, y_train = get_XY_gender(train)
        x_test, y_test = get_XY_gender(test)
        x_valid, y_valid = get_XY_gender(valid)
        num_classes = len(GENDER_CLASSES)
    elif class_type == 'speaker':
        x_train, y_train = get_XY_speaker(train)
        x_test, y_test = get_XY_speaker(test)
        x_valid, y_valid = get_XY_speaker(valid)
        num_classes = NUM_CLASSES
    else:
        print("Invalid class_type. Required 'gender' or 'speaker'. Given: {}".format(class_type))
        return

    return (x_train, to_categorical(y_train, num_classes=num_classes)), \
           (x_test, to_categorical(y_test, num_classes=num_classes)), \
           (x_valid, to_categorical(y_valid, num_classes=num_classes))


def get_XY_gender(fileList):
    x = []
    y = []

    random.shuffle(fileList)

    for f in fileList:
        speaker_id = f.split("/")[SPEAKER_IDX]
        x.append(f)
        g = GENDER_CLASSES.index(speaker_gender_map.get(speaker_id))
        y.append(g)

    return x, y


def get_XY_speaker(fileList):
    x = []
    y = []

    random.shuffle(fileList)

    for f in fileList:
        speaker_id = f.split("/")[SPEAKER_IDX]
        x.append(f)

        if len(CLASSES) > 0:
            s = CLASSES.index(speaker_id)
            y.append(s)
        else:
            if not speaker_id in TEMP_CLASS_INDEX:
                TEMP_CLASS_INDEX.append(speaker_id)
            y.append(TEMP_CLASS_INDEX.index(speaker_id))

    return x, y


def save_to_pkl(data, filename):
    filename = PICKLE_FILE_PREFIX + filename
    print("Storing {} data to file: {}".format(len(data), filename))
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
