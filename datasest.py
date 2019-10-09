import glob
import os
import random

import librosa
import numpy as np
from tensorflow.python.keras.utils import to_categorical

from constants import DATA_DIR, CLASSES, SPEAKER_IDX, CHAPTER_IDX, FILENAME_IDX, DURATION, NUM_MFCC, \
    SPEAKER_FILE, DATASET_STR


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
    GENDER_CLASSES = ['M', 'F']
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
