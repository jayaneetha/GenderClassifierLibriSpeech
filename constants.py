import socket

host = socket.gethostname()
CLASSES = []
NUM_MFCC = 40
NUM_FRAMES = 87
DURATION = 2  # in seconds
GENDER_CLASSES = ['M', 'F']
PICKLE_FILE_PREFIX = 'LibriSpeech-mfcc-'

# environment specific constants
if host == 'asimov':
    DATASET_STR = 'train-clean-100'
    DATA_ROOT = '/data/aq/shared/libri/LibriSpeech-100/LibriSpeech/'
    DATA_DIR = DATA_ROOT + DATASET_STR + '/'
    SPEAKER_FILE = DATA_ROOT + 'SPEAKERS.TXT'
    SPEAKER_IDX = 8
    CHAPTER_IDX = 9
    FILENAME_IDX = 10
    NUM_CLASSES = 251

if host == 'Thejans-MacBook-Pro.local':
    DATASET_STR = 'dev-clean'
    DATA_ROOT = '/Volumes/Kingston/datasets/audio/LibriSpeech/LibriSpeech/'
    DATA_DIR = DATA_ROOT + DATASET_STR + '/'
    SPEAKER_FILE = DATA_ROOT + 'SPEAKERS.TXT'
    SPEAKER_IDX = 8
    CHAPTER_IDX = 9
    FILENAME_IDX = 10
    NUM_CLASSES = 40

if host == 'rnd-gpu':
    DATASET_STR = 'dev-clean'
    DATA_ROOT = '/home/thejanr/dataset/LibriSpeech/'
    DATA_DIR = DATA_ROOT + DATASET_STR + '/'
    SPEAKER_FILE = DATA_ROOT + 'SPEAKERS.TXT'
    SPEAKER_IDX = 6
    CHAPTER_IDX = 7
    FILENAME_IDX = 8
    NUM_CLASSES = 40

if len(CLASSES) > 0:
    NUM_CLASSES = len(CLASSES)
