import os

import pyaudio

from configs.config import OUTPUT_DIRECTORY

# ===================================================================================================
# CRY - Yamnet Model Path
CRY_MODEL_DIR = 'audio-ml/'
YAMNET_MODEL_PATH = 'https://tfhub.dev/google/yamnet/1'
CRY_CLASSIFIER_MODEL_PATH = 'infant_cry_classifier.h5'
# XGBOOST_MODEL_PATH = 'xgboost_model.pkl'
XGBOOST_MODEL_PATH = 'model.joblib'
# ===================================================================================================


# ===================================================================================================
WAVE_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY + "/cry"
# Ensure output directory exists
os.makedirs(WAVE_OUTPUT_DIRECTORY, exist_ok=True)

# Audio parameters
DEVICE_INDEX = 3
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10
SILENCE_THRESHOLD = 10000  # Adjust this value based on your microphone and environment

# Feature extraction parameters
n_mfcc = 40
n_fft = 1024
hop_length = 10 * 16
win_length = 25 * 16
window = 'hann'
n_chroma = 12
n_mels = 128
n_bands = 7
fmin = 100
bins_per_octave = 12

# Cry reason labels
label_names = {0: 'belly_pain', 1: 'burping', 2: 'discomfort', 3: 'hungry', 4: 'tired'}
# ===================================================================================================
