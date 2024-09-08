import wave

import joblib
import librosa
import numpy as np
import pyaudio
import tensorflow as tf
import tensorflow_hub as hub

from configs.config import *
from configs.cry import *

users_ref = firebase_ref.child('cry_predictions')

# Load models
cry_classifier = tf.keras.models.load_model(CRY_MODEL_DIR + CRY_CLASSIFIER_MODEL_PATH)
yamnet_model = hub.load(YAMNET_MODEL_PATH)
# xgb_clf = joblib.load(CRY_MODEL_DIR + XGBOOST_MODEL_PATH)
loaded_model = joblib.load(CRY_MODEL_DIR + 'model.joblib')
loaded_le = joblib.load(CRY_MODEL_DIR + 'label.joblib')

# Define the feature extraction parameters
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


def preprocess_audio(audio, target_sr=16000, duration=5):
    max_length = target_sr * duration
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))
    return audio


def extract_features_cry_classifier(audio_data):
    audio_data = audio_data.astype(np.float32)
    scores, embeddings, _ = yamnet_model(audio_data)
    avg_embeddings = np.mean(embeddings.numpy(), axis=0)
    return np.expand_dims(avg_embeddings, axis=0)


def predict_cry(audio_data):
    features = extract_features_cry_classifier(audio_data)
    prediction = cry_classifier.predict(features)
    return prediction[0][0] < 0.5  # True if it's a cry, False otherwise


def extract_features(audio_data, sr):
    try:
        mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, window=window).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                     win_length=win_length, window='hann', n_mels=n_mels).T, axis=0)
        stft = np.abs(librosa.stft(audio_data))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        contrast = np.mean(
            librosa.feature.spectral_contrast(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                              n_bands=n_bands, fmin=fmin).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=audio_data, sr=sr).T, axis=0)
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None


def predict_cry_reason(audio_data, sr):
    features = extract_features(audio_data, sr)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = loaded_model.predict(features)
        predicted_label = loaded_le.inverse_transform(prediction)
        return predicted_label[0]
    return "Unknown"


def send_prediction_to_firebase(is_cry, cry_reason=None):
    users_ref.push({
        'timestamp': datetime.now().isoformat(),
        'is_cry': is_cry,
        'reason': cry_reason if is_cry else None
    })
    print("Prediction sent to Firebase")


def is_silent(audio_data):
    return np.max(np.abs(audio_data)) < SILENCE_THRESHOLD


def continuous_audio_monitor():
    p = pyaudio.PyAudio()

    device_info = p.get_device_info_by_index(DEVICE_INDEX)
    print(f"Using device: {device_info['name']}")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* Monitoring audio...")

    try:
        while True:
            frames = []
            is_sound = False

            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

                audio_chunk = np.frombuffer(data, dtype=np.int16)
                if not is_silent(audio_chunk):
                    is_sound = True

            if is_sound:
                print("Sound detected, processing...")
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wave_output_filename = os.path.join(WAVE_OUTPUT_DIRECTORY, f"output_{timestamp}.wav")
                wf = wave.open(wave_output_filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                processed_audio = preprocess_audio(audio_data)
                is_cry = predict_cry(processed_audio)

                if is_cry:
                    cry_reason = predict_cry_reason(processed_audio, RATE)
                    print(f"Infant cry detected! Reason: {cry_reason}")
                    send_prediction_to_firebase(True, cry_reason)
                else:
                    print("Sound detected, but no infant cry.")
                    send_prediction_to_firebase(False)
            else:
                print("No sound detected.")

    except KeyboardInterrupt:
        print("Monitoring stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    continuous_audio_monitor()
