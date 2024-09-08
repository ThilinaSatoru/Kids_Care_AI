import joblib
import librosa
import numpy as np

loaded_model = joblib.load('model.joblib')
loaded_le = joblib.load('label.joblib')

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


def extract_features(file_path):
    try:
        # Load audio file and extract features
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = np.mean(
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                 window=window).T, axis=0)
        mel = np.mean(
            librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                           window='hann', n_mels=n_mels).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr, n_fft=n_fft,
                                                             hop_length=hop_length, win_length=win_length,
                                                             n_bands=n_bands, fmin=fmin).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        # print(shape(features))
        return features
    except:
        print("Error: Exception occurred in feature extraction")
        return None


def predict_cry(file_path):
    # Load the saved model and LabelEncoder
    loaded_model = joblib.load('model.joblib')
    loaded_le = joblib.load('label.joblib')

    # Extract features from the new audio file
    features = extract_features(file_path)

    if features is not None:
        # Reshape features to match the input shape expected by the model
        features = features.reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(features)

        # Convert prediction back to original label
        predicted_label = loaded_le.inverse_transform(prediction)

        return predicted_label[0]
    else:
        return "Error: Could not extract features from the audio file"


# Example usage
file_path = 'd6cda191-4962-4308-9a36-46d5648a95ed-1431159262344-1.7-m-04-bp.wav'
result = predict_cry(file_path)
print(f"Predicted cry type: {result}")
