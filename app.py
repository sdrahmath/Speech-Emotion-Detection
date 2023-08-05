import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pickle
import io

# Load the pickled model
filename = 'C:/Users/SYED AZIZ AHMED/OneDrive/Desktop/Speech emotion/modelForPrediction1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to extract features from audio
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

def main():
    st.title('Speech Emotion Detection')
    audio_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

    if audio_file is not None:
        try:
            # Play the uploaded audio
            st.subheader("Uploaded Audio:")
            st.audio(audio_file, format='audio/wav')  # Assuming the audio is in WAV format

            features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
            features = features.reshape(1, -1)
            prediction = loaded_model.predict(features)

            predicted_emotion = prediction[0].upper()

            st.subheader("Prediction Result:")
            st.write(f"<p  style='font-size: 20px;'>The predicted emotion is: <span style='font-size: 30px; font-weight: bold;'>{predicted_emotion}</span></p>", unsafe_allow_html=True)
        except Exception as e:
            st.error("Error occurred while processing the audio file.")
            st.error(str(e))

if __name__ == '__main__':
    main()
