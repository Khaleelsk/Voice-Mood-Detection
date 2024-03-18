import streamlit as st
import soundfile as sf
import pyaudio
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('best_model.h5')

# Load the categories used in the OneHotEncoder
categories = np.array(['fear', 'angry', 'fear', 'neutral', 'sad', 'ps', 'happy'], dtype=object)

# Function to extract MFCC features from audio data
def extract_mfcc(data, sr):
    # Convert the audio data to floating-point
    data = data.astype(np.float32) / np.iinfo(np.int16).max
    
    # Extract MFCC features
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    
    return mfcc

# Function to preprocess audio data and predict emotion
def predict_emotion(audio_data, sr):
    mfcc = extract_mfcc(audio_data, sr=sr)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)

    # Predict emotion
    predictions = model.predict(mfcc)

    # Decode one-hot encoded prediction
    predicted_label = categories[np.argmax(predictions)]

    return predicted_label

# Function to randomly predict emotion for debugging
def predict_emotion_random():
    return np.random.choice(categories)

# Streamlit app
st.title("<-----Voice Mood Detection----->")

option = st.radio("Choose an option:", ("Record Audio", "Upload Audio File"))

if option == "Record Audio":
    st.sidebar.title("Recording Options")
    duration = st.sidebar.slider("Recording Duration (seconds):", 1, 10, 5)

    st.info("Click the 'Start Recording' button to begin recording.")

    if st.button("Start Recording"):
        st.info(f"Recording... Please speak for {duration} seconds.")

        # Record audio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, input=True, frames_per_buffer=1024)
        frames = []

        for i in range(0, int(22050 / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)

        st.success("Recording complete!")

        # Convert frames to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

        # Predict emotion
        predicted_emotion = predict_emotion(audio_data, sr=22050)
        st.success(f"Predicted Emotion: {predicted_emotion}")

elif option == "Upload Audio File":
    st.sidebar.title("Upload Options")
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file:
        st.info("File uploaded successfully!")

        # Load audio file
        audio_data, sr = librosa.load(uploaded_file, sr=None)

        # Randomly predict emotion for debugging
        predicted_emotion = predict_emotion_random()
        st.success(f"Predicted Emotion : {predicted_emotion}")