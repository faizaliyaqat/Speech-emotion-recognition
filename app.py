import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pickle
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Set SER project path
ser_path = "C:/SER_Wav2Vec_Project"
model_path = os.path.join(ser_path, "model.pkl")
features_path = os.path.join(ser_path, "feature_names.pkl")

# Load model & feature names
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(features_path, "rb") as f:
    feature_names = pickle.load(f)

# Load Wav2Vec 2.0 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Streamlit app title
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a WAV file to detect emotion using Wav2Vec 2.0 + Random Forest")

# Upload file section
uploaded_file = st.file_uploader("📁 Choose a WAV file", type=["wav"])

# Feature extraction function
def extract_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

    features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return features.reshape(1, -1)

# If file is uploaded
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save the uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Plot waveform
    st.subheader("📊 Waveform Preview")
    try:
        y, sr = librosa.load("temp.wav", sr=16000)
        fig, ax = plt.subplots(figsize=(8, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#6c63ff")
        ax.set_title("Waveform")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Couldn't show waveform: {e}")

    # Extract features & predict
    try:
        features = extract_features("temp.wav")
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        # Emoji mapping
        emoji_map = {
            "angry": "😠",
            "calm": "😌",
            "disgust": "🤢",
            "fear": "😨",
            "happy": "😄",
            "neutral": "😐",
            "sad": "😢",
            "surprise": "😲"
        }

        emotion_emoji = emoji_map.get(prediction.lower(), "🎭")
        st.markdown(f"<h3 style='color:#6c63ff;'>Predicted Emotion: {emotion_emoji} <b>{prediction.capitalize()}</b></h3>", unsafe_allow_html=True)

        # 📊 Confidence bar chart
        st.subheader("🔍 Model Confidence")
        emotion_labels = model.classes_
        proba_percent = np.round(proba * 100, 2)

        fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
        bars = ax_bar.barh(emotion_labels, proba_percent, color="#ffb703")
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel("Confidence (%)")
        ax_bar.invert_yaxis()
        for i, bar in enumerate(bars):
            ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{proba_percent[i]}%", va='center')
        st.pyplot(fig_bar)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
