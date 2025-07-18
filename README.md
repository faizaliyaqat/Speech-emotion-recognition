# Speech Emotion Recognition using Wav2Vec 2.0 and Random Forest

This project is a real-time Speech Emotion Recognition (SER) system that uses Wav2Vec 2.0 for feature extraction and a Random Forest classifier for emotion prediction. The app is built with Streamlit and integrates SHAP for explainable AI.

---

## Key Features

- Predicts emotion from audio speech input
- Uses Wav2Vec 2.0 for deep feature extraction
- Random Forest classifier trained on RAVDESS and SAVEE datasets
- SHAP integration to explain model predictions
- Streamlit interface with audio waveform and probability output

---

## Emotion Classes

- Neutral  
- Happy  
- Angry  
- Sad  
- Fear  
- Surprise

---

## Project Structure

SER_Wav2Vec_Project/
│
├── app.py # Streamlit app
├── model.pkl # Trained Random Forest model
├── feature_names.pkl # Feature list
├── requirements.txt # Project dependencies
├── .gitignore # Git exclusions
├── README.md # Project documentation
└── data/ # (Not included) Folder for RAVDESS and SAVEE


---

## Datasets (Not Included in Repository)

This project uses the following publicly available datasets:

- RAVDESS: https://zenodo.org/record/1188976  
- SAVEE: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
Due to licensing restrictions and file size, datasets are not included in this repository. To reproduce results, download them manually and place them in a `data/` folder.

---

## Installation and Running the App

### Install Dependencies

```bash
pip install -r requirements.txt
   
## Run Locally
streamlit run app.py
