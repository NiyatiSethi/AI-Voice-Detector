import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

X = []
y = []

def extract_features(file):
    audio, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

for label, folder in [("human", "dataset/human"), ("ai", "dataset/ai")]:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        features = extract_features(path)
        X.append(features)
        y.append(0 if label == "human" else 1)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "voice_model.pkl")
print("Model trained and saved")
