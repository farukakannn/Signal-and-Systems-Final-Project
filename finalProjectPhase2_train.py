import os
import librosa
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
import joblib
warnings.filterwarnings('ignore')

EMOTIONS = ['Neutral', 'Happy', 'Angry', 'Sad', 'Surprised']

def extract_features(data, sample_rate):
    # Comprehensive Feature Extraction Pipeline
    zcr = librosa.feature.zero_crossing_rate(y=data)
    rms = librosa.feature.rms(y=data)
    stft = np.abs(librosa.stft(data))
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    
    features = np.hstack((
        np.mean(zcr), np.std(zcr),
        np.mean(rms), np.std(rms),
        np.mean(chroma_stft, axis=1), np.std(chroma_stft, axis=1), np.min(chroma_stft, axis=1), np.max(chroma_stft, axis=1),
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.min(mfcc, axis=1), np.max(mfcc, axis=1),
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(mel, axis=1), np.std(mel, axis=1), np.min(mel, axis=1), np.max(mel, axis=1)
    ))
    return features

def load_data(folder_path, metadata_path):
    # Cache Check Mechanism
    if os.path.exists('X_features_ultimate.npy') and os.path.exists('y_labels_ultimate.npy'):
        print("INFO: Pre-compiled feature cache detected. Loading dataset into memory...")
        X = np.load('X_features_ultimate.npy')
        y = np.load('y_labels_ultimate.npy')
        return X, y

    print("INFO: Cache not detected. Initiating robust feature extraction and augmentation pipeline...")
    try:
        df = pd.read_excel(metadata_path) 
    except Exception as e:
        print(f"ERROR: Failed to read the metadata file. Details: {e}")
        return np.array([]), np.array([])

    file_to_emotion = {str(row['File name']).strip(): str(row['Feeling']).strip().capitalize() for _, row in df.iterrows()}

    X, y = [], []
    processed_files = 0

    for file in os.listdir(folder_path):
        if file.endswith('.wav') and file in file_to_emotion and file_to_emotion[file] in EMOTIONS:
            emotion = file_to_emotion[file]
            file_path = os.path.join(folder_path, file)
            
            try:
                data, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
                
                # Baseline and Augmentation
                X.append(extract_features(data, sample_rate)); y.append(emotion)
                
                noise = np.random.randn(len(data))
                X.append(extract_features(data + 0.005 * noise, sample_rate)); y.append(emotion)
                X.append(extract_features(librosa.effects.pitch_shift(y=data, sr=sample_rate, n_steps=2), sample_rate)); y.append(emotion)
                X.append(extract_features(librosa.effects.pitch_shift(y=data, sr=sample_rate, n_steps=-2), sample_rate)); y.append(emotion)
                X.append(extract_features(librosa.effects.time_stretch(y=data, rate=1.2), sample_rate)); y.append(emotion)
                X.append(extract_features(librosa.effects.time_stretch(y=data, rate=0.8), sample_rate)); y.append(emotion)
                
                processed_files += 1
            except Exception as e:
                print(f"WARNING: File '{file}' bypassed due to processing error: {e}")
                
    X, y = np.array(X), np.array(y)
    
    # Save the processed cache to disk
    np.save('X_features_ultimate.npy', X)
    np.save('y_labels_ultimate.npy', y)
    print(f"INFO: Successfully processed and exported {processed_files * 6} augmented data samples to local cache.")
    
    return X, y

# --- MAIN EXECUTION BLOCK ---
X, y = load_data('allVoices', 'MetaData.xlsx') 

if len(X) == 0: 
    print("CRITICAL ERROR: Dataset is empty. Execution terminated.")
    exit()

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("INFO: Initializing and training the Ensemble Learning Classifier (LightGBM + XGBoost + Random Forest)...")

lgb_model = lgb.LGBMClassifier(objective='multiclass', learning_rate=0.03, num_leaves=64, max_depth=12, n_estimators=700, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=42, verbosity=-1)
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=5, learning_rate=0.05, max_depth=8, n_estimators=500, subsample=0.8, colsample_bytree=0.8, random_state=42)
rf_model = RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', random_state=42)

ensemble_model = VotingClassifier(estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('rf', rf_model)], voting='soft')
ensemble_model.fit(X_train, y_train)

# Export the trained model and encoder for Phase 3 deployment
joblib.dump(ensemble_model, 'ultimate_emotion_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("PHASE 2 - ADVANCED ENSEMBLE CLASSIFICATION RESULTS")
print("="*50)
print(f"Overall Model Accuracy: {(accuracy * 100):.2f}%")
print("\nDetailed Performance Metrics:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("="*50)