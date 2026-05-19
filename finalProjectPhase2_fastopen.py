import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("INFO: Initializing fast execution module...")

# Önce modeli ve encoder'ı (tercümanı) yüklüyoruz
print("INFO: Loading the pre-trained Ensemble Learning Classifier and Encoder (PKL format)...")
try:
    ensemble_model = joblib.load('ultimate_emotion_model.pkl')
    le = joblib.load('label_encoder.pkl')
except Exception as e:
    print(f"CRITICAL ERROR: Model files (.pkl) not found. Please ensure the training phase is complete. Details: {e}")
    exit()

print("INFO: Loading pre-compiled feature cache into memory...")
try:
    X = np.load('X_features_ultimate.npy')
    y_strings = np.load('y_labels_ultimate.npy') # Diskteki veriler kelime olarak (Happy, Sad vb.) duruyor
    
    # İŞTE ÇÖZÜM: Kelimeleri modelin anlayacağı sayılara (0,1,2..) çeviriyoruz
    y_encoded = le.transform(y_strings)
except Exception as e:
    print(f"CRITICAL ERROR: Cache files (.npy) not found. Please execute 'finalProjectPhase2_train.py' first. Details: {e}")
    exit()

# Train-Test ayırımını eğitimdekiyle birebir aynı yapıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("INFO: Executing predictions on the test dataset...")
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("PHASE 2 - RAPID MODEL EVALUATION RESULTS")
print("="*50)
print(f"Overall Model Accuracy: {(accuracy * 100):.2f}%")
print("\nDetailed Performance Metrics:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("="*50)