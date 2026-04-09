import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os


class EEGPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)

    def extract_frequency_features(self, signal):
        """Extract frequency domain features from EEG signals"""
        fft = np.fft.fft(signal)
        fft_magnitude = np.abs(fft)
        total_power = np.sum(fft_magnitude ** 2)

        if total_power == 0:
            return [0, 0, 0, 0, 0]

        n = len(fft_magnitude)

        delta = np.sum(fft_magnitude[:int(0.1*n)]**2) / total_power
        theta = np.sum(fft_magnitude[int(0.1*n):int(0.2*n)]**2) / total_power
        alpha = np.sum(fft_magnitude[int(0.2*n):int(0.3*n)]**2) / total_power
        beta = np.sum(fft_magnitude[int(0.3*n):int(0.6*n)]**2) / total_power
        gamma = np.sum(fft_magnitude[int(0.6*n):]**2) / total_power

        return [delta, theta, alpha, beta, gamma]


def create_sample_data():
    """Generate balanced synthetic EEG dataset"""

    np.random.seed(42)

    n_samples = 1000
    n_features = 178

    X = np.random.randn(n_samples, n_features)

    # Balanced labels
    y = np.zeros(n_samples)

    seizure_indices = np.random.choice(n_samples, n_samples // 2, replace=False)

    y[seizure_indices] = 1

    # Add seizure-like patterns
    for idx in seizure_indices:
        X[idx] += 2 * np.random.randn(n_features)
        X[idx, :50] += 3 * np.sin(np.linspace(0, 10*np.pi, 50))

    return X, y


def train_models():

    print("Training EEG classification models...")

    X, y = create_sample_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=50)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.3f}")

    os.makedirs('models', exist_ok=True)

    model_data = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'accuracy': accuracy
    }

    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Model saved to models/trained_model.pkl")

    return model_data


if __name__ == "__main__":
    train_models()