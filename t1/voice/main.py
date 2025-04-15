import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

# --- Step 1: Preprocess Audio ---
def preprocess_audio(file_path, sr=16000, duration=3):
    """
    Preprocess audio to a fixed duration and sampling rate.
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    audio = librosa.util.fix_length(audio, sr * duration)
    return audio

# --- Step 2: Extract Features ---
def extract_mfcc(audio, sr=16000, n_mfcc=40):
    """
    Extract MFCC features from audio.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# --- Step 3: Prepare Dataset ---
def load_dataset(data_dir, sr=16000, duration=3):
    """
    Load audio dataset, preprocess, and extract features.
    """
    features, labels = [], []
    label_map = {}
    current_label = 0

    for speaker in os.listdir(data_dir):
        speaker_dir = os.path.join(data_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        # Map speaker name to a numeric label
        if speaker not in label_map:
            label_map[speaker] = current_label
            current_label += 1

        for file in os.listdir(speaker_dir):
            file_path = os.path.join(speaker_dir, file)
            try:
                audio = preprocess_audio(file_path, sr, duration)
                feature = extract_mfcc(audio, sr)
                features.append(feature)
                labels.append(label_map[speaker])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(features), np.array(labels), label_map

# --- Step 4: Build Model ---
def build_model(input_shape, num_classes):
    """
    Build a simple feedforward neural network for speaker recognition.
    """
    model = Sequential([
        Dense(128, activation="relu", input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# --- Main Program ---
if __name__ == "__main__":
    # Path to your dataset
    DATA_DIR = "path/to/your/audio/dataset"  # Structure: DATA_DIR/speaker_name/*.wav

    # Load dataset
    print("Loading dataset...")
    X, y, label_map = load_dataset(DATA_DIR)
    print(f"Dataset loaded. Number of samples: {len(X)}, Number of speakers: {len(label_map)}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    print("Building model...")
    input_shape = (X.shape[1],)  # Shape of feature vector
    num_classes = len(label_map)  # Number of speakers
    model = build_model(input_shape, num_classes)

    # Train model
    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    # Save model
    model.save("speaker_recognition_model.h5")
    print("Model saved.")

    # Evaluate model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Predict on a new sample
    print("Predicting on a new sample...")
    test_audio = preprocess_audio("path/to/new_audio.wav")
    test_feature = extract_mfcc(test_audio).reshape(1, -1)
    prediction = model.predict(test_feature)
    predicted_label = np.argmax(prediction)
    speaker_name = list(label_map.keys())[list(label_map.values()).index(predicted_label)]
    print(f"Predicted Speaker: {speaker_name}")
