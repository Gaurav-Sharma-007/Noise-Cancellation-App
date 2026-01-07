import os
import numpy as np
import librosa
import tensorflow as tf
from model import build_model
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

# --- Configuration ---
CLEAN_DIR = "dataset/clean"
NOISY_DIR = "dataset/noisy"
MODEL_SAVE_PATH = "denoiser_weights.h5"
METRICS_SAVE_PATH = "training_metrics.json"
SAMPLE_RATE = 22050
CHUNK_DURATION = 1.0 # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION) # 22050 samples per chunk
BATCH_SIZE = 16
EPOCHS = 50

def load_and_chunk_files(directory):
    """
    Loads all audio files in a directory and splits them into fixed-size chunks.
    """
    chunks = []
    if not os.path.exists(directory):
        return np.array(chunks)
        
    files = [f for f in os.listdir(directory) if f.endswith(('.wav', '.mp3', '.flac'))]
    files.sort() # Ensure matching order if filenames match
    
    print(f"Loading files from {directory}...")
    for filename in tqdm(files):
        path = os.path.join(directory, filename)
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE)
            
            # Create chunks
            for i in range(0, len(audio) - CHUNK_SAMPLES, CHUNK_SAMPLES):
                chunk = audio[i:i+CHUNK_SAMPLES]
                chunks.append(chunk)
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return np.array(chunks)

def calculate_snr(clean, noisy):
    """
    Calculate Signal-to-Noise Ratio (dB).
    Higher is better.
    """
    noise = noisy - clean
    p_signal = np.sum(clean**2, axis=1)
    p_noise = np.sum(noise**2, axis=1)
    
    # Avoid div by zero
    p_noise = np.maximum(p_noise, 1e-10)
    
    snr = 10 * np.log10(p_signal / p_noise)
    return np.mean(snr)

def train():
    # 1. Load Data
    print("Loading Dataset using 'given data'...")
    clean_audio = load_and_chunk_files(CLEAN_DIR)
    noisy_audio = load_and_chunk_files(NOISY_DIR)

    # 2. Validation
    if len(clean_audio) == 0 or len(noisy_audio) == 0:
        print("ERROR: No Audio files found!")
        print(f"Please place clean audio in: {os.path.abspath(CLEAN_DIR)}")
        print(f"Please place noisy audio in: {os.path.abspath(NOISY_DIR)}")
        return

    if len(clean_audio) != len(noisy_audio):
        print(f"WARNING: Number of chunks mismatch! Clean: {len(clean_audio)}, Noisy: {len(noisy_audio)}")
        print("Truncating to smaller size to match...")
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

    # 3. Preprocess
    clean_audio = clean_audio.reshape(-1, CHUNK_SAMPLES, 1)
    noisy_audio = noisy_audio.reshape(-1, CHUNK_SAMPLES, 1)

    print(f"Total Samples: {len(clean_audio)}")
    
    # 4. Split Data (Train vs Test)
    print("Splitting into Training and Testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(noisy_audio, clean_audio, test_size=0.2, random_state=42)
    
    print(f"Training Set: {x_train.shape}")
    print(f"Testing Set: {x_test.shape}")

    # 5. Build Model
    model = build_model(input_shape=(CHUNK_SAMPLES, 1))
    
    # 6. Train
    print("Starting Training...")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, 
        monitor='val_loss', 
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, 
        y_train, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        validation_data=(x_test, y_test),
        callbacks=[checkpoint, early_stop]
    )
    
    # 7. Evaluate Performance
    print("Calculating Accuracy Metrics...")
    
    # Predict on test set
    y_pred = model.predict(x_test)
    
    # Calculate Metrics
    # Initial SNR (Noisy vs Clean)
    initial_snr = calculate_snr(y_test.squeeze(), x_test.squeeze())
    # Final SNR (Denoised vs Clean)
    final_snr = calculate_snr(y_test.squeeze(), y_pred.squeeze())
    
    print(f"Average Initial SNR: {initial_snr:.2f} dB")
    print(f"Average Denoised SNR: {final_snr:.2f} dB")
    print(f"SNR Improvement: {final_snr - initial_snr:.2f} dB")

    # Save Metrics for Dashboard
    metrics = {
        "history": history.history,
        "evaluation": {
            "initial_snr": float(initial_snr),
            "final_snr": float(final_snr),
            "snr_improvement": float(final_snr - initial_snr),
            "test_loss": float(history.history['val_loss'][-1])
        }
    }
    
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics, f)
        
    print(f"Training Complete. Metrics saved to {METRICS_SAVE_PATH}")
