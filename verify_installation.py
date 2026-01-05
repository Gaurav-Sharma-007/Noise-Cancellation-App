import os
import tensorflow as tf
import numpy as np
import librosa
from model import build_model
from inference import preprocess_audio

def verify_setup():
    print("Checking dependencies...")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Librosa Version: {librosa.__version__}")
    
    print("\nVerifying Model Build...")
    try:
        model = build_model(input_shape=(8000, 1))
        # model.summary()
        print("Model built successfully.")
    except Exception as e:
        print(f"FAILED to build model: {e}")
        return

    print("\nVerifying Inference Flow (Mock Data)...")
    try:
        # Create dummy audio data (1 second of random noise)
        dummy_audio = np.random.uniform(-1, 1, 22050)
        
        # Preprocess
        processed_input, _, target_len = preprocess_audio(dummy_audio)
        print(f"Preprocessed shape: {processed_input.shape}")
        
        # Predict
        output = model.predict(processed_input, verbose=0)
        print(f"Output shape: {output.shape}")
        
        print("Inference flow verified.")
    except Exception as e:
        print(f"FAILED validation inference: {e}")
        return

    print("\n✅ Verification Complete! The app is ready to run.")

if __name__ == "__main__":
    verify_setup()
