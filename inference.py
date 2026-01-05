import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from model import build_model
import os

def load_audio(file_path, target_sr=22050):
    """
    Load an audio file and resample it to the target sample rate.
    Returns the audio time series and the sample rate.
    """
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def preprocess_audio(audio):
    """
    Preprocess audio for the model.
    1. Normalize to [-1, 1]
    2. Reshape to (batch, time_steps, channels)
    3. Pad if necessary to be divisible by 8 (due to 3 pooling layers)
    """
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        
    # Pad to make length divisible by 8
    pad_len = (8 - (len(audio) % 8)) % 8
    audio = np.pad(audio, (0, pad_len), mode='constant')
    
    # Reshape for model: (1, length, 1)
    audio_input = audio.reshape(1, -1, 1)
    return audio_input, max_val, len(audio) - pad_len

def denoise_audio(input_path, output_path=None):
    """
    End-to-end denoising function.
    """
    # Load audio
    audio, sr = load_audio(input_path)
    
    # Preprocess
    audio_input, max_val, original_length = preprocess_audio(audio)
    
    # Initialize model
    # In a real scenario, we would load weights here: model.load_weights('denoiser_weights.h5')
    # For this demo, we initialize a fresh model with random weights (acting as a pass-through/filter)
    model = build_model(input_shape=(None, 1))
    
    # Run inference
    # Note: Processing a very long file at once effectively uses a massive batch size in time dimension.
    # For very long files, we might want to chunk it. For this demo, full file is fine for short clips.
    denoised_output = model.predict(audio_input)
    
    # Post-process
    denoised_audio = denoised_output.reshape(-1)
    denoised_audio = denoised_audio[:original_length] # Trim padding
    denoised_audio = denoised_audio * max_val # Restore amplitude
    
    # Save output if path is provided
    if output_path:
        sf.write(output_path, denoised_audio, sr)
        
    return audio, denoised_audio, sr

if __name__ == "__main__":
    # Test stub
    pass
