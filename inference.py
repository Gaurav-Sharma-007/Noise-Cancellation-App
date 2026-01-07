import soundfile as sf
import librosa
import numpy as np
import tensorflow as tf
from model import build_model
import noisereduce as nr
import os

# Configuration matching training
CHUNK_DURATION = 1.0 
SAMPLE_RATE = 22050
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION) 
WEIGHTS_PATH = "denoiser_weights.h5"

def denoise_audio(input_path, output_path=None):
    """
    Denoise audio using logic:
    1. Try to load Trained U-Net Model.
    2. If no model found, fall back to Spectral Gating (noisereduce).
    """
    
    # Check if trained weights exist
    use_dl_model = os.path.exists(WEIGHTS_PATH)
    
    if use_dl_model:
        return denoise_with_model(input_path, output_path)
    else:
        print("No trained model found. Falling back to Spectral Gating.")
        return denoise_with_dsp(input_path, output_path)

def denoise_with_dsp(input_path, output_path):
    """Fallback: Stationary Noise Reduction"""
    data, rate = librosa.load(input_path, sr=None)
    reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=True, prop_decrease=0.90)
    
    if output_path:
        sf.write(output_path, reduced_noise, rate)
    return data, reduced_noise, rate

def denoise_with_model(input_path, output_path):
    """Primary: Deep Learning U-Net"""
    # Load model structure
    model = build_model(input_shape=(None, 1))
    # Load weights
    try:
        model.load_weights(WEIGHTS_PATH)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return denoise_with_dsp(input_path, output_path)

    # Load audio
    raw_audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)
    
    # Process in chunks to handle long files (and manage memory)
    # Note: U-Net input size must be consistent if dense layers were used, 
    # but since FCN (Fully Convolutional), we can technically process variable lengths,
    # however, we trained on 1.0s chunks. Processing full file might run out of RAM.
    # Let's chunk it.
    
    input_len = len(raw_audio)
    # Pad to nearest chunk size to ensure smooth processing
    pad_needed = (CHUNK_SAMPLES - (input_len % CHUNK_SAMPLES)) % CHUNK_SAMPLES
    audio_padded = np.pad(raw_audio, (0, pad_needed), mode='constant')
    
    processed_chunks = []
    
    # Reshape for batch processing
    # Create batch of chunks
    num_chunks = len(audio_padded) // CHUNK_SAMPLES
    chunks = audio_padded.reshape(num_chunks, CHUNK_SAMPLES, 1)
    
    # Predict batch
    denoised_chunks = model.predict(chunks, verbose=0)
    
    # Flatten output
    denoised_audio = denoised_chunks.flatten()
    
    # Remove padding
    denoised_audio = denoised_audio[:input_len]
    
    if output_path:
        sf.write(output_path, denoised_audio, sr)
        
    return raw_audio, denoised_audio, sr

if __name__ == "__main__":
    pass
