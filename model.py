import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(None, 1)):
    """
    Builds a temporal convolutional autoencoder for audio denoising.
    The model uses 1D convolutions to capture temporal features in the audio waveform.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    
    # Latent space representation
    encoded = layers.MaxPooling1D(2, padding='same')(x)

    # Decoder
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)

    # Output layer
    # We use a single filter to reconstruct the mono audio signal
    decoded = layers.Conv1D(1, 3, activation='tanh', padding='same')(x)

    # Note: Depending on input size and pooling/upsampling, shapes might mismatch slightly.
    # In a production environment, we would handle padding more carefully or use fixed input sizes.
    # For this demo, we assume the input length is divisible by 8 (2^3 pooling operations).
    
    model = models.Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    
    return model

if __name__ == "__main__":
    model = build_model(input_shape=(8000, 1))
    model.summary()
