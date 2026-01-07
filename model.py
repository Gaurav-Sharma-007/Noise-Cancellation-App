import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def build_model(input_shape=(None, 1)):
    """
    Builds a U-Net based Deep Learning model for audio denoising.
    U-Net has skip connections that allow it to preserve high-frequency details 
    much better than a standard Autoencoder.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Encoder (Downsampling) ---
    # Block 1
    c1 = layers.Conv1D(32, 15, activation='relu', padding='same')(inputs)
    c1 = layers.Conv1D(32, 15, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling1D(2)(c1)

    # Block 2
    c2 = layers.Conv1D(64, 15, activation='relu', padding='same')(p1)
    c2 = layers.Conv1D(64, 15, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling1D(2)(c2)

    # Block 3
    c3 = layers.Conv1D(128, 15, activation='relu', padding='same')(p2)
    c3 = layers.Conv1D(128, 15, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling1D(2)(c3)

    # --- Bottleneck ---
    c4 = layers.Conv1D(256, 15, activation='relu', padding='same')(p3)
    c4 = layers.Conv1D(256, 15, activation='relu', padding='same')(c4)

    # --- Decoder (Upsampling) ---
    # Block 3 Up
    u3 = layers.UpSampling1D(2)(c4)
    u3 = layers.Conv1D(128, 15, activation='relu', padding='same')(u3)
    u3 = layers.Concatenate()([u3, c3]) # Skip Connection
    c5 = layers.Conv1D(128, 15, activation='relu', padding='same')(u3)
    c5 = layers.Conv1D(128, 15, activation='relu', padding='same')(c5)

    # Block 2 Up
    u2 = layers.UpSampling1D(2)(c5)
    u2 = layers.Conv1D(64, 15, activation='relu', padding='same')(u2)
    u2 = layers.Concatenate()([u2, c2]) # Skip Connection
    c6 = layers.Conv1D(64, 15, activation='relu', padding='same')(u2)
    c6 = layers.Conv1D(64, 15, activation='relu', padding='same')(c6)

    # Block 1 Up
    u1 = layers.UpSampling1D(2)(c6)
    u1 = layers.Conv1D(32, 15, activation='relu', padding='same')(u1)
    u1 = layers.Concatenate()([u1, c1]) # Skip Connection
    c7 = layers.Conv1D(32, 15, activation='relu', padding='same')(u1)
    c7 = layers.Conv1D(32, 15, activation='relu', padding='same')(c7)

    # Output Layer
    # We use tanh activation because audio is normalized to [-1, 1]
    outputs = layers.Conv1D(1, 1, activation='tanh', padding='same')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mae') # MAE often works better for audio than MSE

    return model

if __name__ == "__main__":
    # Test model shape with dummy data
    model = build_model(input_shape=(16384, 1)) # 16384 is a power of 2, good for U-Net
    model.summary()
