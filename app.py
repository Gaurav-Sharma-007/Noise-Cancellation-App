import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from inference import denoise_audio

# Set page configuration
st.set_page_config(
    page_title="SonicPure - Premium Noise Cancellation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        color: #4CAF50; /* Green accent */
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 400;
        font-size: 1.5rem;
        color: #B0BEC5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .uploaded-file {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">SonicPure</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Noise Cancellation Studio</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.info("Currently using Pre-trained Conv1D Autoencoder.")
    st.markdown("---")
    st.write("Supported Formats: MP3, WAV, FLAC")
    st.markdown("---")
    st.caption("Powered by TensorFlow & Librosa")

# Main Content
uploaded_file = st.file_uploader("Upload your audio file to begin", type=['mp3', 'wav', 'flac'])

if uploaded_file is not None:
    st.markdown("### 🎧 Analysis & Processing")
    
    col1, col2 = st.columns(2)
    
    # Save uploaded file temporarily for processing
    temp_input_path = "temp_input.wav"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with col1:
        st.markdown('<div class="uploaded-file">', unsafe_allow_html=True)
        st.subheader("Original Audio")
        st.audio(uploaded_file)
        
        # Visualize Original
        with st.expander("View Waveform", expanded=True):
            y, sr = librosa.load(temp_input_path, sr=None)
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
            librosa.display.waveshow(y, sr=sr, alpha=0.8, ax=ax, color='#FF5252')
            ax.set_facecolor('#0E1117')
            ax.axis('off')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Process Button
    if st.button("✨ Denoise Audio", use_container_width=True):
        with st.spinner("AI is removing noise... Please wait."):
            try:
                temp_output_path = "temp_output.wav"
                
                # Call inference
                _, _, output_sr = denoise_audio(temp_input_path, temp_output_path)
                
                with col2:
                    st.markdown('<div class="uploaded-file">', unsafe_allow_html=True)
                    st.subheader("Cleaned Audio")
                    st.audio(temp_output_path)
                    
                    # Visualize Output
                    with st.expander("View Waveform", expanded=True):
                        y_out, _ = librosa.load(temp_output_path, sr=output_sr)
                        fig_out, ax_out = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
                        librosa.display.waveshow(y_out, sr=output_sr, alpha=0.8, ax=ax_out, color='#4CAF50')
                        ax_out.set_facecolor('#0E1117')
                        ax_out.axis('off')
                        st.pyplot(fig_out)
                    
                    # Download Button
                    with open(temp_output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Cleaned Audio",
                            data=f,
                            file_name="cleaned_audio.wav",
                            mime="audio/wav"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                st.success("Processing Complete!")
                
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                
    st.markdown("---") 
    
    # Cleanup (Optional - handled by OS usually for temp files, but good practice)
    # os.remove(temp_input_path) 
    # if os.path.exists("temp_output.wav"):
    #    os.remove("temp_output.wav")

else:
    st.info("👆 Upload an audio file to get started.")

