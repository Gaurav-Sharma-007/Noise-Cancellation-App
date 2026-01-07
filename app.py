import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="SonicPure - Premium Noise Cancellation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show loading status immediately
with st.spinner("Initializing AI engine..."):
    import numpy as np
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import soundfile as sf
    from inference import denoise_audio

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

# Navigation
page = st.sidebar.radio("Navigation", ["Home", "Training Dashboard"])

if page == "Training Dashboard":
    st.markdown("### 📊 Model Training Metrics")
    
    METRICS_PATH = "training_metrics.json"
    if os.path.exists(METRICS_PATH):
        try:
            import json
            import pandas as pd
            
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
                
            # Key Metrics
            col1, col2, col3 = st.columns(3)
            eval_data = metrics.get("evaluation", {})
            history = metrics.get("history", {})
            
            with col1:
                st.metric("Final Test Loss", f"{eval_data.get('test_loss', 0):.4f}")
            with col2:
                st.metric("SNR Improvement", f"{eval_data.get('snr_improvement', 0):.2f} dB")
            with col3:
                st.metric("Final SNR", f"{eval_data.get('final_snr', 0):.2f} dB") # Changed to Final SNR
                
            st.markdown("---")
            
            # Loss Curve
            st.subheader("📉 Training Loss Curve")
            if 'loss' in history and 'val_loss' in history:
                chart_data = pd.DataFrame({
                    'Training Loss': history['loss'],
                    'Validation Loss': history['val_loss']
                })
                st.line_chart(chart_data)
                st.caption("Lower loss is better. Convergence indicates successful learning.")
            else:
                st.info("No detailed history available.")
                
            # Explanation
            st.info("""
            **Metrics Explanation:**
            - **SNR (Signal-to-Noise Ratio):** Measures signal quality. Higher is better.
            - **Loss (MAE/MSE):** Measures reconstruction error. Lower is better.
            - **Validation Loss:** Performance on unseen 'Test' data.
            """)
            
        except Exception as e:
            st.error(f"Error reading metrics: {e}")
    else:
        st.warning("No training data found.")
        st.markdown("Run `python train.py` after adding data to `dataset/` to generate metrics.")

    # Stop execution for this page
    st.stop()

# Home Page Logic (Existing)
# ...
    st.header("Settings")
    import os
    if os.path.exists("denoiser_weights.h5"):
        st.success("Mode: **Deep Learning (U-Net)**")
        st.caption("Using your custom trained model.")
    else:
        st.warning("Mode: **Spectral Gating (DSP)**")
        st.caption("No trained model found. Using fallback.")
        st.markdown("To train: put data in `dataset/` and run `python train.py`")
        
    st.markdown("---")
    st.write("Supported Formats: MP3, WAV, FLAC, WEBM")
    st.markdown("---")
    st.caption("Powered by TensorFlow & Librosa")

# Main Content
uploaded_file = st.file_uploader("Upload your audio file to begin", type=['mp3', 'wav', 'flac', 'webm'])

if uploaded_file is not None:
    st.markdown("### 🎧 Analysis & Processing")
    
    col1, col2 = st.columns(2)
    
    # Save uploaded file temporarily for processing
    file_ext = os.path.splitext(uploaded_file.name)[1]
    temp_input_path = f"temp_input{file_ext}"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with col1:
        st.markdown('<div class="uploaded-file">', unsafe_allow_html=True)
        st.subheader("Original Audio")
        st.audio(uploaded_file)
        
        # Visualize Original
        with st.expander("View Waveform", expanded=True):
            try:
                # Load with librosa
                y, sr = librosa.load(temp_input_path, sr=None)
                
                fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
                # Try simple plot first to ensure it works
                times = np.linspace(0, len(y) / sr, num=len(y))
                # Downsample for performance/rendering if file is large
                if len(y) > 10000:
                    step = len(y) // 10000
                    y_plot = y[::step]
                    times_plot = times[::step]
                else:
                    y_plot = y
                    times_plot = times
                
                ax.plot(times_plot, y_plot, color='#FF5252', alpha=0.8)
                ax.set_facecolor('#0E1117')
                ax.axis('off')
                # Tight layout to remove white margins
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate waveform: {e}")
                st.caption("Tip: For MP3/WebM, ensure FFmpeg is installed.")
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

