import streamlit as st
import numpy as np
from PIL import Image
import time
import plotly.graph_objects as go
import plotly.express as px
import cv2
import tempfile
import os
from model_utils import DogCatClassifier

# Page config
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize classifier
@st.cache_resource
def load_classifier():
    """Load model with caching"""
    try:
        classifier = DogCatClassifier()
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# App header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🐶 Dog vs Cat Classifier 🐱</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Teachable Machine & TensorFlow Lite")

# Sidebar
with st.sidebar:
    st.image("https://www.teachablemachine.withgoogle.com/assets/img/logo-tm.svg", width=200)
    st.title("Settings")
    
    # Model info
    classifier = load_classifier()
    if classifier:
        model_info = classifier.get_model_info()
        st.subheader("Model Information")
        st.write(f"**Input Size:** {model_info['input_shape'][0]}x{model_info['input_shape'][1]}")
        st.write(f"**Labels:** {', '.join(model_info['labels'])}")
        st.write(f"**Model Size:** {model_info['model_size']}")
    
    st.divider()
    
    # Confidence threshold
    st.subheader("Confidence Settings")
    confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.7,
        step=0.01
    )
    
    st.divider()
    
    # About section
    st.subheader("About")
    st.info("""
    This app uses a model trained with Google's Teachable Machine.
    
    **How to use:**
    1. Upload an image of a dog or cat
    2. Or take a picture with your webcam
    3. Click 'Classify' to see results
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Webcam", "📊 Batch Test"])

with tab1:
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of a dog or cat"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_path = tmp_file.name
            
            # Classify button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Predict
                        predicted_class, confidence, probs = classifier.predict_from_file(temp_path)
                        
                        # Display results
                        with col2:
                            st.markdown("### Classification Results")
                            
                            # Prediction card
                            with st.container():
                                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                                
                                # Emoji based on prediction
                                emoji = "🐶" if predicted_class == "dog" else "🐱"
                                
                                # Confidence color
                                if confidence >= 0.9:
                                    conf_class = "confidence-high"
                                elif confidence >= 0.7:
                                    conf_class = "confidence-medium"
                                else:
                                    conf_class = "confidence-low"
                                
                                st.markdown(f"### {emoji} **Prediction:** {predicted_class.upper()}")
                                st.markdown(f"### <span class='{conf_class}'>Confidence: {confidence:.2%}</span>", unsafe_allow_html=True)
                                
                                # Progress bar
                                st.progress(float(confidence))
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Probability chart
                            st.subheader("Probability Distribution")
                            
                            # Bar chart với Plotly
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(probs.keys()),
                                    y=list(probs.values()),
                                    text=[f"{v:.2%}" for v in probs.values()],
                                    textposition='auto',
                                    marker_color=['#4CAF50' if k == predicted_class else '#FF9800' for k in probs.keys()]
                                )
                            ])
                            
                            fig.update_layout(
                                yaxis=dict(range=[0, 1]),
                                yaxis_title="Probability",
                                xaxis_title="Class"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Confidence check
                            if confidence < confidence_threshold:
                                st.warning(f"⚠️ Low confidence! Below threshold ({confidence_threshold:.0%})")
                            
                            # Model interpretation
                            st.subheader("Model Interpretation")
                            if predicted_class == "dog":
                                st.success("""
                                **Dog Characteristics Detected:**
                                - Pointed ears
                                - Longer snout
                                - Different fur patterns
                                """)
                            else:
                                st.success("""
                                **Cat Characteristics Detected:**
                                - Triangular ears
                                - Shorter snout
                                - Whiskers pattern
                                """)
                    
                    except Exception as e:
                        st.error(f"Error during classification: {e}")
                    
                    finally:
                        # Cleanup temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

with tab2:
    # Webcam capture
    st.subheader("Webcam Capture")
    
    # Option 1: Use device camera
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Display captured image
        img = Image.open(camera_image)
        st.image(img, caption="Captured Image", use_column_width=True)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            img.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Classify
        if st.button("🔍 Classify Webcam Image", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    predicted_class, confidence, probs = classifier.predict_from_file(temp_path)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Prediction", predicted_class.upper())
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    with col2:
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#4CAF50"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 75], 'color': "gray"},
                                    {'range': [75, 100], 'color': "darkgray"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display emoji
                    if predicted_class == "dog":
                        st.balloons()
                        st.success("🎉 Woof! That's a dog!")
                    else:
                        st.snow()
                        st.success("🎉 Meow! That's a cat!")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

with tab3:
    # Batch testing
    st.subheader("Batch Testing")
    
    uploaded_files = st.file_uploader(
        "Upload multiple images for testing",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Run Batch Test", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                try:
                    # Save and predict
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        img = Image.open(uploaded_file)
                        img.save(tmp_file.name)
                        
                        predicted_class, confidence, probs = classifier.predict_from_file(tmp_file.name)
                        
                        results.append({
                            'filename': uploaded_file.name,
                            'prediction': predicted_class,
                            'confidence': confidence,
                            'correct': None  # Could add manual validation
                        })
                    
                    # Cleanup
                    os.unlink(tmp_file.name)
                    
                except Exception as e:
                    st.warning(f"Failed to process {uploaded_file.name}: {e}")
            
            # Display results table
            if results:
                st.subheader("Test Results")
                
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_conf = df['confidence'].mean()
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                
                with col2:
                    dog_count = len(df[df['prediction'] == 'dog'])
                    st.metric("Dogs Detected", dog_count)
                
                with col3:
                    cat_count = len(df[df['prediction'] == 'cat'])
                    st.metric("Cats Detected", cat_count)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Teachable Machine, TensorFlow Lite, and Streamlit</p>
    <p>Model trained on custom dataset | For educational purposes</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # Run with: streamlit run web_app.py
    pass
