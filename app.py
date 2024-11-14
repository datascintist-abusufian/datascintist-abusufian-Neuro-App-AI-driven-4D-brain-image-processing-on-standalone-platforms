import io
from datetime import datetime
import requests  
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import time
from skimage import exposure
from skimage.measure import label
import urllib.parse

# Set up page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
    .stTabs [data-baseweb="tab"] {padding: 1rem;}
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stAlert {padding: 1rem; margin-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config():
    BASE_GITHUB_URL = "https://github.com/datascintist-abusufian/datascintist-abusufian-Neuro-App-AI-driven-4D-brain-image-processing-on-standalone-platforms/blob/main/"
    return {
        'MODEL_PATH': f"{BASE_GITHUB_URL}BrainTumor10EpochsCategorical.h5?raw=true",
        'GIF_PATH': f"{BASE_GITHUB_URL}TAC_Brain_tumor_glioblastoma-Transverse_plane.gif?raw=true",
        'YES_IMAGES_DIR': BASE_GITHUB_URL + "test_images/yes/",
        'NO_IMAGES_DIR': BASE_GITHUB_URL + "test_images/no/"
    }

@st.cache_resource
def download_model(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open('model.h5', 'wb') as f:
                f.write(response.content)
            return 'model.h5'
        else:
            st.error(f"Failed to download model: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return None

@st.cache_resource
def load_cached_model(model_url):
    model_path = download_model(model_url)
    if model_path:
        return load_model(model_path)
    else:
        st.error("Model loading failed.")
        return None

# Calculate advanced metrics
def calculate_advanced_metrics(img_array):
    gray_img = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    metrics = {
        'Mean Intensity': np.mean(gray_img),
        'Std Intensity': np.std(gray_img),
        'Dynamic Range': np.ptp(gray_img)
    }
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    metrics.update({
        'Contrast': np.mean(graycoprops(glcm, 'contrast')),
        'Homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
        'Energy': np.mean(graycoprops(glcm, 'energy')),
        'Correlation': np.mean(graycoprops(glcm, 'correlation')),
        'Dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
        'ASM': np.mean(graycoprops(glcm, 'ASM'))
    })
    lbp = local_binary_pattern(gray_img, 8 * 3, 3, method='uniform')
    metrics['LBP Variance'] = np.var(lbp)
    return metrics

# Sensitivity Analysis
def perform_sensitivity_analysis(model, image_array, n_iterations=10):
    orig_pred = model.predict(image_array, verbose=0)
    orig_result = np.argmax(orig_pred[0])
    orig_confidence = orig_pred[0][orig_result] * 100
    noise_impacts, blur_impacts, confidences = [], [], []
    
    for i in range(n_iterations):
        noise_level = i / (n_iterations * 2)
        noisy_image = np.clip(image_array + np.random.normal(0, noise_level, image_array.shape), 0, 1)
        noise_pred = model.predict(noisy_image, verbose=0)
        noise_impacts.append(np.abs(noise_pred[0][orig_result] - orig_pred[0][orig_result]) * 100)
        confidences.append(noise_pred[0][orig_result] * 100)
        
        kernel_size = 2 * i + 1
        blurred_image = cv2.GaussianBlur(image_array[0], (kernel_size, kernel_size), 0)
        blurred_image = np.expand_dims(blurred_image, 0)
        blur_pred = model.predict(blurred_image, verbose=0)
        blur_impacts.append(np.abs(blur_pred[0][orig_result] - orig_pred[0][orig_result]) * 100)

    stability = 100 - (np.mean(noise_impacts + blur_impacts))
    
    plt.figure(figsize=(10, 5))
    plt.plot(confidences, label='Confidence Under Noise', marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Confidence (%)')
    plt.legend()
    st.pyplot(plt)
    
    return pd.DataFrame({
        'Iteration': range(1, n_iterations + 1),
        'Noise Impact (%)': noise_impacts,
        'Blur Impact (%)': blur_impacts,
        'Confidence (%)': confidences
    })

# Visualization Functions
def plot_intensity_profile(img_array):
    center_row = img_array[img_array.shape[0] // 2, :]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=center_row, mode='lines', name='Intensity Profile'))
    fig.update_layout(
        title='Intensity Profile Across Center',
        xaxis_title='Position',
        yaxis_title='Intensity'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_region_segmentation(img_array):
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled_img = label(binary)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_array, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(labeled_img, cmap='nipy_spectral')
    ax2.set_title('Segmented Regions')
    st.pyplot(fig)

def create_3d_visualization(img_array):
    fig = go.Figure(data=[go.Surface(z=img_array)])
    fig.update_layout(
        title='3D Surface Plot of MRI Intensities',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Intensity'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ImageAnalyzer Class
class ImageAnalyzer:
    def __init__(self):
        self.config = load_config()
        self.model = load_cached_model(self.config['MODEL_PATH'])
    
    def process_image(self, img):
        img = img.convert('RGB').resize((64, 64))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def predict(self, img_array):
        start_time = time.time()
        prediction = self.model.predict(img_array, verbose=0)
        result = np.argmax(prediction[0])
        confidence = prediction[0][result] * 100
        inference_time = time.time() - start_time
        return result, confidence, inference_time

    def display_slice_views(self, image_array):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('Original View')
        
        enhanced = exposure.equalize_adapthist(image_array)
        axes[1].imshow(enhanced, cmap='gray')
        axes[1].set_title('Enhanced Contrast')
        
        edges = cv2.Canny(image_array, 100, 200)
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title('Edge Detection')
        
        plt.tight_layout()
        st.pyplot(fig)

def main():
    config = load_config()
    
    st.image(config['GIF_PATH'], width=800)
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    with st.sidebar:
        st.title("🧠 Brain Tumor Detection")
        input_method = st.radio("Select Input Method", ["Upload Image", "Use Sample Images"])
        
        if input_method == "Upload Image":
            selected_file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])
        else:
            GITHUB_BASE_URL = config['YES_IMAGES_DIR']
            demo_images = {
                "Tumor Cases": [
                    "Y1.jpg",
                    "Y2.jpg",
                    "Y3.jpg",
                    "Y4.jpg",
                ],
                "Normal Cases": [
                    "1 no.jpeg",
                    "3 no.jpeg"
                    "4 no.jpeg"
                    "5 no.jpeg"
                ]
            }
            case_type = st.selectbox("Select case type:", ["Tumor Cases", "Normal Cases"])
            selected_demo = st.selectbox("Choose a sample image:", demo_images[case_type])
            
            folder = "yes" if case_type == "Tumor Cases" else "no"
            encoded_filename = urllib.parse.quote(selected_demo)
            selected_file = f"{config['YES_IMAGES_DIR'] if folder == 'yes' else config['NO_IMAGES_DIR']}{encoded_filename}?raw=true"
        
        st.subheader("📊 System Metrics")
        total_analyses = len(st.session_state.analysis_history)
        successful_analyses = sum(1 for x in st.session_state.analysis_history if x.get('success', False))
        success_rate = (successful_analyses / total_analyses) * 100 if total_analyses > 0 else 0
        st.metric("Total Analyses", total_analyses)
        st.metric("Success Rate", f"{success_rate:.1f}%")

    st.title("4D AI Driven Neuro App - Advanced Analytics")
    
    if selected_file:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Main Analysis",
            "📊 Advanced Metrics",
            "📈 Sensitivity Analysis",
            "📜 Historical Data",
            "🔍 Advanced Visualizations"
        ])
        
        try:
            if isinstance(selected_file, str):
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(selected_file, headers=headers)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                else:
                    st.error(f"Failed to load image from URL: Status code {response.status_code}")
                    return
            else:
                image = Image.open(selected_file)
            
            analyzer = ImageAnalyzer()
            img_array = analyzer.process_image(image)
            result, confidence, inference_time = analyzer.predict(img_array)
            
            with tab1:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Input MRI Image", use_container_width=True)
                    st.metric("Processing Time", f"{inference_time:.3f}s")
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={'text': "Detection Confidence"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if result == 1 else "green"},
                            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 70}
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    if result == 1:
                        st.error("🚨 Tumor Detected")
                        st.warning("Please consult with a medical professional.")
                    else:
                        st.success("✅ No Tumor Detected")
                        st.info("Regular check-ups are recommended.")
            
            with tab2:
                st.header("📊 Advanced Metrics")
                metrics = calculate_advanced_metrics(img_array[0])
                metrics_df = pd.DataFrame(metrics, index=[0])
                st.table(metrics_df.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]))

                plt.figure(figsize=(8, 5))
                ax = sns.barplot(x=metrics_df.columns, y=metrics_df.iloc[0])
                plt.xticks(rotation=45)
                plt.title("Advanced Metrics")
                for p in ax.patches:
                    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                              ha='center', va='bottom')
                st.pyplot(plt)
            
            with tab3:
                st.header("📈 Sensitivity Analysis")
                analysis_df = perform_sensitivity_analysis(analyzer.model, img_array)
                st.table(analysis_df.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]))
            
            with tab4:
                st.header("📜 Historical Data")
                if st.session_state.analysis_history:
                    historical_df = pd.DataFrame(st.session_state.analysis_history)
                    st.table(historical_df.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]))
            
            with tab5:
                st.header("Advanced MRI Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Multi-View Analysis")
                    gray_img = cv2.cvtColor((img_array[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    analyzer.display_slice_views(gray_img)
                    
                    st.subheader("Intensity Profile")
                    plot_intensity_profile(gray_img)
                
                with col2:
                    st.subheader("Region Segmentation")
                    show_region_segmentation(gray_img)
                    
                    st.subheader("3D Visualization")
                    create_3d_visualization(gray_img)

                st.markdown("""
                ### Visualization Explanations
                - **Multi-View Analysis**: Shows original, contrast-enhanced, and edge-detected views
                - **Intensity Profile**: Shows intensity values across the center of the image
                - **Region Segmentation**: Identifies and labels distinct regions in the image
                - **3D Visualization**: Shows intensity values as a 3D surface plot
                """)
            
            st.session_state.analysis_history.append({
                "timestamp": datetime.now(),
                "result": int(result) if result is not None else None,
                "confidence_score": float(confidence) if confidence is not None else None,
                "inference_time": float(inference_time) if inference_time is not None else None,
                "success": True if result is not None else False
            })
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    else:
        st.info("👆 Please select an input method and image to begin analysis.")

if __name__ == "__main__":
    main()
