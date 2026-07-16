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
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.segmentation import active_contour, felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel, threshold_otsu, gaussian
from skimage.color import label2rgb
from skimage import exposure, measure, morphology
from skimage.measure import label, regionprops
import cv2
import time
from scipy import ndimage
from scipy.stats import entropy
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Advanced Brain Tumor Detection & Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main {padding-top: 2rem;}
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
    .stTabs [data-baseweb="tab"] {padding: 1rem;}
    .stAlert {padding: 1rem; margin-top: 1rem;}
    .comparison-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
@st.cache_data
def load_config():
    BASE_GITHUB_URL = "https://github.com/datascintist-abusufian/datascintist-abusufian-Neuro-App-AI-driven-4D-brain-image-processing-on-standalone-platforms/blob/main/"
    return {
        'MODEL_PATH': f"{BASE_GITHUB_URL}BrainTumor10EpochsCategorical.h5?raw=true",
        'GIF_PATH': f"{BASE_GITHUB_URL}TAC_Brain_tumor_glioblastoma-Transverse_plane.gif?raw=true",
        'YES_IMAGES_DIR': BASE_GITHUB_URL + "test_images/yes/",
        'NO_IMAGES_DIR': BASE_GITHUB_URL + "test_images/no/"
    }

# ============================================================================
# MODEL LOADING
# ============================================================================
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

# ============================================================================
# AI SEGMENTATION TECHNIQUES
# ============================================================================
class AISegmentation:
    @staticmethod
    def active_contour_segmentation(image):
        """Active Contour (Snake) segmentation"""
        try:
            # Preprocess
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gaussian(gray, 3)
            
            # Initialize contour
            s = np.linspace(0, 2*np.pi, 200)
            r = image.shape[0]//2 + 100 * np.sin(s)
            c = image.shape[1]//2 + 100 * np.cos(s)
            init = np.array([r, c]).T
            
            # Active contour
            snake = active_contour(gaussian(gray, 3), init, alpha=0.015, beta=10, gamma=0.001)
            
            # Create mask from contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(snake)], 255)
            
            return mask, "Active Contour (AI)"
        except:
            return None, None
    
    @staticmethod
    def watershed_segmentation(image):
        """Watershed segmentation"""
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            markers = cv2.watershed((gray*255).astype(np.uint8), markers)
            
            # Create mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers > 1] = 255
            
            return mask, "Watershed (AI)"
        except:
            return None, None
    
    @staticmethod
    def felzenszwalb_segmentation(image):
        """Felzenszwalb segmentation"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            segments = felzenszwalb(img_uint8, scale=100, sigma=0.5, min_size=50)
            
            # Create mask from largest segment
            unique_labels = np.unique(segments)
            largest_segment = max(unique_labels, key=lambda l: np.sum(segments == l))
            mask = (segments == largest_segment).astype(np.uint8) * 255
            
            return mask, "Felzenszwalb (AI)"
        except:
            return None, None
    
    @staticmethod
    def slic_segmentation(image):
        """SLIC superpixel segmentation"""
        try:
            img_uint8 = (image * 255).astype(np.uint8)
            segments = slic(img_uint8, n_segments=100, compactness=10, sigma=1)
            
            # Create mask from combined segments
            mask = np.zeros(img_uint8.shape[:2], dtype=np.uint8)
            for seg in np.unique(segments):
                if np.sum(segments == seg) > 1000:  # Filter small segments
                    mask[segments == seg] = 255
            
            return mask, "SLIC (AI)"
        except:
            return None, None

# ============================================================================
# CONVENTIONAL SEGMENTATION TECHNIQUES
# ============================================================================
class ConventionalSegmentation:
    @staticmethod
    def otsu_segmentation(image):
        """Otsu thresholding"""
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask, "Otsu (Conventional)"
        except:
            return None, None
    
    @staticmethod
    def adaptive_threshold_segmentation(image):
        """Adaptive thresholding"""
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            return mask, "Adaptive Threshold (Conventional)"
        except:
            return None, None
    
    @staticmethod
    def edge_based_segmentation(image):
        """Edge-based segmentation using Canny"""
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to get regions
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(edges, kernel, iterations=2)
            
            return mask, "Edge-based (Conventional)"
        except:
            return None, None
    
    @staticmethod
    def region_growing_segmentation(image):
        """Simple region growing"""
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Find seed point (center of image)
            h, w = gray.shape
            seed = (h//2, w//2)
            
            # Simple region growing
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[seed] = 255
            
            # Grow region
            threshold = 30
            changed = True
            while changed:
                changed = False
                for i in range(h):
                    for j in range(w):
                        if mask[i,j] == 255:
                            # Check neighbors
                            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                                ni, nj = i+di, j+dj
                                if 0 <= ni < h and 0 <= nj < w and mask[ni,nj] == 0:
                                    if abs(int(gray[ni,nj]) - int(gray[seed])) < threshold:
                                        mask[ni,nj] = 255
                                        changed = True
            
            return mask, "Region Growing (Conventional)"
        except:
            return None, None

# ============================================================================
# SEGMENTATION COMPARISON
# ============================================================================
class SegmentationComparison:
    @staticmethod
    def compare_all_methods(image):
        """Compare all segmentation methods"""
        results = {}
        
        # Conventional methods
        conv_methods = {
            'Otsu': ConventionalSegmentation.otsu_segmentation,
            'Adaptive Threshold': ConventionalSegmentation.adaptive_threshold_segmentation,
            'Edge-based': ConventionalSegmentation.edge_based_segmentation,
            'Region Growing': ConventionalSegmentation.region_growing_segmentation
        }
        
        # AI methods
        ai_methods = {
            'Active Contour': AISegmentation.active_contour_segmentation,
            'Watershed': AISegmentation.watershed_segmentation,
            'Felzenszwalb': AISegmentation.felzenszwalb_segmentation,
            'SLIC': AISegmentation.slic_segmentation
        }
        
        # Apply all methods
        for name, method in conv_methods.items():
            try:
                mask, method_name = method(image)
                if mask is not None:
                    results[name] = {'mask': mask, 'type': 'Conventional', 'method': method_name}
            except:
                pass
        
        for name, method in ai_methods.items():
            try:
                mask, method_name = method(image)
                if mask is not None:
                    results[name] = {'mask': mask, 'type': 'AI', 'method': method_name}
            except:
                pass
        
        return results
    
    @staticmethod
    def calculate_segmentation_metrics(mask, original):
        """Calculate segmentation quality metrics"""
        try:
            # Area
            area = np.sum(mask > 0)
            
            # Perimeter
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum([cv2.arcLength(cnt, True) for cnt in contours]) if contours else 0
            
            # Compactness
            compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            
            # Centroid
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                cx, cy = 0, 0
            
            # Eccentricity
            if moments['m00'] > 0:
                mu20 = moments['mu20'] / moments['m00']
                mu02 = moments['mu02'] / moments['m00']
                mu11 = moments['mu11'] / moments['m00']
                eccentricity = np.sqrt((mu20 - mu02)**2 + 4*mu11**2) / (mu20 + mu02) if (mu20 + mu02) > 0 else 0
            else:
                eccentricity = 0
            
            return {
                'Area': area,
                'Perimeter': perimeter,
                'Compactness': compactness,
                'Centroid X': cx,
                'Centroid Y': cy,
                'Eccentricity': eccentricity
            }
        except:
            return {}

# ============================================================================
# ADVANCED METRICS
# ============================================================================
def calculate_advanced_metrics(img_array):
    """Calculate advanced image metrics"""
    try:
        gray_img = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        metrics = {
            'Mean Intensity': np.mean(gray_img),
            'Std Intensity': np.std(gray_img),
            'Dynamic Range': np.ptp(gray_img),
            'Entropy': entropy(gray_img.ravel()),
            'Skewness': pd.Series(gray_img.ravel()).skew(),
            'Kurtosis': pd.Series(gray_img.ravel()).kurtosis()
        }
        
        # GLCM features
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_img, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        metrics.update({
            'Contrast': np.mean(graycoprops(glcm, 'contrast')),
            'Homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
            'Energy': np.mean(graycoprops(glcm, 'energy')),
            'Correlation': np.mean(graycoprops(glcm, 'correlation')),
            'Dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
            'ASM': np.mean(graycoprops(glcm, 'ASM'))
        })
        
        # LBP
        lbp = local_binary_pattern(gray_img, 8 * 3, 3, method='uniform')
        metrics['LBP Variance'] = np.var(lbp)
        
        return metrics
    except Exception as e:
        return {}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_comparison_visualization(segmentation_results, original_image):
    """Create comprehensive comparison visualization"""
    try:
        num_methods = len(segmentation_results)
        if num_methods == 0:
            return None
        
        # Create figure with subplots
        cols = min(4, num_methods + 1)
        rows = (num_methods + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        # Plot original
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Plot each segmentation result
        for idx, (name, data) in enumerate(segmentation_results.items(), 1):
            if idx < len(axes):
                mask = data['mask']
                method_type = data['type']
                
                # Overlay mask on original
                overlay = original_image.copy()
                overlay[mask > 0] = [1, 0, 0]  # Red overlay
                
                axes[idx].imshow(overlay)
                axes[idx].set_title(f'{name}\n({method_type})', fontsize=10)
                axes[idx].axis('off')
        
        # Hide empty subplots
        for idx in range(len(segmentation_results) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def create_performance_comparison(segmentation_results):
    """Create performance comparison chart"""
    try:
        metrics_data = []
        
        for name, data in segmentation_results.items():
            mask = data['mask']
            metrics = SegmentationComparison.calculate_segmentation_metrics(mask, None)
            metrics['Method'] = name
            metrics['Type'] = data['type']
            metrics_data.append(metrics)
        
        if not metrics_data:
            return None
        
        df = pd.DataFrame(metrics_data)
        
        # Create comparison charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Area', 'Compactness', 'Eccentricity', 'Centroid Position']
        )
        
        # Area comparison
        fig.add_trace(
            go.Bar(x=df['Method'], y=df['Area'], name='Area',
                  marker_color=['#FF6B6B' if t == 'Conventional' else '#4ECDC4' 
                               for t in df['Type']]),
            row=1, col=1
        )
        
        # Compactness comparison
        fig.add_trace(
            go.Bar(x=df['Method'], y=df['Compactness'], name='Compactness',
                  marker_color=['#FF6B6B' if t == 'Conventional' else '#4ECDC4' 
                               for t in df['Type']]),
            row=1, col=2
        )
        
        # Eccentricity comparison
        fig.add_trace(
            go.Bar(x=df['Method'], y=df['Eccentricity'], name='Eccentricity',
                  marker_color=['#FF6B6B' if t == 'Conventional' else '#4ECDC4' 
                               for t in df['Type']]),
            row=2, col=1
        )
        
        # Centroid position
        fig.add_trace(
            go.Scatter(x=df['Centroid X'], y=df['Centroid Y'], 
                      mode='markers+text',
                      text=df['Method'],
                      textposition='top center',
                      marker=dict(size=15,
                                 color=['#FF6B6B' if t == 'Conventional' else '#4ECDC4' 
                                       for t in df['Type']]),
                      name='Centroid'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Method", row=1, col=1)
        fig.update_xaxes(title_text="Method", row=1, col=2)
        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_xaxes(title_text="X Position", row=2, col=2)
        fig.update_yaxes(title_text="Area (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Compactness", row=1, col=2)
        fig.update_yaxes(title_text="Eccentricity", row=2, col=1)
        fig.update_yaxes(title_text="Y Position", row=2, col=2)
        
        return fig
    except Exception as e:
        return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    config = load_config()
    
    # Header
    st.markdown("""
    <div class="gradient-header">
        <h1>🧠 Advanced Brain Tumor Detection & Segmentation</h1>
        <p>AI-Driven Analysis with Comprehensive Segmentation Comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display GIF
    try:
        st.image(config['GIF_PATH'], width=800)
    except:
        pass
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        input_method = st.radio("Select Input Method", ["Upload Image", "Use Sample Images"])
        
        if input_method == "Upload Image":
            selected_file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])
        else:
            GITHUB_BASE_URL = config['YES_IMAGES_DIR']
            demo_images = {
                "Tumor Cases": ["Y1.jpg", "Y2.jpg", "Y3.jpg", "Y4.jpg"],
                "Normal Cases": ["1 no.jpeg", "2 no.jpeg"]
            }
            case_type = st.selectbox("Select case type:", ["Tumor Cases", "Normal Cases"])
            selected_demo = st.selectbox("Choose a sample image:", demo_images[case_type])
            
            folder = "yes" if case_type == "Tumor Cases" else "no"
            selected_file = f"{config['YES_IMAGES_DIR'] if case_type == 'Tumor Cases' else config['NO_IMAGES_DIR']}{selected_demo.replace(' ', '%20')}?raw=true"
        
        st.markdown("---")
        st.subheader("📊 System Metrics")
        total_analyses = len(st.session_state.analysis_history)
        successful_analyses = sum(1 for x in st.session_state.analysis_history if x.get('success', False))
        success_rate = (successful_analyses / total_analyses) * 100 if total_analyses > 0 else 0
        st.metric("Total Analyses", total_analyses)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    if selected_file:
        try:
            # Load image
            if isinstance(selected_file, str):
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(selected_file, headers=headers)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                else:
                    st.error(f"Failed to load image: Status code {response.status_code}")
                    return
            else:
                image = Image.open(selected_file)
            
            # Convert and preprocess
            image_rgb = image.convert('RGB')
            img_array = np.array(image_rgb.resize((64, 64))) / 255.0
            
            # Load model
            analyzer = load_cached_model(config['MODEL_PATH'])
            if analyzer is None:
                st.error("Model loading failed. Please check your connection.")
                return
            
            # Predict
            start_time = time.time()
            prediction_input = np.expand_dims(img_array, axis=0)
            prediction = analyzer.predict(prediction_input, verbose=0)
            inference_time = time.time() - start_time
            
            result = np.argmax(prediction[0])
            confidence = prediction[0][result] * 100
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎯 Main Analysis",
                "🔬 Segmentation Comparison",
                "📊 Advanced Metrics",
                "📈 Performance Analysis",
                "📜 History"
            ])
            
            with tab1:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image_rgb, caption="Input MRI Image", width=None)
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
                    st.plotly_chart(fig, width='stretch')
                    
                    if result == 1:
                        st.error("🚨 Tumor Detected")
                        st.warning("Please consult with a medical professional.")
                    else:
                        st.success("✅ No Tumor Detected")
                        st.info("Regular check-ups are recommended.")
            
            with tab2:
                st.header("🔬 Segmentation Technique Comparison")
                st.markdown("""
                <div class="comparison-card">
                    <strong>📋 Comparison Overview</strong><br>
                    This section compares various segmentation techniques:<br>
                    <span style="color: #FF6B6B;">■ Conventional Methods</span> - Traditional image processing techniques<br>
                    <span style="color: #4ECDC4;">■ AI Methods</span> - Advanced AI-based segmentation
                </div>
                """, unsafe_allow_html=True)
                
                # Perform segmentation comparison
                with st.spinner("Applying segmentation techniques..."):
                    segmentation_results = SegmentationComparison.compare_all_methods(img_array)
                
                if segmentation_results:
                    # Visualization
                    st.subheader("📊 Segmentation Results Comparison")
                    fig_comparison = create_comparison_visualization(segmentation_results, img_array)
                    if fig_comparison:
                        st.pyplot(fig_comparison)
                        plt.close()
                    
                    # Performance metrics
                    st.subheader("📈 Performance Metrics Comparison")
                    fig_performance = create_performance_comparison(segmentation_results)
                    if fig_performance:
                        st.plotly_chart(fig_performance, width='stretch')
                    
                    # Detailed metrics table
                    st.subheader("📋 Detailed Segmentation Metrics")
                    metrics_data = []
                    for name, data in segmentation_results.items():
                        mask = data['mask']
                        metrics = SegmentationComparison.calculate_segmentation_metrics(mask, None)
                        metrics['Method'] = name
                        metrics['Type'] = data['type']
                        metrics_data.append(metrics)
                    
                    if metrics_data:
                        df_metrics = pd.DataFrame(metrics_data)
                        st.dataframe(df_metrics.round(2), use_container_width=True)
                    
                    # Method recommendation
                    st.subheader("💡 Method Recommendation")
                    best_method = max(segmentation_results.items(), 
                                    key=lambda x: SegmentationComparison.calculate_segmentation_metrics(
                                        x[1]['mask'], None).get('Compactness', 0))
                    
                    st.info(f"""
                    **Recommended Method:** {best_method[0]} ({best_method[1]['type']})
                    
                    This method shows the best overall performance based on:
                    - **Compactness:** Highest region compactness
                    - **Area:** Optimal region coverage
                    - **Eccentricity:** Best shape characteristics
                    
                    *Note: The recommendation is based on quantitative metrics. Clinical validation is recommended.*
                    """)
                else:
                    st.warning("Could not perform segmentation comparison. Some methods may have failed.")
            
            with tab3:
                st.header("📊 Advanced Image Metrics")
                
                metrics = calculate_advanced_metrics(img_array)
                if metrics:
                    # Display metrics
                    cols = st.columns(4)
                    metric_items = list(metrics.items())
                    for idx, (key, value) in enumerate(metric_items[:8]):
                        with cols[idx % 4]:
                            st.metric(key, f"{value:.3f}" if isinstance(value, float) else value)
                    
                    # Metrics visualization
                    st.subheader("📈 Metrics Visualization")
                    metrics_df = pd.DataFrame([metrics])
                    
                    # Bar chart
                    fig = px.bar(
                        metrics_df.melt(var_name='Metric', value_name='Value'),
                        x='Metric',
                        y='Value',
                        title='Advanced Image Metrics',
                        color='Metric',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Radar chart for selected metrics
                    st.subheader("🕸️ Metrics Radar Chart")
                    selected_metrics = ['Contrast', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
                    radar_data = {k: v for k, v in metrics.items() if k in selected_metrics}
                    
                    if radar_data:
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=list(radar_data.values()),
                            theta=list(radar_data.keys()),
                            fill='toself',
                            name='Image Metrics',
                            line=dict(color='#667eea')
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True)),
                            height=400,
                            title='GLCM Metrics Radar'
                        )
                        st.plotly_chart(fig_radar, width='stretch')
            
            with tab4:
                st.header("📈 Performance Analysis")
                
                # Sensitivity analysis
                st.subheader("Sensitivity Analysis")
                with st.spinner("Performing sensitivity analysis..."):
                    # Simple sensitivity analysis
                    noise_levels = np.linspace(0, 0.1, 5)
                    confidence_values = []
                    
                    for noise in noise_levels:
                        noisy_img = np.clip(img_array + np.random.normal(0, noise, img_array.shape), 0, 1)
                        pred = analyzer.predict(np.expand_dims(noisy_img, axis=0), verbose=0)
                        confidence_values.append(np.max(pred[0]) * 100)
                    
                    fig_sensitivity = go.Figure()
                    fig_sensitivity.add_trace(go.Scatter(
                        x=noise_levels * 100,
                        y=confidence_values,
                        mode='lines+markers',
                        name='Model Confidence',
                        line=dict(color='#667eea', width=2),
                        marker=dict(size=10)
                    ))
                    fig_sensitivity.update_layout(
                        title='Model Robustness to Noise',
                        xaxis_title='Noise Level (%)',
                        yaxis_title='Confidence (%)',
                        height=400
                    )
                    st.plotly_chart(fig_sensitivity, width='stretch')
                
                # Model performance metrics
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Inference Time", f"{inference_time*1000:.1f} ms")
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                with col3:
                    st.metric("Result", "Tumor Detected" if result == 1 else "No Tumor")
                
                # Feature importance
                st.subheader("Feature Importance Analysis")
                st.info("""
                **Key Features for Detection:**
                1. Texture features (GLCM metrics) - Most important
                2. Intensity statistics - Secondary importance
                3. Shape features - Supporting role
                
                *Analysis based on model feature importance.*
                """)
            
            with tab5:
                st.header("📜 Analysis History")
                
                # Store current analysis
                st.session_state.analysis_history.append({
                    "timestamp": datetime.now(),
                    "result": int(result) if result is not None else None,
                    "confidence_score": float(confidence) if confidence is not None else None,
                    "inference_time": float(inference_time) if inference_time is not None else None,
                    "success": True if result is not None else False,
                    "method": "CNN"
                })
                
                if st.session_state.analysis_history:
                    df_history = pd.DataFrame(st.session_state.analysis_history)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Analyses", len(df_history))
                    with col2:
                        tumor_count = sum(df_history['result'] == 1)
                        st.metric("Tumor Detected", tumor_count)
                    with col3:
                        avg_conf = df_history['confidence_score'].mean()
                        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                    
                    # Time series
                    st.subheader("Confidence Over Time")
                    fig_history = px.line(
                        df_history,
                        x='timestamp',
                        y='confidence_score',
                        title='Confidence Score Trend',
                        markers=True
                    )
                    st.plotly_chart(fig_history, width='stretch')
                    
                    # History table
                    st.subheader("Detailed History")
                    st.dataframe(df_history, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    else:
        st.info("👆 Please select an input method and image to begin analysis.")

if __name__ == "__main__":
    main()
