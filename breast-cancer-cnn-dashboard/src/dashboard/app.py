"""
Interactive Streamlit dashboard for Breast Cancer CNN Classifier.
Provides real-time visualization of training progress, model evaluation, and predictions.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image
import io

# Import project modules - add src to path for Streamlit Cloud
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import config
from src.models import create_model
from src.data import create_preprocessor, BreastCancerDataset, get_val_transforms
from src.evaluation import ModelEvaluator, generate_evaluation_plots


# Page configuration
st.set_page_config(
    page_title="Breast Cancer CNN Classifier Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .dataset-status {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .dataset-success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
    .dataset-warning { background-color: #fff3cd; border: 1px solid #ffeeba; color: #856404; }
    .dataset-error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    .dataset-info { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str, architecture: str = 'CustomCNN', device: str = 'cpu'):
    """Load trained model from checkpoint."""
    device = torch.device(device)
    model = create_model(architecture)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


@st.cache_data
def load_training_history(history_path: str):
    """Load training history from JSON."""
    with open(history_path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_evaluation_results(results_path: str):
    """Load evaluation results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_split_info(split_path: str):
    """Load dataset split information."""
    with open(split_path, 'r') as f:
        return json.load(f)


@st.cache_data
def check_dataset_status() -> Dict[str, Any]:
    """Check the status of the dataset directories."""
    data_dir = Path("data")
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    status = {
        "data_dir_exists": data_dir.exists(),
        "raw_dir_exists": raw_dir.exists(),
        "raw_has_files": raw_dir.exists() and any(raw_dir.iterdir()),
        "processed_dir_exists": processed_dir.exists(),
        "processed_has_files": processed_dir.exists() and any(processed_dir.iterdir()),
        "model_dir_exists": Path("assets/models").exists(),
        "model_files": list(Path("assets/models").glob("**/*.pth")) if Path("assets/models").exists() else []
    }
    
    # Count files in raw
    if status["raw_has_files"]:
        status["raw_file_count"] = len(list(raw_dir.rglob("*.dcm"))) + len(list(raw_dir.rglob("*.DCM")))
    else:
        status["raw_file_count"] = 0
        
    return status


@st.cache_resource
def download_cbis_ddsm_dataset() -> Dict[str, Any]:
    """
    Download CBIS-DDSM dataset from Kaggle using kagglehub.
    Returns status dict with success/error info.
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        # Create directories
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset - this downloads the entire dataset to kagglehub cache
        # Then we need to copy/move files to our data/raw directory
        path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
        
        # Copy DICOM files to our data/raw directory
        copied = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.dcm', '.dicom')):
                    src = Path(root) / f
                    dst = Path("data/raw") / f
                    shutil.copy2(src, dst)
                    copied += 1
        
        return {
            "success": True,
            "message": f"Dataset downloaded successfully! {copied} DICOM files copied to data/raw/",
            "copied_files": copied,
            "source_path": path
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error downloading dataset: {str(e)}",
            "error": str(e)
        }


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">🔬 Breast Cancer CNN Classifier Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Invasive Ductal Carcinoma (IDC) Detection from Histopathology Images**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset Management Section
        st.subheader("📂 Dataset Management (CBIS-DDSM)")
        
        # Check dataset status
        status = check_dataset_status()
        
        # Display status
        if status["raw_has_files"]:
            st.markdown(f'<div class="dataset-status dataset-success">✅ Dataset ready: {status["raw_file_count"]} DICOM files in data/raw/</div>', unsafe_allow_html=True)
        elif status["raw_dir_exists"]:
            st.markdown('<div class="dataset-status dataset-warning">⚠️ data/raw/ exists but is empty</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="dataset-status dataset-error">❌ data/raw/ directory not found</div>', unsafe_allow_html=True)
        
        # Download button
        if not status["raw_has_files"]:
            if st.button("📥 Download CBIS-DDSM from Kaggle", type="primary", use_container_width=True):
                with st.spinner("Downloading CBIS-DDSM dataset from Kaggle... This may take a few minutes."):
                    result = download_cbis_ddsm_dataset()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])
        else:
            if st.button("🔄 Re-download Dataset", use_container_width=True):
                with st.spinner("Re-downloading CBIS-DDSM dataset from Kaggle..."):
                    result = download_cbis_ddsm_dataset()
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])
        
        st.divider()
        
        # Model selection
        model_dir = Path("assets/models")
        if model_dir.exists():
            model_files = list(model_dir.glob("**/*.pth"))
            if model_files:
                selected_model = st.selectbox(
                    "Select Model",
                    [str(f) for f in model_files],
                    format_func=lambda x: Path(x).name
                )
            else:
                selected_model = None
                st.warning("Models directory exists but no checkpoints found (.pth files)")
        else:
            selected_model = None
            st.warning("Models directory not found")
        
        # Device selection
        device = st.selectbox(
            "Device",
            ["cuda" if torch.cuda.is_available() else "cpu", "cpu"],
            index=0 if torch.cuda.is_available() else 1
        )
        
        # Architecture
        architecture = st.selectbox(
            "Architecture",
            ["CustomCNN", "resnet18", "resnet34", "efficientnet_b0"],
            index=0
        )
        
        st.divider()
        
        # Data paths
        st.subheader("📁 Data Paths")
        data_dir = Path("data")
        if data_dir.exists():
            raw_dir = Path("data/raw")
            processed_dir = Path("data/processed")
            if raw_dir.exists() and any(raw_dir.iterdir()):
                st.success("Data directory found")
            else:
                st.warning("Data directory exists but raw/ is empty - click 'Download CBIS-DDSM from Kaggle' above")
        else:
            st.warning("Data directory not found")
        
        # Threshold
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Training History",
        "📈 Model Evaluation",
        "🔍 Live Prediction",
        "📁 Data Explorer"
    ])
    
    # Tab 1: Training History
    with tab1:
        st.markdown('<div class="sub-header">Training History</div>', unsafe_allow_html=True)
        
        if selected_model:
            # Try to find history file
            model_path = Path(selected_model)
            history_path = model_path.parent / "training_history.json"
            
            if history_path.exists():
                history = load_training_history(history_path)
                
                # Create training curves
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy',
                                   'Learning Rate', 'Metric Comparison')
                )
                
                epochs = list(range(1, len(history.get('train_loss', [])) + 1))
                
                # Loss curves
                if 'train_loss' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', line=dict(color='#2E86AB')),
                        row=1, col=1
                    )
                if 'val_loss' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='#A23B72')),
                        row=1, col=1
                    )
                
                # Accuracy curves
                if 'train_accuracy' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['train_accuracy'], name='Train Acc', line=dict(color='#2E86AB')),
                        row=1, col=2
                    )
                if 'val_accuracy' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Acc', line=dict(color='#A23B72')),
                        row=1, col=2
                    )
                
                # Learning rate
                if 'lr' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['lr'], name='Learning Rate', line=dict(color='#F39C12')),
                        row=2, col=1
                    )
                
                # Final metrics comparison
                final_metrics = {}
                for key in ['val_accuracy', 'val_precision', 'val_recall', 'val_f1_score', 'val_roc_auc']:
                    if key in history and history[key]:
                        final_metrics[key.replace('val_', '').title()] = history[key][-1]
                
                if final_metrics:
                    fig.add_trace(
                        go.Bar(x=list(final_metrics.keys()), y=list(final_metrics.values()), name='Final Metrics'),
                        row=2, col=2
                    )
                
                fig.update_layout(height=800, showlegend=True, title_text="Training Progress")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display raw history
                with st.expander("View Raw History Data"):
                    st.json(history)
            else:
                st.info("No training history found for selected model")
        else:
            st.info("Select a model from sidebar to view training history")
    
    # Tab 2: Model Evaluation
    with tab2:
        st.markdown('<div class="sub-header">Model Evaluation</div>', unsafe_allow_html=True)
        
        if selected_model:
            # Try to load evaluation results
            model_path = Path(selected_model)
            results_path = model_path.parent / "evaluation_results.json"
            
            if results_path.exists():
                results = load_evaluation_results(results_path)
                
                # Display key metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                metrics = [
                    ("Accuracy", results.get('accuracy', 0)),
                    ("Precision", results.get('precision', 0)),
                    ("Recall", results.get('recall', 0)),
                    ("F1-Score", results.get('f1_score', 0)),
                    ("ROC-AUC", results.get('roc_auc', 0))
                ]
                
                for col, (name, value) in zip([col1, col2, col3, col4, col5], metrics):
                    with col:
                        st.metric(name, f"{value:.4f}")
                
                st.divider()
                
                # Confusion Matrix
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    cm = np.array(results.get('confusion_matrix', [[0, 0], [0, 0]]))
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=config.data.class_names,
                        y=config.data.class_names,
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Normalized Confusion Matrix")
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    fig = px.imshow(
                        cm_norm,
                        text_auto=".2f",
                        labels=dict(x="Predicted", y="Actual", color="Proportion"),
                        x=config.data.class_names,
                        y=config.data.class_names,
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(title="Normalized Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve
                st.subheader("ROC Curve")
                roc_data = results.get('roc_curve', {})
                if roc_data:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=roc_data.get('fpr', []),
                        y=roc_data.get('tpr', []),
                        mode='lines',
                        name=f'ROC (AUC = {roc_data.get("auc", 0):.4f})',
                        line=dict(color='#2E86AB', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(color='gray', dash='dash')
                    ))
                    fig.update_layout(
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        title='ROC Curve',
                        width=600, height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # PR Curve
                st.subheader("Precision-Recall Curve")
                pr_data = results.get('pr_curve', {})
                if pr_data:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pr_data.get('recall', []),
                        y=pr_data.get('precision', []),
                        mode='lines',
                        name=f'PR (AUC = {pr_data.get("auc", 0):.4f})',
                        line=dict(color='#A23B72', width=3)
                    ))
                    fig.update_layout(
                        xaxis_title='Recall',
                        yaxis_title='Precision',
                        title='Precision-Recall Curve',
                        width=600, height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Bootstrap CI
                if 'bootstrap_ci' in results:
                    st.subheader("Bootstrap 95% Confidence Intervals")
                    ci_data = results['bootstrap_ci']
                    ci_df = pd.DataFrame([
                        {"Metric": k, "Lower CI": v[0], "Upper CI": v[1], "Width": v[1] - v[0]}
                        for k, v in ci_data.items()
                    ])
                    st.dataframe(ci_df, use_container_width=True)
            else:
                st.info("No evaluation results found. Run evaluation to generate metrics.")
            
            if st.button("Run Evaluation"):
                with st.spinner("Evaluating model..."):
                    run_evaluation(selected_model, device, threshold)
        else:
            st.info("Select a model from sidebar to view evaluation")
    
    # Tab 3: Live Prediction
    with tab3:
        st.markdown('<div class="sub-header">Live Prediction</div>', unsafe_allow_html=True)
        
        if selected_model:
            model = load_model(selected_model, architecture, device)
            
            uploaded_file = st.file_uploader(
                "Upload histopathology image",
                type=['png', 'jpg', 'jpeg', 'tif', 'tiff']
            )
            
            if uploaded_file:
                # Display image
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", width=300)
                
                # Preprocess
                transform = get_val_transforms(tuple(config.data.image_size))
                img_array = np.array(image)
                augmented = transform(image=img_array)
                img_tensor = augmented['image'].unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    prob_idc = probs[0, 1].item()
                    pred_class = 1 if prob_idc >= threshold else 0
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction", config.data.class_names[pred_class])
                    st.metric("IDC Probability", f"{prob_idc:.4f}")
                
                with col2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_idc * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "IDC Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#A23B72" if prob_idc >= threshold else "#2E86AB"},
                            'steps': [
                                {'range': [0, threshold * 100], 'color': "#E8F5E9"},
                                {'range': [threshold * 100, 100], 'color': "#FCE4EC"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': threshold * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                if pred_class == 1:
                    st.error("⚠️ **IDC Detected**: The model predicts Invasive Ductal Carcinoma. Further clinical validation required.")
                else:
                    st.success("✅ **Non-IDC**: The model predicts no Invasive Ductal Carcinoma.")
                
                st.info("**Disclaimer**: This is a research tool. Clinical decisions should be made by qualified pathologists.")
        else:
            st.info("Select a model from sidebar to enable live prediction")
    
    # Tab 4: Data Explorer
    with tab4:
        st.markdown('<div class="sub-header">Data Explorer</div>', unsafe_allow_html=True)
        
        # Load split info
        split_path = Path("data/processed/split_info.json")
        if split_path.exists():
            split_info = load_split_info(split_path)
            
            # Dataset statistics
            st.subheader("Dataset Statistics")
            
            col1, col2, col3 = st.columns(3)
            total = sum(v['num_samples'] for v in split_info.values())
            
            with col1:
                st.metric("Total Samples", total)
            with col2:
                st.metric("Classes", len(config.data.class_names))
            with col3:
                # Class balance
                train_dist = split_info.get('train', {}).get('class_distribution', {})
                imbalance = max(train_dist.values()) / min(train_dist.values()) if train_dist else 1
                st.metric("Imbalance Ratio", f"{imbalance:.2f}")
            
            # Split distribution
            fig = make_subplots(rows=1, cols=3, subplot_titles=list(split_info.keys()))
            
            for i, (split_name, info) in enumerate(split_info.items()):
                dist = info.get('class_distribution', {})
                fig.add_trace(
                    go.Bar(
                        x=list(dist.keys()),
                        y=list(dist.values()),
                        name=split_name,
                        marker_color=['#2E86AB', '#A23B72']
                    ),
                    row=1, col=i+1
                )
            
            fig.update_layout(title="Class Distribution by Split", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Exploration report
            explore_path = Path("reports/exploration/exploration_report.json")
            if explore_path.exists():
                with st.expander("View Full Exploration Report"):
                    report = load_evaluation_results(explore_path)  # Reuse JSON loader
                    st.json(report)
            
            # Note: Technical report is generated separately as LaTeX/PDF in the repository
            # See scripts/generate_latex_report.py and reports/reporte_tecnico.tex
        
        else:
            st.info("Run data preprocessing to generate split information")


def run_evaluation(model_path: str, device: str, threshold: float):
    """Run model evaluation and save results."""
    device = torch.device(device)
    model = load_model(model_path, 'CustomCNN', device)
    
    # Create data loaders
    preprocessor = create_preprocessor()
    dataloaders = preprocessor.run_full_pipeline()
    
    # Evaluate
    evaluator = ModelEvaluator(model, device, config.data.class_names, threshold)
    results = evaluator.evaluate(dataloaders['test'])
    
    # Save results
    model_path = Path(model_path)
    output_path = model_path.parent / "evaluation_results.json"
    evaluator.save_results(results, str(output_path))
    
    # Generate plots
    generate_evaluation_plots(results, config.data.class_names, str(model_path.parent / "eval_plots"))
    
    st.success("Evaluation completed!")
    st.rerun()


if __name__ == "__main__":
    main()