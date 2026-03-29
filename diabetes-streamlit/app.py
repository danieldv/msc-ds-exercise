"""
Enhanced Streamlit Application for Diabetes Progression Prediction

This application demonstrates Streamlit's capabilities for building
interactive ML applications with minimal code.
"""
import streamlit as st
import numpy as np
import pandas as pd
import dill as pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os
from pathlib import Path

# Add config to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


@st.cache_resource
def load_model(model_path: str):
    """
    Load the pre-trained model with caching.
    Streamlit will cache this function so the model is only loaded once.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, True
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.info("Please run the training notebook first to generate the model.")
        return None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False


def calculate_feature_importance(model, feature_names):
    """Calculate feature importance for visualization."""
    # For Ridge regression, use absolute coefficients as importance
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        return importance_df
    return None


def create_prediction_gauge(prediction_value: float):
    """Create a gauge chart for the prediction."""
    # Normalize prediction to 0-100 scale (assuming typical range 25-350)
    normalized = max(0, min(100, (prediction_value / 350) * 100))
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Progression Score"},
        delta = {'reference': 150},
        gauge = {
            'axis': {'range': [None, 350]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 100], 'color': "lightgreen"},
                {'range': [100, 200], 'color': "yellow"},
                {'range': [200, 350], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 200
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def main():
    """Main application function."""
    
    # Sidebar navigation
    st.sidebar.title("🩺 Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Prediction", "Model Info", "Prediction History", "Batch Prediction"]
    )
    
    # Load model (cached)
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            model, success = load_model(Config.MODEL_PATH)
            if success:
                st.session_state.model = model
                st.session_state.model_loaded = True
            else:
                st.stop()
    
    model = st.session_state.model
    feature_names = Config.get_feature_names()
    
    # Main content based on selected page
    if page == "Prediction":
        show_prediction_page(model, feature_names)
    elif page == "Model Info":
        show_model_info_page(model, feature_names)
    elif page == "Prediction History":
        show_history_page()
    elif page == "Batch Prediction":
        show_batch_prediction_page(model, feature_names)


def show_prediction_page(model, feature_names):
    """Show the main prediction interface."""
    
    # Check for preset values that need to be applied (before widgets are created)
    if '_preset_type' in st.session_state:
        preset_type = st.session_state['_preset_type']
        del st.session_state['_preset_type']  # Clear the flag
        
        if preset_type == 'low':
            apply_low_risk_preset(feature_names)
        elif preset_type == 'medium':
            apply_medium_risk_preset(feature_names)
        elif preset_type == 'high':
            apply_high_risk_preset(feature_names)
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        if Path(Config.LOGO_PATH).is_file():
            st.image(Config.LOGO_PATH, width=150)
    with col2:
        st.title("Diabetes Progression Prediction")
        st.markdown("Enter patient information to predict diabetes progression score.")
    
    st.divider()
    
    # Preset scenarios
    st.subheader("📋 Quick Presets")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Low Risk Profile", use_container_width=True, key="preset_low"):
            st.session_state['_preset_type'] = 'low'
            st.rerun()
    with col2:
        if st.button("Medium Risk Profile", use_container_width=True, key="preset_medium"):
            st.session_state['_preset_type'] = 'medium'
            st.rerun()
    with col3:
        if st.button("High Risk Profile", use_container_width=True, key="preset_high"):
            st.session_state['_preset_type'] = 'high'
            st.rerun()
    
    st.divider()
    
    # Input form
    st.subheader("📝 Patient Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    user_inputs = {}
    feature_defaults = Config.get_feature_defaults()
    
    # First column
    with col1:
        for i, feature in enumerate(feature_names[:5]):
            info = Config.FEATURE_INFO[feature]
            value = st.number_input(
                f"{feature} ({info['unit']})" if info['unit'] else feature,
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=float(st.session_state.get(f'input_{feature}', info['default'])),
                step=0.1,
                help=info['description'],
                key=f'input_{feature}'
            )
            user_inputs[feature] = value
    
    # Second column
    with col2:
        for i, feature in enumerate(feature_names[5:]):
            info = Config.FEATURE_INFO[feature]
            value = st.number_input(
                f"{feature} ({info['unit']})" if info['unit'] else feature,
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=float(st.session_state.get(f'input_{feature}', info['default'])),
                step=0.1,
                help=info['description'],
                key=f'input_{feature}'
            )
            user_inputs[feature] = value
    
    st.divider()
    
    # Prediction button and results
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Diabetes Progression", 
                                  use_container_width=True, 
                                  type="primary")
    
    if predict_button:
        # Prepare input array
        input_array = np.array([user_inputs[f] for f in feature_names]).reshape(1, -1)
        
        # Make prediction
        with st.spinner("Calculating prediction..."):
            prediction = model.predict(input_array)[0]
        
        # Store in history
        history_entry = {
            'timestamp': datetime.now(),
            'inputs': user_inputs.copy(),
            'prediction': float(prediction)
        }
        st.session_state.prediction_history.append(history_entry)
        
        # Display results
        st.success("✅ Prediction completed!")
        
        # Results in columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Prediction Result")
            
            # Gauge chart
            fig = create_prediction_gauge(prediction)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.info(f"""
            **Predicted Diabetes Progression Score: {prediction:.2f}**
            
            - **Low Risk**: 0-100 (Green zone)
            - **Medium Risk**: 100-200 (Yellow zone)
            - **High Risk**: 200+ (Red zone)
            """)
        
        with col2:
            st.subheader("📈 Model Metrics")
            st.metric("R² Score", f"{Config.MODEL_METRICS['r2_score']:.4f}")
            st.metric("MAPE", f"{Config.MODEL_METRICS['mape']:.2f}%")
            
            # Feature comparison
            st.subheader("🔍 Input Summary")
            df_input = pd.DataFrame({
                'Feature': list(user_inputs.keys()),
                'Value': list(user_inputs.values())
            })
            st.dataframe(df_input, use_container_width=True, hide_index=True)


def show_model_info_page(model, feature_names):
    """Show model information and performance."""
    
    st.title("📊 Model Information")
    
    # Model performance metrics
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{Config.MODEL_METRICS['r2_score']:.4f}", 
                 help="Coefficient of determination. Higher is better (max 1.0)")
    with col2:
        st.metric("MAPE", f"{Config.MODEL_METRICS['mape']:.2f}%",
                 help="Mean Absolute Percentage Error. Lower is better")
    with col3:
        st.metric("Model Type", "Ridge Regression",
                 help="Regularized linear regression model")
    
    st.divider()
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = calculate_feature_importance(model, feature_names)
    
    if importance_df is not None:
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance (Absolute Coefficients)",
            labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Dataset information
    st.subheader("Dataset Information")
    with st.expander("📋 Feature Descriptions"):
        for feature, info in Config.FEATURE_INFO.items():
            st.markdown(f"""
            **{feature}** ({info['unit']})
            - {info['description']}
            - Range: {info['min']} - {info['max']}
            """)
    
    # Model details
    with st.expander("🔧 Model Details"):
        st.code(f"""
        Model Type: Ridge Regression
        Model Path: {Config.MODEL_PATH}
        Features: {len(feature_names)}
        Training R²: {Config.MODEL_METRICS['r2_score']:.4f}
        Training MAPE: {Config.MODEL_METRICS['mape']:.2f}%
        """)


def show_history_page():
    """Show prediction history."""
    
    st.title("📜 Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No predictions made yet. Go to the Prediction page to make your first prediction!")
        return
    
    # Convert history to DataFrame
    history = st.session_state.prediction_history
    df_history = pd.DataFrame([
        {
            'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'Prediction': entry['prediction'],
            **{f: entry['inputs'][f] for f in Config.get_feature_names()}
        }
        for entry in history
    ])
    
    # Display table
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    # Visualization
    st.subheader("Prediction Trend")
    fig = px.line(
        df_history,
        x='Timestamp',
        y='Prediction',
        title="Prediction History Over Time",
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(history))
    with col2:
        st.metric("Average Prediction", f"{df_history['Prediction'].mean():.2f}")
    with col3:
        st.metric("Latest Prediction", f"{df_history['Prediction'].iloc[-1]:.2f}")
    
    # Download button
    csv = df_history.to_csv(index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Clear history
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.prediction_history = []
        st.rerun()


def show_batch_prediction_page(model, feature_names):
    """Show batch prediction interface."""
    
    st.title("📦 Batch Prediction")
    st.markdown("Upload a CSV file with patient data to make multiple predictions at once.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file should have columns matching the feature names"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} rows.")
            
            # Check if required columns exist
            missing_cols = set(feature_names) - set(df.columns)
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
                st.info(f"Required columns: {', '.join(feature_names)}")
            else:
                # Display preview
                st.subheader("Data Preview")
                st.dataframe(df[feature_names], use_container_width=True)
                
                # Make predictions
                if st.button("🔮 Predict All", type="primary"):
                    with st.spinner("Making predictions..."):
                        # Prepare input array
                        X_batch = df[feature_names].values
                        predictions = model.predict(X_batch)
                        
                        # Add predictions to dataframe
                        df_results = df.copy()
                        df_results['Predicted_Progression'] = predictions
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                        with col2:
                            st.metric("Average Prediction", f"{predictions.mean():.2f}")
                        with col3:
                            st.metric("Std Deviation", f"{predictions.std():.2f}")
                        
                        # Download results
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        # Show example format
        st.info("💡 Upload a CSV file with the following columns:")
        example_df = pd.DataFrame({
            feature: [Config.FEATURE_INFO[feature]['default']] 
            for feature in feature_names
        })
        st.dataframe(example_df, use_container_width=True)
        
        # Download template
        csv_template = example_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="prediction_template.csv",
            mime="text/csv"
        )


def apply_low_risk_preset(feature_names):
    """Apply low risk preset values before widgets are created."""
    low_risk_values = {
        'Age': 35.0,
        'Sex': 0.0,
        'BMI': 22.0,
        'BP': 70.0,
        'S1': 180.0,
        'S2': 100.0,
        'S3': 55.0,
        'S4': 3.5,
        'S5': 4.0,
        'S6': 80.0
    }
    for feature in feature_names:
        st.session_state[f'input_{feature}'] = low_risk_values.get(feature, Config.FEATURE_INFO[feature]['default'])


def apply_medium_risk_preset(feature_names):
    """Apply medium risk preset values before widgets are created."""
    medium_risk_values = {
        'Age': 50.0,
        'Sex': 0.0,
        'BMI': 25.0,
        'BP': 75.0,
        'S1': 200.0,
        'S2': 150.0,
        'S3': 45.0,
        'S4': 5.0,
        'S5': 4.5,
        'S6': 85.0
    }
    for feature in feature_names:
        st.session_state[f'input_{feature}'] = medium_risk_values.get(feature, Config.FEATURE_INFO[feature]['default'])


def apply_high_risk_preset(feature_names):
    """Apply high risk preset values before widgets are created."""
    high_risk_values = {
        'Age': 65.0,
        'Sex': 1.0,
        'BMI': 30.0,
        'BP': 90.0,
        'S1': 250.0,
        'S2': 180.0,
        'S3': 35.0,
        'S4': 7.0,
        'S5': 5.5,
        'S6': 120.0
    }
    for feature in feature_names:
        st.session_state[f'input_{feature}'] = high_risk_values.get(feature, Config.FEATURE_INFO[feature]['default'])


if __name__ == "__main__":
    main()
