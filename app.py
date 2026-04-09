import time

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio  # FIXED: Use plotly.io instead of plotly.utils
import json  # Add json import
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from model import EEGPreprocessor, train_models
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and preprocessor
model = None
preprocessor = None
scaler = None
pca = None

def load_models():
    """Load trained models and preprocessors"""
    global model, preprocessor, scaler, pca
    try:
        # Load or train models if they don't exist
        if not os.path.exists('models/trained_model.pkl'):
            print("Training new models...")
            train_models()
        
        with open('models/trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            preprocessor = model_data['preprocessor']
            scaler = model_data['scaler']
            pca = model_data['pca']
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")



# Add this to your app.py file after creating the Flask app
from markupsafe import Markup
import json

@app.template_filter('tojsonhtml')
def to_json_html(obj):
    """Convert object to JSON for use in HTML templates"""
    return Markup(json.dumps(obj))

# Or use the simpler approach:
@app.template_filter('tojsonhtml') 
def to_json_html(obj):
    return json.dumps(obj)




@app.route('/')
def index():
    """Homepage with hero section"""
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    """Prediction page for file upload"""
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction - FIXED VERSION"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('prediction'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('prediction'))
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and validate data
            try:
                data = pd.read_csv(filepath)
                
                # Basic validation
                if data.empty:
                    raise ValueError("The uploaded CSV file is empty.")
                
                if data.shape[0] == 0:
                    raise ValueError("No data rows found in the CSV file.")
                
                print(f"Loaded CSV with shape: {data.shape}")
                print(f"Columns: {list(data.columns)}")
                
            except pd.errors.EmptyDataError:
                raise ValueError("The CSV file appears to be empty or corrupted.")
            except pd.errors.ParserError as e:
                raise ValueError(f"Error parsing CSV file: {str(e)}")
            
            # Process the data
            results = process_eeg_data(data)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('result.html', results=results)
            
        except Exception as e:
            # Clean up file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    time.sleep(2)   # wait for Windows to release the file
                    os.remove(filepath)
                except PermissionError:
                    print("File is still in use, skipping delete")
            
            error_msg = str(e)
            print(f"Full error: {error_msg}")
            flash(f'Error processing file: {error_msg}', 'error')
            return redirect(url_for('prediction'))
    else:
        flash('Please upload a valid CSV file', 'error')
        return redirect(url_for('prediction'))


def process_eeg_data(data):
    """Process EEG data and return predictions with visualizations - FIXED VERSION"""
    try:
        # DEBUG: Print initial data info
        print(f"DEBUG: Initial data shape: {data.shape}")
        print(f"DEBUG: Initial columns: {list(data.columns)}")
        print(f"DEBUG: First few rows:\n{data.head()}")
        
        # Check if data is empty
        if data.empty:
            raise ValueError("Uploaded CSV file is empty. Please upload a valid EEG data file.")
        
        # Prepare feature matrix X
        cols_to_drop = []
        if 'Unnamed' in data.columns:
            cols_to_drop.append('Unnamed')
        if 'y' in data.columns:
            cols_to_drop.append('y')
        
        # Drop columns if they exist
        X = data.drop(cols_to_drop, axis=1, errors='ignore')
        
        print(f"DEBUG: After dropping columns, X shape: {X.shape}")
        print(f"DEBUG: X columns: {list(X.columns)}")
        
        # Check if X is empty after dropping columns
        if X.empty or X.shape[0] == 0:
            raise ValueError("No data samples remaining after processing. Please check your CSV file format.")
        
        if X.shape[1] == 0:
            raise ValueError("No feature columns found. Please ensure your CSV has EEG feature columns (X1, X2, etc.)")
        
        # Handle different numbers of features
        required_features = 178
        
        if X.shape[1] < required_features:
            print(f"DEBUG: Padding features from {X.shape[1]} to {required_features}")
            # Pad with zeros if we have fewer features
            padding = np.zeros((X.shape[0], required_features - X.shape[1]))
            X_padded = np.hstack([X.values, padding])
            X = pd.DataFrame(X_padded, columns=[f'X{i+1}' for i in range(required_features)])
        elif X.shape[1] > required_features:
            print(f"DEBUG: Trimming features from {X.shape[1]} to {required_features}")
            # Take only first 178 features
            X = X.iloc[:, :required_features]
        
        # Final check before scaling
        if X.shape[0] == 0:
            raise ValueError("No data samples available for prediction.")
        
        if X.shape[1] != required_features:
            raise ValueError(f"Feature dimension mismatch. Expected {required_features}, got {X.shape[1]}")
        
        print(f"DEBUG: Final X shape before scaling: {X.shape}")
        
        # Check for NaN or infinite values
        if X.isnull().any().any():
            print("DEBUG: Found NaN values, filling with zeros")
            X = X.fillna(0)
        
        if np.isinf(X.values).any():
            print("DEBUG: Found infinite values, replacing with finite values")
            X = X.replace([np.inf, -np.inf], 0)
        
        # Scale and transform data
        try:
            X_scaled = scaler.transform(X)
            print(f"DEBUG: X_scaled shape: {X_scaled.shape}")
        except Exception as e:
            raise ValueError(f"Error during feature scaling: {str(e)}. Check if your data format matches the training data.")
        
        try:
            X_pca = pca.transform(X_scaled)
            print(f"DEBUG: X_pca shape: {X_pca.shape}")
        except Exception as e:
            raise ValueError(f"Error during PCA transformation: {str(e)}")
        
        # Make predictions
        try:
            predictions = model.predict(X_pca)
            probabilities = model.predict_proba(X_pca)
            print(f"DEBUG: Predictions shape: {predictions.shape}")
            print(f"DEBUG: Probabilities shape: {probabilities.shape}")
        except Exception as e:
            raise ValueError(f"Error during model prediction: {str(e)}")
        
        # Extract frequency features
        try:
            freq_features = preprocessor.extract_frequency_features_batch(X.values)
            print(f"DEBUG: Frequency features shape: {freq_features.shape}")
        except Exception as e:
            print(f"DEBUG: Warning - Could not extract frequency features: {str(e)}")
            freq_features = None
        
        # Create visualizations
        try:
            plots = create_visualizations(X, predictions, probabilities, freq_features)
        except Exception as e:
            print(f"DEBUG: Warning - Could not create visualizations: {str(e)}")
            plots = {}
        
        # Calculate statistics
        seizure_count = np.sum(predictions == 1)  # Assuming class 1 is seizure
        non_seizure_count = len(predictions) - seizure_count
        seizure_percentage = (seizure_count / len(predictions)) * 100 if len(predictions) > 0 else 0
        
        results = {
            'total_samples': int(len(predictions)),
            'seizure_count': int(seizure_count),
            'non_seizure_count': int(non_seizure_count),
            'seizure_percentage': round(float(seizure_percentage), 2),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'plots': plots
        }
        
        print("DEBUG: Processing completed successfully")
        return results
        
    except Exception as e:
        print(f"DEBUG: Error in process_eeg_data: {str(e)}")
        raise Exception(f"Error in data processing: {str(e)}")


def create_visualizations(X, predictions, probabilities, freq_features):
    """FIXED: Create various visualizations for the results"""
    plots = {}
    
    # 1. Prediction Distribution Pie Chart
    labels = ['Non-Seizure', 'Seizure']
    values = [np.sum(predictions == 0), np.sum(predictions == 1)]
    colors = ['#2E86AB', '#A23B72']
    
    fig1 = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.4,
        marker_colors=colors,
        textfont=dict(color='white', size=14)
    )])
    fig1.update_layout(
        title='Seizure Detection Results',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title_font=dict(size=18, color='white')
    )
    # FIXED: Use plotly.io.to_json and json.loads for proper serialization
    plots['pie_chart'] = json.loads(pio.to_json(fig1))
    
    # 2. Confidence Distribution
    confidence_scores = np.max(probabilities, axis=1)
    fig2 = go.Figure(data=[go.Histogram(
        x=confidence_scores,
        nbinsx=20,
        marker_color='#F18F01',
        opacity=0.8
    )])
    fig2.update_layout(
        title='Prediction Confidence Distribution',
        xaxis_title='Confidence Score',
        yaxis_title='Frequency',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,28,36,0.8)',
        font=dict(color='white'),
        title_font=dict(size=16, color='white'),
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(gridcolor='#444')
    )
    plots['confidence_hist'] = json.loads(pio.to_json(fig2))
    
    # 3. Sample EEG Signals
    fig3 = go.Figure()
    
    # Plot sample seizure and non-seizure signals
    seizure_indices = np.where(predictions == 1)[0]
    non_seizure_indices = np.where(predictions == 0)[0]
    
    if len(seizure_indices) > 0:
        seizure_sample = X.iloc[seizure_indices[0]].values
        fig3.add_trace(go.Scatter(
            y=seizure_sample,
            mode='lines',
            name='Seizure Signal',
            line=dict(color='#A23B72', width=2)
        ))
    
    if len(non_seizure_indices) > 0:
        non_seizure_sample = X.iloc[non_seizure_indices[0]].values
        fig3.add_trace(go.Scatter(
            y=non_seizure_sample,
            mode='lines',
            name='Non-Seizure Signal',
            line=dict(color='#2E86AB', width=2)
        ))
    
    fig3.update_layout(
        title='Sample EEG Signals',
        xaxis_title='Time Points',
        yaxis_title='Amplitude',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,28,36,0.8)',
        font=dict(color='white'),
        title_font=dict(size=16, color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(gridcolor='#444')
    )
    plots['eeg_signals'] = json.loads(pio.to_json(fig3))
    
    # 4. Frequency Band Analysis
    if freq_features is not None and len(freq_features) > 0:
        band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        seizure_freq = freq_features[predictions == 1] if np.sum(predictions) > 0 else np.array([[0]*5])
        non_seizure_freq = freq_features[predictions == 0] if np.sum(predictions == 0) > 0 else np.array([[0]*5])
        
        fig4 = go.Figure()
        
        if len(seizure_freq) > 0:
            fig4.add_trace(go.Bar(
                x=band_names,
                y=np.mean(seizure_freq, axis=0),
                name='Seizure',
                marker_color='#A23B72',
                opacity=0.8
            ))
        
        if len(non_seizure_freq) > 0:
            fig4.add_trace(go.Bar(
                x=band_names,
                y=np.mean(non_seizure_freq, axis=0),
                name='Non-Seizure',
                marker_color='#2E86AB',
                opacity=0.8
            ))
        
        fig4.update_layout(
            title='Average Frequency Band Power',
            xaxis_title='Frequency Bands',
            yaxis_title='Normalized Power',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,28,36,0.8)',
            font=dict(color='white'),
            title_font=dict(size=16, color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)'),
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444'),
            barmode='group'
        )
        plots['frequency_bands'] = json.loads(pio.to_json(fig4))
    
    return plots

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/visualization')
def visualization():
    """Visualization page with sample data"""
    return render_template('visualization.html')

@app.route('/api/sample_data')
def sample_data():
    """API endpoint for sample visualization data"""
    # Generate sample data for demonstration
    sample_data = generate_sample_visualization_data()
    return jsonify(sample_data)

def generate_sample_visualization_data():
    """Generate sample data for visualization page"""
    np.random.seed(42)
    
    # Sample EEG signals
    time_points = np.arange(0, 178)
    seizure_signal = np.random.normal(0, 50, 178) + 20 * np.sin(0.1 * time_points) + 30 * np.random.random(178)
    normal_signal = np.random.normal(0, 20, 178) + 10 * np.sin(0.05 * time_points)
    
    # Frequency bands data
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    seizure_powers = [0.35, 0.25, 0.15, 0.15, 0.10]
    normal_powers = [0.20, 0.30, 0.25, 0.20, 0.05]
    
    return {
        'eeg_signals': {
            'time_points': time_points.tolist(),
            'seizure_signal': seizure_signal.tolist(),
            'normal_signal': normal_signal.tolist()
        },
        'frequency_bands': {
            'bands': bands,
            'seizure_powers': seizure_powers,
            'normal_powers': normal_powers
        }
    }

if __name__ == '__main__':
    # Load models on startup
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)