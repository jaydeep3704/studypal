from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# Global variables for the model
classifier = None
model_metadata = None
label_encoder = None

def load_model(model_path='./scraper/learning_level_classifier.pkl'):
    """Load the trained model and metadata"""
    global classifier, model_metadata, label_encoder
    
    try:
        # Load the model
        loaded_data = joblib.load(model_path)
        classifier = loaded_data['model']
        label_encoder = loaded_data['label_encoder']
        
        # Load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        
        logger.info(f"✅ Model loaded successfully: {loaded_data.get('model_type', 'Unknown')}")
        logger.info(f"📊 Classes: {label_encoder.classes_}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def preprocess_video_data(video_data):
    """Preprocess incoming video data for prediction"""
    # Create a copy of the data
    processed_data = video_data.copy()
    
    # Ensure all required features are present with proper types
    if 'title' in processed_data:
        processed_data['title'] = str(processed_data['title'])
    if 'description' in processed_data:
        processed_data['description'] = str(processed_data['description'])
    if 'keywords' in processed_data:
        processed_data['keywords'] = str(processed_data['keywords'])
    
    # Calculate missing features if possible
    if 'title_length' not in processed_data and 'title' in processed_data:
        processed_data['title_length'] = int(len(processed_data['title']))
    
    if 'description_length' not in processed_data and 'description' in processed_data:
        processed_data['description_length'] = int(len(processed_data['description']))
    
    if 'keyword_count' not in processed_data and 'keywords' in processed_data:
        if isinstance(processed_data['keywords'], str):
            processed_data['keyword_count'] = int(len(processed_data['keywords'].split(',')))
        else:
            processed_data['keyword_count'] = 0
    
    # Set default values for missing numerical features with proper types
    numerical_defaults = {
        'duration_minutes': 20.0,
        'views_per_minute': 0.5,
        'engagement_ratio': 0.002,
        'comment_sentiment': 0.0,
        'title_length': 0,
        'description_length': 0,
        'keyword_count': 0
    }
    
    for feature, default in numerical_defaults.items():
        if feature not in processed_data:
            processed_data[feature] = default
        else:
            # Ensure proper type
            if feature in ['title_length', 'description_length', 'keyword_count']:
                processed_data[feature] = int(processed_data[feature])
            else:
                processed_data[feature] = float(processed_data[feature])
    
    # Create combined text feature
    processed_data['combined_text'] = (
        str(processed_data.get('title', '')) + ' ' +
        str(processed_data.get('description', '')) + ' ' +
        str(processed_data.get('keywords', ''))
    )
    
    return processed_data

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    info = {
        'model_type': model_metadata.get('model_type', 'Unknown') if model_metadata else 'Unknown',
        'classes': convert_numpy_types(label_encoder.classes_.tolist()),
        'feature_names': model_metadata.get('feature_names', []) if model_metadata else [],
        'loaded_at': model_metadata.get('timestamp', 'Unknown') if model_metadata else 'Unknown'
    }
    
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict learning level for a video"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded. Please load the model first.'}), 503
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess the data
        processed_data = preprocess_video_data(data)
        
        # Convert to DataFrame for prediction
        video_df = pd.DataFrame([processed_data])
        
        # Make prediction
        prediction = classifier.predict(video_df)
        probabilities = classifier.predict_proba(video_df)
        
        # Convert numpy types to Python native types
        prediction = convert_numpy_types(prediction)
        probabilities = convert_numpy_types(probabilities)
        
        # Get the predicted level
        level_index = prediction[0]
        level = label_encoder.classes_[level_index]
        
        # Create probabilities dictionary
        prob_dict = {}
        for i, class_name in enumerate(label_encoder.classes_):
            prob_dict[str(class_name)] = float(probabilities[0][i])
        
        # Prepare response
        response = {
            'predicted_level': str(level),
            'confidence': float(np.max(probabilities[0])),
            'probabilities': prob_dict,
            'timestamp': datetime.now().isoformat(),
            'input_features': {
                'title': str(processed_data.get('title', '')),
                'duration_minutes': float(processed_data.get('duration_minutes', 0)),
                'keyword_count': int(processed_data.get('keyword_count', 0)),
                'engagement_ratio': float(processed_data.get('engagement_ratio', 0))
            }
        }
        
        logger.info(f"📊 Prediction: {response['predicted_level']} (confidence: {response['confidence']:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predict learning levels for multiple videos"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected a list of videos'}), 400
        
        results = []
        for i, video_data in enumerate(data):
            try:
                processed_data = preprocess_video_data(video_data)
                video_df = pd.DataFrame([processed_data])
                
                prediction = classifier.predict(video_df)
                probabilities = classifier.predict_proba(video_df)
                
                # Convert numpy types
                prediction = convert_numpy_types(prediction)
                probabilities = convert_numpy_types(probabilities)
                
                # Get prediction details
                level_index = prediction[0]
                level = label_encoder.classes_[level_index]
                confidence = float(np.max(probabilities[0]))
                
                results.append({
                    'index': i,
                    'predicted_level': str(level),
                    'confidence': confidence,
                    'title': str(processed_data.get('title', 'Unknown')),
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'title': str(video_data.get('title', 'Unknown')),
                    'success': False
                })
        
        response = {
            'results': results,
            'total_processed': len(results),
            'successful': len([r for r in results if r.get('success', False)]),
            'failed': len([r for r in results if not r.get('success', True)])
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}", exc_info=True)
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    """Load or reload the model"""
    global classifier, model_metadata, label_encoder
    
    model_path = request.json.get('model_path', 'learning_level_classifier.pkl') if request.json else 'learning_level_classifier.pkl'
    
    try:
        success = load_model(model_path)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Model loaded from {model_path}',
                'model_type': model_metadata.get('model_type', 'Unknown') if model_metadata else 'Unknown'
            })
        else:
            return jsonify({'error': 'Failed to load model'}), 500
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load the model when starting the server
    model_loaded = load_model()
    
    if not model_loaded:
        logger.warning("⚠️ Starting server without model. Use POST /load-model to load it.")
    
    # Start the Flask server
    logger.info("🚀 Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)