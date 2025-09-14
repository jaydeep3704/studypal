import requests
import json

def test_prediction():
    url = "http://localhost:5000/predict"
    
    video_data = {
        "title": "Advanced Machine Learning with TensorFlow 2.0",
        "description": "Deep dive into advanced machine learning techniques using TensorFlow 2.0. Covers neural networks, optimization, and deployment.",
        "keywords": "machine learning, tensorflow, neural networks, deep learning, advanced, AI",
        "duration_minutes": 45,
        "keyword_count": 6,
        "views_per_minute": 0.3,
        "engagement_ratio": 0.0015,
        "comment_sentiment": 8.5
    }
    
    try:
        response = requests.post(url, json=video_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction Successful!")
            print(f"🎯 Level: {result['predicted_level']}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print("📈 Probabilities:")
            for level, prob in result['probabilities'].items():
                print(f"   {level}: {prob:.3f}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_health():
    url = "http://localhost:5000/health"
    try:
        response = requests.get(url, timeout=10)
        print(f"🏥 Health Check: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def test_model_info():
    url = "http://localhost:5000/model-info"
    try:
        response = requests.get(url, timeout=10)
        print(f"📋 Model Info: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Model info failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Flask ML Server...")
    print("=" * 50)
    
    test_health()
    print()
    
    test_model_info()
    print()
    
    test_prediction()