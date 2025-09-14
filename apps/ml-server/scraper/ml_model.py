import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from collections import Counter
import re

class LearningLevelClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.text_vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = []
        self.best_model_name = None
        
    def load_and_preprocess(self, file_path):
        """Load and preprocess the labeled dataset"""
        df = pd.read_csv(file_path)
        
        print(f"📥 Loaded dataset with {len(df)} videos")
        
        # Ensure we have the learning_level column
        if 'learning_level' not in df.columns:
            raise ValueError("Dataset must contain 'learning_level' column")
        
        # Clean data
        df = df.dropna(subset=['keywords', 'title', 'description', 'learning_level'])
        df = df[df['keywords'] != 'No keywords']
        df = df[df['keyword_count'] > 2]
        
        # Create features
        df['title_length'] = df['title'].str.len()
        df['description_length'] = df['description'].str.len()
        df['views_per_minute'] = np.log1p(df['views'] / (df['duration_minutes'].replace(0, 1) + 1))
        df['engagement_ratio'] = np.log1p(df['likes'] / (df['views'].replace(0, 1) + 1))
        df['comment_sentiment'] = df['positive_comments(%)'] - df['negative_comments(%)']
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        print(f"📊 After cleaning: {len(df)} videos")
        print(f"🎯 Learning level distribution:")
        print(df['learning_level'].value_counts())
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        # Combine text features
        df['combined_text'] = df['title'] + ' ' + df['description'] + ' ' + df['keywords']
        
        # Text features
        text_features = ['combined_text']
        
        # Numerical features
        numerical_features = [
            'duration_minutes', 'keyword_count', 'title_length',
            'description_length', 'views_per_minute', 'engagement_ratio',
            'comment_sentiment'
        ]
        
        # Select available features
        available_numerical = [f for f in numerical_features if f in df.columns]
        self.feature_names = text_features + available_numerical
        
        X = df[self.feature_names]
        y = df['learning_level']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✅ Prepared {X.shape[0]} samples with {len(self.feature_names)} features")
        print(f"📊 Classes: {dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))}")
        
        return X, y_encoded
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        text_transformer = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2)))
        ])
        
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer([
            ('text', text_transformer, 'combined_text'),
            ('numeric', numeric_transformer, [
                'duration_minutes', 'keyword_count', 'title_length',
                'description_length', 'views_per_minute', 'engagement_ratio',
                'comment_sentiment'
            ])
        ])
        
        return preprocessor
    
    def train_models(self, X, y):
        """Train and compare multiple ML models"""
        preprocessor = self.create_preprocessor()
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'GradientBoosting': GradientBoostingClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🚀 Training {name}...")
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
            
            # Train on full data
            pipeline.fit(X, y)
            
            results[name] = {
                'pipeline': pipeline,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model_type': name
            }
            
            print(f"✅ {name} - CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results
    
    def select_best_model(self, results, X, y):
        """Select the best performing model"""
        best_score = -1
        best_model = None
        best_name = None
        
        for name, result in results.items():
            if result['cv_mean'] > best_score:
                best_score = result['cv_mean']
                best_model = result['pipeline']
                best_name = name
        
        self.best_model_name = best_name
        print(f"\n🎯 Best model: {best_name} with CV accuracy: {best_score:.3f}")
        
        # Final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"📊 Test Accuracy: {accuracy:.3f}")
        print(f"📊 Test F1 Score: {f1:.3f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        self.model = best_model
        return best_model
    
    def hyperparameter_tuning(self, pipeline, X, y):
        """Perform hyperparameter tuning with algorithm-specific parameters"""
        print(f"\n🔧 Performing hyperparameter tuning for {self.best_model_name}...")
        
        # Algorithm-specific parameter grids
        param_grids = {
            'RandomForest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 0.9, 1.0]
            },
            'SVM': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.1, 1],
                'classifier__kernel': ['linear', 'rbf']
            },
            'LogisticRegression': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__penalty': ['l2', 'none'],
                'classifier__solver': ['lbfgs', 'sag']
            }
        }
        
        # Get the appropriate parameter grid
        param_grid = param_grids.get(self.best_model_name, {})
        
        if not param_grid:
            print(f"⚠️ No parameter grid defined for {self.best_model_name}, skipping tuning")
            return pipeline
        
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=10, cv=3, 
            scoring='accuracy', random_state=self.random_state, n_jobs=-1,
            verbose=1
        )
        
        print(f"🔍 Tuning parameters: {list(param_grid.keys())}")
        search.fit(X, y)
        
        print(f"✅ Best parameters: {search.best_params_}")
        print(f"✅ Best CV score: {search.best_score_:.3f}")
        
        self.model = search.best_estimator_
        return search.best_estimator_
    
    def evaluate_model(self, X, y):
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        print(f"📊 Overall Accuracy: {accuracy:.3f}")
        print(f"📊 Overall F1 Score: {f1:.3f}")
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        self.plot_confusion_matrix(y, y_pred)
        
        # Feature importance (if available)
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            self.plot_feature_importance()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        try:
            if not hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                print("⚠️ Model doesn't support feature importance")
                return
            
            # Get feature names from the preprocessor
            feature_names = []
            
            # Text features from TF-IDF
            text_features = self.model.named_steps['preprocessor'].named_transformers_['text'].named_steps['tfidf'].get_feature_names_out()
            feature_names.extend(text_features)
            
            # Numerical features
            numerical_features = [
                'duration_minutes', 'keyword_count', 'title_length',
                'description_length', 'views_per_minute', 'engagement_ratio',
                'comment_sentiment'
            ]
            feature_names.extend(numerical_features)
            
            # Get feature importances
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Get top 20 features
            indices = np.argsort(importances)[-20:]
            plt.figure(figsize=(12, 8))
            plt.title('Top 20 Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot feature importance: {e}")
    
    def save_model(self, file_path):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save the model
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.best_model_name
        }, file_path)
        
        # Save metadata
        metadata = {
            'classes': self.label_encoder.classes_.tolist(),
            'feature_names': self.feature_names,
            'model_type': self.best_model_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(file_path.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to {file_path}")
        print(f"📋 Metadata saved to {file_path.replace('.pkl', '_metadata.json')}")
    
    def predict_new_video(self, video_data):
        """Predict learning level for a new video"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Create DataFrame with the same structure as training data
        if isinstance(video_data, dict):
            video_df = pd.DataFrame([video_data])
        else:
            video_df = video_data.copy()
        
        # Ensure all required features are present
        required_features = self.feature_names
        for feature in required_features:
            if feature not in video_df.columns:
                if feature == 'combined_text':
                    video_df['combined_text'] = video_df.get('title', '') + ' ' + \
                                              video_df.get('description', '') + ' ' + \
                                              video_df.get('keywords', '')
                else:
                    video_df[feature] = 0  # Fill missing with default
        
        # Make prediction
        prediction = self.model.predict(video_df)
        probabilities = self.model.predict_proba(video_df)
        
        # Decode prediction
        level = self.label_encoder.inverse_transform(prediction)[0]
        prob_dict = dict(zip(self.label_encoder.classes_, probabilities[0]))
        
        return {
            'predicted_level': level,
            'probabilities': prob_dict,
            'confidence': max(probabilities[0])
        }

# Example usage and training pipeline
def train_classification_model():
    """Complete training pipeline"""
    print("🚀 Training Learning Level Classification Model...")
    
    # Initialize classifier
    classifier = LearningLevelClassifier(random_state=42)
    
    # Load and preprocess data (use your rule-based labeled dataset)
    df = classifier.load_and_preprocess('rule_based_classified_dataset.csv')
    
    # Prepare features
    X, y = classifier.prepare_features(df)
    
    # Train and compare models
    results = classifier.train_models(X, y)
    
    # Select best model
    best_model = classifier.select_best_model(results, X, y)
    
    # Hyperparameter tuning (only if we have parameters for this model type)
    if classifier.best_model_name in ['RandomForest', 'GradientBoosting', 'SVM', 'LogisticRegression']:
        tuned_model = classifier.hyperparameter_tuning(best_model, X, y)
    else:
        print("⚠️ Skipping hyperparameter tuning for this model type")
        tuned_model = best_model
    
    # Final evaluation
    classifier.evaluate_model(X, y)
    
    # Save the model
    classifier.save_model('learning_level_classifier.pkl')
    
    print("\n🎉 Model training completed!")
    return classifier

# Function to load and use the trained model
def load_classifier(model_path):
    """Load a trained classifier"""
    loaded_data = joblib.load(model_path)
    
    classifier = LearningLevelClassifier()
    classifier.model = loaded_data['model']
    classifier.label_encoder = loaded_data['label_encoder']
    classifier.feature_names = loaded_data['feature_names']
    classifier.best_model_name = loaded_data.get('model_type', 'Unknown')
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model type: {classifier.best_model_name}")
    print(f"📊 Classes: {list(classifier.label_encoder.classes_)}")
    
    return classifier

# Example of classifying a new video
def classify_new_video_example():
    """Example of how to classify a new video"""
    # Load trained model
    classifier = load_classifier('learning_level_classifier.pkl')
    
    # New video data (this would come from your YouTube API)
    new_video = {
        'title': 'Python Programming Tutorial for Beginners - Learn Python Basics',
        'description': 'Complete Python tutorial for absolute beginners. Learn basic syntax, data types, and simple programs.',
        'keywords': 'python, programming, beginner, tutorial, basics, learn, easy',
        'duration_minutes': 25,
        'keyword_count': 7,
        'title_length': 50,
        'description_length': 120,
        'views_per_minute': 0.5,
        'engagement_ratio': 0.002,
        'comment_sentiment': 15.0
    }
    
    # Predict learning level
    result = classifier.predict_new_video(new_video)
    
    print(f"\n🎯 Prediction for new video:")
    print(f"   Title: {new_video['title']}")
    print(f"   Predicted Level: {result['predicted_level']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Probabilities: {result['probabilities']}")

if __name__ == "__main__":
    # Train the model
    classifier = train_classification_model()
    
    # Example of classifying a new video
    classify_new_video_example()