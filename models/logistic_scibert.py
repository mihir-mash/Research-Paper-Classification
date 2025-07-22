import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
import numpy as np
from tqdm import tqdm
import warnings
from scibert_embedder import SciBERTEmbedder

loaded_model = None

def get_loaded_model(path="models/scibert_small_classifier.pkl"):
    global loaded_model
    if loaded_model is None:
        loaded_model = joblib.load(path)
    return loaded_model

# Suppress warnings
warnings.filterwarnings('ignore')

# Paths
LABELS_CSV = "reference_labels.csv"
TEXTS_DIR = "cleaned_texts"
MODELS_DIR = "."

def load_data():
    """Load texts and labels from files."""
    # Load labels
    df = pd.read_csv(LABELS_CSV)
    print(f"Total labeled samples: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts()}")
    
    # Load texts and labels
    texts = []
    labels = []
    filenames = []
    
    for _, row in df.iterrows():
        file_path = os.path.join(TEXTS_DIR, row['filename'])
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty texts
                    texts.append(content)
                    labels.append(row['label'])
                    filenames.append(row['filename'])
                else:
                    print(f"Warning: {file_path} is empty.")
        else:
            print(f"Warning: {file_path} not found.")
    
    print(f"Successfully loaded {len(texts)} texts")
    return texts, labels, filenames

def main():
    # Load data
    texts, labels, filenames = load_data()
    
    if len(texts) == 0:
        print("No texts loaded. Please check your file paths.")
        return
    
    # Initialize SciBERT embedder
    embedder = SciBERTEmbedder()
    
    # Get embeddings
    print("Generating SciBERT embeddings...")
    X = embedder.get_embeddings(texts)
    y = np.array(labels)
    
    print(f"Embeddings shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    configs = [
        {"C": 0.01, "solver": "liblinear"},
        {"C": 0.1, "solver": "liblinear"},
        {"C": 1.0, "solver": "liblinear"},
        {"C": 10.0, "solver": "liblinear"},
        {"C": 0.1, "solver": "lbfgs"},
        {"C": 1.0, "solver": "lbfgs"},
        {"C": 10.0, "solver": "lbfgs"},
        {"C": 0.1, "solver": "saga", "penalty": "l1"},
        {"C": 1.0, "solver": "saga", "penalty": "l1"},
    ]
    
    print("\nTesting SciBERT configurations for small dataset:")
    print("-" * 70)
    
    best_score = 0
    best_config = None
    best_f1 = 0
    
    for i, config in enumerate(configs):

        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            **config
        )
        
        # Use stratified cross-validation for small datasets
        if len(set(y)) > 1 and len(y) >= 10:
            cv = StratifiedKFold(n_splits=min(5, len(y)//2), shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_macro')
        else:
            # Fall back to LOO for very small datasets
            cv = LeaveOneOut()
            scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_macro')
        
        avg_score = scores.mean()
        avg_f1 = f1_scores.mean()
        
        print(f"Config {i+1}: C={config['C']}, solver={config['solver']}")
        if 'penalty' in config:
            print(f"  penalty={config['penalty']}")
        print(f"  CV Accuracy: {avg_score:.4f} (±{scores.std():.3f})")
        print(f"  CV F1-macro: {avg_f1:.4f} (±{f1_scores.std():.3f})")
        print()
        
        # Select best based on F1 score (better for imbalanced data)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_score = avg_score
            best_config = config
    
    print("-" * 70)
    print(f"Best configuration: {best_config}")
    print(f"Best CV Accuracy: {best_score:.4f}")
    print(f"Best CV F1-macro: {best_f1:.4f}")
    
    # Train final model with best config
    print(f"\nTraining final model with SciBERT embeddings...")
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        **best_config
    )
    clf.fit(X_scaled, y)
    
    # Get feature importance (coefficients)
    feature_importance = np.abs(clf.coef_[0])
    print(f"Average absolute coefficient: {feature_importance.mean():.4f}")
    print(f"Max absolute coefficient: {feature_importance.max():.4f}")
    
    # Save model, scaler, and embedder info
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'model_name': embedder.model_name,
        'max_length': embedder.max_length,
        'best_config': best_config,
        'cv_accuracy': best_score,
        'cv_f1': best_f1
    }
    
    joblib.dump(model_data, os.path.join(MODELS_DIR, "scibert_small_classifier.pkl"))
    print(f"Model saved to: {os.path.join(MODELS_DIR, 'scibert_small_classifier.pkl')}")
    
    # Final evaluation on full dataset (for reference)
    y_pred = clf.predict(X_scaled)
    final_accuracy = accuracy_score(y, y_pred)
    final_f1 = f1_score(y, y_pred, average='macro')
    
    print(f"\nFinal model performance on full dataset:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"F1-macro: {final_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Create a results DataFrame
    results_df = pd.DataFrame({
        'filename': filenames,
        'true_label': y,
        'predicted_label': y_pred,
        'correct': y == y_pred
    })
    
    results_df.to_csv(os.path.join(MODELS_DIR, "scibert_predictions.csv"), index=False)
    print(f"\nDetailed results saved to: {os.path.join(MODELS_DIR, 'scibert_predictions.csv')}")
    
    return model_data

def predict_new_texts(model_path, texts):
    """Function to make predictions on new texts."""
    model_data = get_loaded_model(model_path)
    clf = model_data['classifier']
    scaler = model_data['scaler']
    
    # Initialize embedder with same config
    embedder = SciBERTEmbedder(
        model_name=model_data['model_name'],
        max_length=model_data['max_length']
    )
    
    # Get embeddings and scale
    X = embedder.get_embeddings(texts)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = clf.predict(X_scaled)
    probabilities = clf.predict_proba(X_scaled)
    
    return predictions, probabilities

if __name__ == "__main__":
    main()