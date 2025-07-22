import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Paths
LABELS_CSV = "reference_labels.csv"
TEXTS_DIR = "cleaned_texts"
MODELS_DIR = "."

# Load labels
df = pd.read_csv(LABELS_CSV)
print(f"Total labeled samples: {len(df)}")
print(f"Label distribution: {df['label'].value_counts()}")

# Load texts and labels
texts = []
labels = []
for _, row in df.iterrows():
    file_path = os.path.join(TEXTS_DIR, row['filename'])
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
        labels.append(row['label'])
    else:
        print(f"Warning: {file_path} not found.")

print(f"Successfully loaded {len(texts)} texts")

# For small datasets, we need to be more careful
# Test simpler configurations that work better with small data

configs = [
    {"max_features": 100, "min_df": 1, "max_df": 1.0, "C": 0.1},
    {"max_features": 200, "min_df": 1, "max_df": 1.0, "C": 0.1},
    {"max_features": 300, "min_df": 1, "max_df": 1.0, "C": 0.1},
    {"max_features": 500, "min_df": 1, "max_df": 1.0, "C": 1.0},
    {"max_features": 100, "min_df": 1, "max_df": 0.8, "C": 0.1},
    {"max_features": 200, "min_df": 1, "max_df": 0.8, "C": 1.0},
]

print("\nTesting configurations for small dataset:")
print("-" * 60)

best_score = 0
best_config = None

for i, config in enumerate(configs):
    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        min_df=config["min_df"],
        max_df=config["max_df"],
        ngram_range=(1, 1)
    )
    
    X = vectorizer.fit_transform(texts)
    y = labels
    
    # Use Leave-One-Out cross-validation for small datasets
    clf = LogisticRegression(max_iter=1000, C=config["C"])
    
    # Use Leave-One-Out CV - better for small datasets
    loo_scores = cross_val_score(clf, X, y, cv=LeaveOneOut())
    avg_score = loo_scores.mean()
    
    print(f"Config {i+1}: max_feat={config['max_features']}, C={config['C']}, max_df={config['max_df']}")
    print(f"  LOO CV Score: {avg_score:.4f} | Features used: {X.shape[1]}")
    
    if avg_score > best_score:
        best_score = avg_score
        best_config = config

print("-" * 60)
print(f"Best configuration: {best_config}")
print(f"Best LOO CV score: {best_score:.4f}")
print(f"Improvement: {best_score - 0.67:.4f}")

# Train final model with best config
print(f"\nTraining final model...")
vectorizer = TfidfVectorizer(
    max_features=best_config["max_features"],
    min_df=best_config["min_df"],
    max_df=best_config["max_df"],
    ngram_range=(1, 1)
)
X = vectorizer.fit_transform(texts)
clf = LogisticRegression(max_iter=1000, C=best_config["C"])
clf.fit(X, y)

# Save model and vectorizer
joblib.dump(clf, os.path.join(MODELS_DIR, "small_data_classifier.pkl"))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "small_data_vectorizer.pkl"))