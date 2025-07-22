import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.semi_supervised import LabelSpreading
import joblib
import numpy as np

# Paths
LABELS_CSV = "reference_labels.csv"
TEXTS_DIR = "cleaned_texts"
MODELS_DIR = "."

# Load labeled data
df = pd.read_csv(LABELS_CSV)
print(f"Labeled samples: {len(df)}")
print(f"Label distribution: {df['label'].value_counts()}")

# Load all text files in the directory
all_files = [f for f in os.listdir(TEXTS_DIR) if f.endswith('.txt')]
print(f"Total text files found: {len(all_files)}")

# Load all texts
all_texts = []
all_labels = []
labeled_files = set(df['filename'].values)

for filename in all_files:
    file_path = os.path.join(TEXTS_DIR, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        all_texts.append(text)
        
        # Check if this file has a label
        if filename in labeled_files:
            label = df[df['filename'] == filename]['label'].iloc[0]
            # Convert to numeric: 1 for Publishable, 0 for Non-Publishable
            numeric_label = 1 if label == 'Publishable' else 0
            all_labels.append(numeric_label)
        else:
            # -1 indicates unlabeled data for semi-supervised learning
            all_labels.append(-1)

print(f"Total texts loaded: {len(all_texts)}")
print(f"Labeled: {sum(1 for l in all_labels if l != -1)}")
print(f"Unlabeled: {sum(1 for l in all_labels if l == -1)}")

# Create TF-IDF features using ALL texts (labeled + unlabeled)
vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
X_all = vectorizer.fit_transform(all_texts)
y_all = np.array(all_labels)

print(f"Feature matrix shape: {X_all.shape}")

# Try Label Spreading (semi-supervised)
print("\nTrying semi-supervised learning...")

# Test different parameters for Label Spreading
gamma_values = [0.1, 0.2, 0.5, 1.0]
alpha_values = [0.1, 0.2, 0.3]

best_score = 0
best_params = None

for gamma in gamma_values:
    for alpha in alpha_values:
        # Label Spreading
        label_spread = LabelSpreading(kernel='rbf', gamma=gamma, alpha=alpha)
        label_spread.fit(X_all, y_all)
        
        # Extract predictions for labeled data only
        labeled_indices = [i for i, label in enumerate(y_all) if label != -1]
        X_labeled = X_all[labeled_indices]
        y_labeled = y_all[labeled_indices]
        
        # Evaluate using cross-validation on labeled data
        # Create a simple classifier using the label spreading predictions
        predicted_labels = label_spread.predict(X_labeled)
        
        # Use Leave-One-Out CV to evaluate
        scores = []
        for train_idx, test_idx in LeaveOneOut().split(X_labeled):
            # Train label spreading on training fold
            X_train, y_train = X_labeled[train_idx], y_labeled[train_idx]
            
            # Create temporary labels for all data
            temp_y_all = y_all.copy()
            temp_y_all[labeled_indices] = -1  # Mark all as unlabeled first
            temp_y_all[[labeled_indices[i] for i in train_idx]] = y_train  # Only training labels
            
            # Fit and predict
            temp_model = LabelSpreading(kernel='rbf', gamma=gamma, alpha=alpha)
            temp_model.fit(X_all, temp_y_all)
            
            # Predict on test sample
            test_pred = temp_model.predict(X_labeled[test_idx])
            scores.append(test_pred[0] == y_labeled[test_idx][0])
        
        avg_score = np.mean(scores)
        print(f"Gamma={gamma}, Alpha={alpha}: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = (gamma, alpha)

print(f"\nBest semi-supervised score: {best_score:.4f}")
print(f"Best parameters: gamma={best_params[0]}, alpha={best_params[1]}")

# Compare with regular supervised learning
print("\nComparing with regular supervised learning:")
labeled_indices = [i for i, label in enumerate(y_all) if label != -1]
X_labeled = X_all[labeled_indices]
y_labeled = y_all[labeled_indices]

clf = LogisticRegression(max_iter=1000, C=0.1)
supervised_scores = cross_val_score(clf, X_labeled, y_labeled, cv=LeaveOneOut())
supervised_avg = supervised_scores.mean()

print(f"Regular supervised: {supervised_avg:.4f}")
print(f"Semi-supervised: {best_score:.4f}")
print(f"Improvement: {best_score - supervised_avg:.4f}")

# Train final model
if best_score > supervised_avg:
    print("\nUsing semi-supervised approach...")
    final_model = LabelSpreading(kernel='rbf', gamma=best_params[0], alpha=best_params[1])
    final_model.fit(X_all, y_all)
    model_type = "semi_supervised"
else:
    print("\nUsing supervised approach...")
    final_model = LogisticRegression(max_iter=1000, C=0.1)
    final_model.fit(X_labeled, y_labeled)
    model_type = "supervised"

# Save model and vectorizer
joblib.dump(final_model, os.path.join(MODELS_DIR, f"{model_type}_classifier.pkl"))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, f"{model_type}_vectorizer.pkl"))