import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve,
                             balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import time

def load_and_prepare_data(csv_path):
    """Load the prepared dataset and prepare features/labels."""
    print("Loading training data...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Label'].value_counts()}")
    print(f"Class ratio: {df['Label'].value_counts()[1] / df['Label'].value_counts()[0]:.2f}:1")
    
    # Define features and target
    feature_columns = ['Length', 'Net_Charge', 'Hydrophobicity_GRAVY', 
                      'Instability_Index', 'Fraction_Positive_AA', 
                      'Fraction_Negative_AA', 'Fraction_Hydrophobic_AA']
    
    # Check if all columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[feature_columns]
    y = df['Label']
    
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    return X, y, feature_columns

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation."""
    print(f"\n{'='*50}")
    print(f"📊 {model_name} EVALUATION")
    print(f"{'='*50}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-AMP', 'AMP']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-AMP', 'AMP'], 
                yticklabels=['Non-AMP', 'AMP'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_roc_curve.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, balanced_acc, roc_auc

def train_random_forest(X_train, y_train, X_test, y_test, feature_columns):
    """Train and evaluate a Random Forest model."""
    print("\n🌲 Training Random Forest Classifier...")
    
    # Train model with class weighting to handle imbalance
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Evaluate
    accuracy, balanced_acc, roc_auc = evaluate_model(model, X_test, y_test, "Random Forest")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Feature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, accuracy, feature_importance

def main():
    # Configuration
    data_path = "data/amp_vs_non_amp_training_dataset.csv"
    model_save_path = "models/efficacy_predictor.pkl"
    
    print("🎯 TRAINING EFFICACY PREDICTOR MODEL")
    print("=" * 60)
    
    # 1. Load and prepare data
    X, y, feature_columns = load_and_prepare_data(data_path)
    
    # 2. Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    # 3. Scale features (optional for Random Forest, but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    print("✅ Feature scaler saved as 'models/feature_scaler.pkl'")
    
    # 4. Train Random Forest model
    model, accuracy, feature_importance = train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_columns
    )
    
    # 5. Save the trained model
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved as '{model_save_path}'")
    
    # 6. Final assessment
    print(f"\n{'='*60}")
    print("🏆 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    if accuracy >= 0.85:
        print("🎉 SUCCESS: Model achieved target accuracy (>85%)!")
        print("The efficacy predictor is ready for use in your evolutionary algorithm.")
    else:
        print("⚠️  Model accuracy below target. Consider:")
        print("   - Adding more training data")
        print("   - Trying different model parameters")
        print("   - Feature engineering")
    
    print(f"\n📈 Visualizations saved:")
    print("   - random_forest_confusion_matrix.png")
    print("   - random_forest_roc_curve.png") 
    print("   - random_forest_feature_importance.png")
    print("   - feature_scaler.pkl (for scaling new data)")

if __name__ == "__main__":
    main()