import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
warnings.filterwarnings('ignore')

# Feature engineering functions
def calculate_aa_composition(sequence):
    """Calculate amino acid composition features"""
    try:
        analysis = ProteinAnalysis(sequence)
        aa_count = analysis.count_amino_acids()
        total_aa = sum(aa_count.values())
        aa_percent = {f'aa_{k}': (v/total_aa)*100 for k, v in aa_count.items()}
        return aa_percent
    except:
        return {f'aa_{aa}': 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

def calculate_physicochemical_properties(sequence):
    """Calculate various physicochemical properties from sequence"""
    try:
        analysis = ProteinAnalysis(sequence)
        properties = {}
        
        # Molecular weight
        properties['molecular_weight'] = analysis.molecular_weight()
        
        # Aromaticity
        properties['aromaticity'] = analysis.aromaticity()
        
        # Instability index
        properties['instability_index'] = analysis.instability_index()
        
        # Isoelectric point
        properties['isoelectric_point'] = analysis.isoelectric_point()
        
        # Secondary structure fraction
        sec_struct = analysis.secondary_structure_fraction()
        properties['helix_fraction'] = sec_struct[0]
        properties['turn_fraction'] = sec_struct[1]
        properties['sheet_fraction'] = sec_struct[2]
        
        # Molar extinction coefficient
        ext_coeff = analysis.molar_extinction_coefficient()
        properties['extinction_coefficient_reduced'] = ext_coeff[0]
        properties['extinction_coefficient_oxidized'] = ext_coeff[1]
        
        # Gravy (Grand Average of Hydropathicity)
        properties['gravy'] = analysis.gravy()
        
        return properties
    except:
        return {
            'molecular_weight': 0, 'aromaticity': 0, 'instability_index': 0,
            'isoelectric_point': 0, 'helix_fraction': 0, 'turn_fraction': 0,
            'sheet_fraction': 0, 'extinction_coefficient_reduced': 0,
            'extinction_coefficient_oxidized': 0, 'gravy': 0
        }

def calculate_sequence_features(sequence):
    """Calculate additional sequence-based features"""
    if not sequence or pd.isna(sequence):
        return {
            'sequence_length': 0, 'charge_density': 0, 'hydrophobic_residues': 0,
            'positive_residues': 0, 'negative_residues': 0, 'polar_residues': 0
        }
    
    sequence_length = len(sequence)
    
    # Count different types of residues
    hydrophobic = sum(1 for aa in sequence if aa in ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'])
    positive = sum(1 for aa in sequence if aa in ['K', 'R', 'H'])
    negative = sum(1 for aa in sequence if aa in ['D', 'E'])
    polar = sum(1 for aa in sequence if aa in ['S', 'T', 'N', 'Q', 'Y', 'C'])
    
    return {
        'sequence_length': sequence_length,
        'charge_density': (positive - negative) / sequence_length if sequence_length > 0 else 0,
        'hydrophobic_residues': hydrophobic / sequence_length if sequence_length > 0 else 0,
        'positive_residues': positive / sequence_length if sequence_length > 0 else 0,
        'negative_residues': negative / sequence_length if sequence_length > 0 else 0,
        'polar_residues': polar / sequence_length if sequence_length > 0 else 0
    }

def prepare_features(df):
    """Prepare features for model training"""
    # Create a copy of the dataframe
    features_df = df.copy()
    
    # Drop rows with missing target values
    features_df = features_df.dropna(subset=['hemolytic_activity'])
    
    # Extract features from sequences
    aa_features = []
    physchem_features = []
    seq_features = []
    
    for _, row in features_df.iterrows():
        sequence = row['sequence']
        
        # Amino acid composition
        aa_features.append(calculate_aa_composition(sequence))
        
        # Physicochemical properties
        physchem_features.append(calculate_physicochemical_properties(sequence))
        
        # Sequence-based features
        seq_features.append(calculate_sequence_features(sequence))
    
    # Convert to dataframes
    aa_df = pd.DataFrame(aa_features)
    physchem_df = pd.DataFrame(physchem_features)
    seq_df = pd.DataFrame(seq_features)
    
    # Combine all features
    features = pd.concat([aa_df, physchem_df, seq_df], axis=1)
    
    # Add original non-sequence features
    numeric_features = features_df[['hemolytic_concentration']].copy()
    numeric_features['hemolytic_concentration'] = pd.to_numeric(
        numeric_features['hemolytic_concentration'], errors='coerce'
    ).fillna(0)
    
    # Combine all features
    all_features = pd.concat([features, numeric_features], axis=1)
    
    # Target variable
    target = pd.to_numeric(features_df['hemolytic_activity'], errors='coerce')
    
    # For classification, create a binary target (toxic vs non-toxic)
    # Using median activity as threshold
    threshold = target.median()
    binary_target = (target > threshold).astype(int)
    
    return all_features, target, binary_target

def train_regression_models(X, y):
    """Train and evaluate regression models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    return results, X_test, y_test

def train_classification_models(X, y):
    """Train and evaluate classification models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='rbf', probability=True),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}")
        print(report)
    
    return results, X_test, y_test

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.barh(range(top_n), importances[indices][:top_n][::-1], color='b', align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n][::-1]])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("This model doesn't support feature importance visualization.")

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix for classification results"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load the data
    df = pd.read_csv('data/toxicity_data.csv')
    
    # Prepare features
    print("Preparing features...")
    X, y_reg, y_clf = prepare_features(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    # Check for class distribution in classification target
    print("\nClass distribution:")
    print(y_clf.value_counts())
    
    # Train regression models
    print("\nTraining regression models...")
    reg_results, X_test_reg, y_test_reg = train_regression_models(X, y_reg)
    
    # Train classification models
    print("\nTraining classification models...")
    clf_results, X_test_clf, y_test_clf = train_classification_models(X, y_clf)
    
    # Plot feature importance for the best regression model
    best_reg_name = max(reg_results.keys(), key=lambda k: reg_results[k]['r2'])
    print(f"\nBest regression model: {best_reg_name}")
    plot_feature_importance(reg_results[best_reg_name]['model'], X.columns)
    
    # Plot confusion matrix for the best classification model
    best_clf_name = max(clf_results.keys(), key=lambda k: clf_results[k]['accuracy'])
    print(f"Best classification model: {best_clf_name}")
    plot_confusion_matrix(y_test_clf, clf_results[best_clf_name]['predictions'], best_clf_name)
    
    # Cross-validation for the best models
    print("\nCross-validation results:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # For regression
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_scores = cross_val_score(rf_reg, X_scaled, y_reg, cv=5, scoring='r2')
    print(f"Random Forest Regression CV R²: {reg_scores.mean():.4f} (±{reg_scores.std():.4f})")
    
    # For classification
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_scores = cross_val_score(rf_clf, X_scaled, y_clf, cv=5, scoring='accuracy')
    print(f"Random Forest Classification CV Accuracy: {clf_scores.mean():.4f} (±{clf_scores.std():.4f})")
    
    # Hyperparameter tuning for the best model (Random Forest)
    print("\nPerforming hyperparameter tuning for Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # For regression
    grid_search_reg = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid, cv=3, scoring='r2', n_jobs=-1
    )
    grid_search_reg.fit(X_scaled, y_reg)
    print(f"Best parameters for regression: {grid_search_reg.best_params_}")
    print(f"Best CV score for regression: {grid_search_reg.best_score_:.4f}")
    
    # For classification
    grid_search_clf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid_search_clf.fit(X_scaled, y_clf)
    print(f"Best parameters for classification: {grid_search_clf.best_params_}")
    print(f"Best CV score for classification: {grid_search_clf.best_score_:.4f}")

if __name__ == "__main__":
    main()