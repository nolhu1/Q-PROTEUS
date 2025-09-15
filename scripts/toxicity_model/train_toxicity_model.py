import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

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
    
    return all_features, binary_target

def analyze_population_diversity(population_sequences, model, scaler, feature_columns):
    """Analyze diversity of population sequences"""
    if not population_sequences:
        return {"diversity_score": 0, "unique_sequences": 0, "avg_toxicity_prob": 0.5}
    
    # Extract features for all sequences
    population_features = []
    for seq in population_sequences:
        features = extract_features_single_sequence(seq)
        population_features.append(features)
    
    # Convert to DataFrame and ensure correct column order
    population_df = pd.DataFrame(population_features)
    population_df = population_df.reindex(columns=feature_columns, fill_value=0)
    
    # Scale features
    population_scaled = scaler.transform(population_df)
    
    # Predict toxicity probabilities
    toxicity_probs = model.predict_proba(population_scaled)[:, 1]  # Probability of being non-toxic
    
    # Calculate diversity metrics
    unique_sequences = len(set(population_sequences))
    diversity_score = unique_sequences / len(population_sequences)
    avg_toxicity_prob = np.mean(toxicity_probs)
    
    return {
        "diversity_score": diversity_score,
        "unique_sequences": unique_sequences,
        "avg_toxicity_prob": avg_toxicity_prob,
        "toxicity_probs": toxicity_probs
    }

def extract_features_single_sequence(sequence):
    """Extract features for a single peptide sequence"""
    # Amino acid composition
    aa_features = calculate_aa_composition(sequence)
    
    # Physicochemical properties
    physchem_features = calculate_physicochemical_properties(sequence)
    
    # Sequence-based features
    seq_features = calculate_sequence_features(sequence)
    
    # Combine all features (no experimental data for new sequences)
    all_features = {**aa_features, **physchem_features, **seq_features, 'hemolytic_concentration': 0}
    
    return all_features

def create_toxicity_predictor(sequence, model, scaler, feature_columns):
    """Predict toxicity probability for a single sequence"""
    features = extract_features_single_sequence(sequence)
    
    # Convert to DataFrame with correct column order
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=feature_columns, fill_value=0)
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Predict probability of being non-toxic (class 1)
    probability_non_toxic = model.predict_proba(features_scaled)[0, 1]
    
    return probability_non_toxic

def main():
    # Load the data
    df = pd.read_csv('data/toxicity_data.csv')
    
    # Prepare features
    print("Preparing features...")
    X, y = prepare_features(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    # Check for class distribution
    print("\nClass distribution:")
    print(y.value_counts())
    print(f"Baseline accuracy: {max(y.value_counts(normalize=True)):.3f}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier (best performing model)
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced'  # Handle any class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (non-toxic)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    print("\nCross-validation results:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save model and scaler
    joblib.dump(model, 'models/toxicity_predictor.pkl')
    joblib.dump(scaler, 'models/toxicity_feature_scaler.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')
    
    print("\nModels saved successfully!")
    print("Files created:")
    print("  - models/toxicity_predictor.pkl (classification model)")
    print("  - models/toxicity_feature_scaler.pkl (feature scaler)")
    print("  - models/feature_columns.pkl (feature column names)")
    
    # Test the predictor with example sequences
    print("\nTesting predictor with example sequences...")
    test_sequences = [
        "SKEKIGKEFKRIVQRIKDFLR",  # Known non-toxic from training
        "AAAAAAAAAAAAAAAAAAAA",    # Simple poly-alanine (likely non-toxic)
        "KKKKKKKKKKKKKKKKKKKK",    # Poly-lysine (might be toxic)
    ]
    
    for seq in test_sequences:
        prob = create_toxicity_predictor(seq, model, scaler, X.columns)
        print(f"Sequence: {seq[:20]}... -> Non-toxic probability: {prob:.3f}")
    
    # Demonstrate population diversity analysis
    print("\nTesting population diversity analysis...")
    population = test_sequences * 3  # Create a small population
    diversity = analyze_population_diversity(population, model, scaler, X.columns)
    print(f"Population diversity: {diversity['diversity_score']:.3f}")
    print(f"Unique sequences: {diversity['unique_sequences']}")
    print(f"Average non-toxic probability: {diversity['avg_toxicity_prob']:.3f}")

if __name__ == "__main__":
    main()