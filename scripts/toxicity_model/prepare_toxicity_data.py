import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def calculate_properties(sequence):
    """Calculates physicochemical properties for a single sequence."""
    try:
        analysed_seq = ProteinAnalysis(sequence)
        net_charge = analysed_seq.charge_at_pH(7.4)
        hydrophobicity = analysed_seq.gravy()
        molecular_weight = analysed_seq.molecular_weight()
        instability_index = analysed_seq.instability_index()
        aa_comp = analysed_seq.amino_acids_percent  # Updated from deprecated method
        
        return pd.Series({
            'Net_Charge': net_charge,
            'Hydrophobicity_GRAVY': hydrophobicity,
            'Molecular_Weight': molecular_weight,
            'Instability_Index': instability_index,
            'Fraction_Positive_AA': (aa_comp.get('R', 0) + aa_comp.get('H', 0) + aa_comp.get('K', 0)),
            'Fraction_Negative_AA': (aa_comp.get('D', 0) + aa_comp.get('E', 0)),
            'Fraction_Hydrophobic_AA': (aa_comp.get('A', 0) + aa_comp.get('V', 0) + aa_comp.get('L', 0) + 
                                       aa_comp.get('I', 0) + aa_comp.get('P', 0) + aa_comp.get('F', 0) + 
                                       aa_comp.get('M', 0) + aa_comp.get('W', 0))
        })
    except Exception as e:
        print(f"Error calculating properties for sequence: {sequence[:20]}... Error: {e}")
        return pd.Series({
            'Net_Charge': np.nan, 'Hydrophobicity_GRAVY': np.nan,
            'Molecular_Weight': np.nan, 'Instability_Index': np.nan,
            'Fraction_Positive_AA': np.nan, 'Fraction_Negative_AA': np.nan,
            'Fraction_Hydrophobic_AA': np.nan
        })

def load_and_clean_data(csv_path):
    """Load and clean the toxicity dataset."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Initial dataset shape: {df.shape}")
    print(f"Columns available: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['Sequence', 'Hemolytic_Label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Basic cleaning
    initial_count = len(df)
    df = df.dropna(subset=['Sequence', 'Hemolytic_Label'])
    df = df[df['Sequence'].notna() & df['Sequence'].str.strip().astype(bool)]
    print(f"Removed {initial_count - len(df)} rows with missing data")
    
    # Filter sequences with non-standard amino acids
    standard_aas = set('ACDEFGHIKLMNPQRSTVWY')
    df = df[df['Sequence'].apply(lambda x: all(aa in standard_aas for aa in x))]
    print(f"Sequences after filtering non-standard AAs: {len(df)}")
    
    # Add length
    df['Length'] = df['Sequence'].str.len()
    
    # Filter by length (similar to AMP range)
    df = df[(df['Length'] >= 10) & (df['Length'] <= 80)]
    print(f"Sequences after length filtering (10-80 AA): {len(df)}")
    
    return df

def analyze_toxicity_distribution(df):
    """Analyze the distribution of toxicity labels and values."""
    print("\n" + "="*50)
    print("TOXICITY DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Binary label distribution
    print("Binary Label Distribution:")
    print(df['Hemolytic_Label'].value_counts())
    print(f"Toxic:Non-toxic ratio: {df['Hemolytic_Label'].value_counts().get(1, 0) / df['Hemolytic_Label'].value_counts().get(0, 1):.2f}:1")
    
    # Quantitative analysis if available
    if 'HC50_uM' in df.columns:
        print(f"\nHC50 (μM) Statistics:")
        print(f"Range: {df['HC50_uM'].min():.2f} - {df['HC50_uM'].max():.2f} μM")
        print(f"Mean: {df['HC50_uM'].mean():.2f} ± {df['HC50_uM'].std():.2f} μM")
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['HC50_uM'], bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(100, color='red', linestyle='--', label='100 μM threshold (common cutoff)')
        plt.xlabel('HC50 (μM) - Lower = More Toxic')
        plt.ylabel('Frequency')
        plt.title('Distribution of Hemolytic Activity (HC50)')
        plt.legend()
        plt.savefig('hc50_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Compare with binary labels
        if 'Hemolytic_Label' in df.columns:
            toxic_hc50 = df[df['Hemolytic_Label'] == 1]['HC50_uM']
            non_toxic_hc50 = df[df['Hemolytic_Label'] == 0]['HC50_uM']
            
            print(f"\nHC50 by Label:")
            print(f"Toxic (1): {toxic_hc50.mean():.2f} ± {toxic_hc50.std():.2f} μM")
            print(f"Non-toxic (0): {non_toxic_hc50.mean():.2f} ± {non_toxic_hc50.std():.2f} μM")

def calculate_sequence_properties(df):
    """Calculate physicochemical properties for all sequences."""
    print("\nCalculating physicochemical properties...")
    
    # Calculate properties
    properties_df = df['Sequence'].apply(calculate_properties)
    df = pd.concat([df, properties_df], axis=1)
    
    # Remove sequences that failed property calculation
    initial_count = len(df)
    df = df.dropna(subset=['Net_Charge', 'Hydrophobicity_GRAVY'])
    print(f"Removed {initial_count - len(df)} sequences with property calculation errors")
    
    return df

def analyze_property_correlations(df):
    """Analyze correlations between properties and toxicity."""
    print("\n" + "="*50)
    print("PROPERTY-TOXICITY CORRELATION ANALYSIS")
    print("="*50)
    
    # Properties to analyze
    properties = ['Net_Charge', 'Hydrophobicity_GRAVY', 'Length', 
                 'Fraction_Positive_AA', 'Fraction_Hydrophobic_AA']
    
    # Analyze correlation with binary labels
    for prop in properties:
        if prop in df.columns:
            toxic_vals = df[df['Hemolytic_Label'] == 1][prop]
            non_toxic_vals = df[df['Hemolytic_Label'] == 0][prop]
            
            print(f"\n{prop}:")
            print(f"  Toxic: {toxic_vals.mean():.2f} ± {toxic_vals.std():.2f}")
            print(f"  Non-toxic: {non_toxic_vals.mean():.2f} ± {non_toxic_vals.std():.2f}")
            
            # Point-biserial correlation (correlation between binary and continuous)
            from scipy.stats import pointbiserialr
            corr, p_value = pointbiserialr(df['Hemolytic_Label'], df[prop])
            print(f"  Correlation with toxicity: {corr:.3f} (p={p_value:.2e})")
    
    # Plot property distributions by toxicity
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    properties_to_plot = properties[:6]  # Take first 6 properties
    
    for i, prop in enumerate(properties_to_plot):
        if prop in df.columns:
            row, col = i // 3, i % 3
            for label, color in [(1, 'red'), (0, 'blue')]:
                data = df[df['Hemolytic_Label'] == label][prop]
                axes[row, col].hist(data, bins=20, alpha=0.7, color=color, 
                                   label=f"{'Toxic' if label == 1 else 'Non-toxic'} (n={len(data)})")
            axes[row, col].set_xlabel(prop)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('toxicity_property_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def prepare_final_dataset(df):
    """Prepare the final dataset for training."""
    print("\nPreparing final dataset...")
    
    # Select columns for training
    feature_columns = [
        'Length', 'Net_Charge', 'Hydrophobicity_GRAVY', 'Instability_Index',
        'Fraction_Positive_AA', 'Fraction_Negative_AA', 'Fraction_Hydrophobic_AA'
    ]
    
    # Only include columns that exist
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Create final dataset
    final_df = df[['Sequence', 'Hemolytic_Label'] + feature_columns].copy()
    
    # Add HC50 if available (for regression option)
    if 'HC50_uM' in df.columns:
        final_df['HC50_uM'] = df['HC50_uM']
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Features: {feature_columns}")
    print(f"Toxic:Non-toxic ratio: {final_df['Hemolytic_Label'].value_counts().get(1, 0)}:{final_df['Hemolytic_Label'].value_counts().get(0, 0)}")
    
    return final_df

def main():
    # Configuration
    input_csv_path = "data/hemolytic_data.csv"  # Update with your file path
    output_csv_path = "data/toxicity_training_dataset.csv"
    
    print("🧪 PREPARING TOXICITY TRAINING DATASET")
    print("=" * 60)
    
    try:
        # 1. Load and clean data
        df = load_and_clean_data(input_csv_path)
        
        # 2. Analyze toxicity distribution
        analyze_toxicity_distribution(df)
        
        # 3. Calculate sequence properties
        df = calculate_sequence_properties(df)
        
        # 4. Analyze property-toxicity correlations
        analyze_property_correlations(df)
        
        # 5. Prepare final dataset
        final_df = prepare_final_dataset(df)
        
        # 6. Save final dataset
        final_df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Final dataset saved to: {output_csv_path}")
        print(f"   Total sequences: {len(final_df)}")
        print(f"   Toxic: {len(final_df[final_df['Hemolytic_Label'] == 1])}")
        print(f"   Non-toxic: {len(final_df[final_df['Hemolytic_Label'] == 0])}")
        
        print(f"\n📊 Visualizations saved:")
        print(f"   - hc50_distribution.png (if HC50 data available)")
        print(f"   - toxicity_property_distributions.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check that your CSV has the required columns: 'Sequence' and 'Hemolytic_Label'")

if __name__ == "__main__":
    main()