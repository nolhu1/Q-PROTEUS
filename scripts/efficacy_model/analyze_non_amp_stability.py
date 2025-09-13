import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Load your existing function for calculating properties
# Make sure this function is defined or imported
def calculate_properties(sequence):
    """Calculates physicochemical properties for a single sequence."""
    try:
        analysed_seq = ProteinAnalysis(sequence)
        net_charge = analysed_seq.charge_at_pH(7.4)
        hydrophobicity = analysed_seq.gravy()
        molecular_weight = analysed_seq.molecular_weight()
        instability_index = analysed_seq.instability_index()
        aa_comp = analysed_seq.get_amino_acids_percent()
        
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
    except:
        return pd.Series({
            'Net_Charge': np.nan, 'Hydrophobicity_GRAVY': np.nan,
            'Molecular_Weight': np.nan, 'Instability_Index': np.nan,
            'Fraction_Positive_AA': np.nan, 'Fraction_Negative_AA': np.nan,
            'Fraction_Hydrophobic_AA': np.nan
        })

def load_and_process_fasta(fasta_path, label, source_name):
    """Load a FASTA file and calculate properties."""
    from Bio import SeqIO
    
    print(f"Processing {source_name}...")
    sequences = []
    ids = []
    
    filtered = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq) and 10 <= len(seq) <= 80:
            sequences.append(seq)
            ids.append(record.id)
        else:
            filtered += 1
    print(f"  Filtered out {filtered} sequences due to non-standard AAs or length")
    
    df = pd.DataFrame({'ID': ids, 'Sequence': sequences, 'Source': source_name, 'Label': label})
    df['Length'] = df['Sequence'].str.len()
    
    # Calculate properties
    properties_df = df['Sequence'].apply(calculate_properties)
    df = pd.concat([df, properties_df], axis=1)
    df = df.dropna()  # Remove sequences that failed calculation
    
    print(f"  Successfully processed {len(df)} sequences")
    return df

def main():
    # Configuration
    amp_file_path = "data/complete_curated_combined_gramnegative_amps.csv"  # Your AMPs
    non_amp_fasta_path = "data/uniprotkb_length_10_TO_55_AND_reviewed_2025_09_03.fasta"  # Your new non-AMPs
    
    print("=== ANALYZING NON-AMP DATASET SUITABILITY ===")
    
    # 1. Load AMP data (from your existing CSV)
    print("\n1. Loading AMP data...")
    amp_df = pd.read_csv(amp_file_path)
    amp_df['Label'] = 1  # AMPs
    amp_df['Source'] = 'AMP_DB'
    print(f"   AMP sequences: {len(amp_df)}")
    
    # 2. Load and process non-AMP data (from FASTA)
    non_amp_df = load_and_process_fasta(non_amp_fasta_path, label=0, source_name='UniProt_Non_AMP')
    
    # 3. Combine datasets for analysis
    combined_df = pd.concat([amp_df, non_amp_df], ignore_index=True)
    print(f"\n   Total sequences for analysis: {len(combined_df)}")
    print(f"   - AMPs: {len(amp_df)}")
    print(f"   - Non-AMPs: {len(non_amp_df)}")
    
    # 4. Basic property comparison
    print("\n2. Property Comparison:")
    properties = ['Length', 'Net_Charge', 'Hydrophobicity_GRAVY', 'Instability_Index']
    
    for prop in properties:
        amp_vals = combined_df[combined_df['Label'] == 1][prop]
        non_amp_vals = combined_df[combined_df['Label'] == 0][prop]
        
        t_stat, p_value = stats.ttest_ind(amp_vals, non_amp_vals, nan_policy='omit')
        
        print(f"   {prop}:")
        print(f"     AMPs: {amp_vals.mean():.2f} ± {amp_vals.std():.2f}")
        print(f"     Non-AMPs: {non_amp_vals.mean():.2f} ± {non_amp_vals.std():.2f}")
        print(f"     T-test p-value: {p_value:.2e} {'(SIGNIFICANT)' if p_value < 0.05 else '(NOT significant)'}")
    
    # 5. Check for obvious separability (the "cheating" test)
    print("\n3. Checking for Obvious Separability:")
    
    # Calculate mean difference in key AMP properties
    charge_diff = abs(amp_df['Net_Charge'].mean() - non_amp_df['Net_Charge'].mean())
    hydrophobicity_diff = abs(amp_df['Hydrophobicity_GRAVY'].mean() - non_amp_df['Hydrophobicity_GRAVY'].mean())
    
    print(f"   Mean charge difference: {charge_diff:.2f}")
    print(f"   Mean hydrophobicity difference: {hydrophobicity_diff:.2f}")
    
    if charge_diff > 3.0 or hydrophobicity_diff > 1.0:
        print("   ⚠️  LARGE differences detected - model might learn simple rules")
    else:
        print("   ✅ Moderate differences - model will need to learn complex patterns")
    
    # 6. Visualize the distributions
    print("\n4. Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('AMP vs Non-AMP Property Distributions', fontsize=16)
    
    # Plot key properties
    properties_to_plot = ['Net_Charge', 'Hydrophobicity_GRAVY', 'Length', 'Fraction_Positive_AA']
    for i, prop in enumerate(properties_to_plot):
        row, col = i // 2, i % 2
        for label, color in [(1, 'red'), (0, 'blue')]:
            data = combined_df[combined_df['Label'] == label][prop]
            axes[row, col].hist(data, bins=30, alpha=0.7, color=color, 
                               label=f"{'AMP' if label == 1 else 'Non-AMP'} (n={len(data)})")
        axes[row, col].set_xlabel(prop)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('amp_vs_non_amp_properties.png', dpi=300, bbox_inches='tight')
    
    # 7. PCA to check separability
    print("\n5. Running PCA to check for complex separability...")
    
    # Prepare features for PCA
    features = ['Net_Charge', 'Hydrophobicity_GRAVY', 'Fraction_Positive_AA', 
                'Fraction_Negative_AA', 'Fraction_Hydrophobic_AA', 'Length']
    
    X = combined_df[features].fillna(0)
    y = combined_df['Label']
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot PCA
    plt.figure(figsize=(10, 8))
    for label, color, marker in [(1, 'red', 'o'), (0, 'blue', '^')]:
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=color, marker=marker, alpha=0.6,
                   label=f"{'AMP' if label == 1 else 'Non-AMP'}")
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: AMPs vs Non-AMPs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pca_amp_vs_non_amp.png', dpi=300, bbox_inches='tight')
    
    # 8. Final assessment
    print("\n6. FINAL ASSESSMENT:")
    print("=" * 50)
    
    overlap_score = 0
    for prop in ['Net_Charge', 'Hydrophobicity_GRAVY', 'Fraction_Positive_AA']:
        amp_mean = amp_df[prop].mean()
        non_amp_mean = non_amp_df[prop].mean()
        overlap = min(amp_mean, non_amp_mean) / max(amp_mean, non_amp_mean) if max(amp_mean, non_amp_mean) != 0 else 0
        overlap_score += overlap
    
    overlap_score = overlap_score / 3  # Average overlap
    
    print(f"   Dataset size ratio: {len(non_amp_df)} non-AMPs / {len(amp_df)} AMPs = {len(non_amp_df)/len(amp_df):.2f}")
    print(f"   Property overlap score: {overlap_score:.3f} (1.0 = identical, 0.0 = completely different)")
    
    if len(non_amp_df) >= 0.8 * len(amp_df) and overlap_score > 0.5:
        print("   ✅ EXCELLENT - Well-balanced dataset with good property overlap")
        print("   ✅ Model will need to learn non-trivial patterns")
        print("   ✅ Low risk of 'cheating' or simple rule learning")
    elif len(non_amp_df) >= 0.5 * len(amp_df):
        print("   ✅ GOOD - Adequate dataset size")
        print("   ⚠️  Model might still learn some simple patterns")
    else:
        print("   ⚠️  CAUTION - Dataset may be too small or imbalanced")
        print("   ❌ Consider finding more non-AMP sequences")
    
    print(f"\nVisualizations saved:")
    print(f"   - amp_vs_non_amp_properties.png")
    print(f"   - pca_amp_vs_non_amp.png")
    # 9. Save the combined dataset for training
    print("\n7. Saving combined dataset for training...")
    
    # Select only the columns we need for training
    training_columns = ['Sequence', 'Label', 'Length', 'Net_Charge', 'Hydrophobicity_GRAVY', 
                       'Instability_Index', 'Fraction_Positive_AA', 'Fraction_Negative_AA', 
                       'Fraction_Hydrophobic_AA', 'Source']
    
    # Filter to only include these columns if they exist
    training_df = combined_df[[col for col in training_columns if col in combined_df.columns]]
    
    # Save to CSV
    training_csv_path = "data/amp_vs_non_amp_training_dataset.csv"
    training_df.to_csv(training_csv_path, index=False)
    print(f"   ✅ Training dataset saved to: {training_csv_path}")
    print(f"   📊 Final training set size: {len(training_df)} sequences")
    print(f"   🏷️  Class balance: {training_df['Label'].value_counts().to_dict()}")
if __name__ == "__main__":
    main()