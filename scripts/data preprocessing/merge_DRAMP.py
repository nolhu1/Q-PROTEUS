import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt

def load_and_clean_dataset(file_path, dataset_name):
    """Loads a FASTA file and performs initial cleaning."""
    print(f"Loading {dataset_name}...")
    
    sequences = []
    ids = []
    
    # Use the robust parser approach we know works
    with open(file_path, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
    
    lines = content.split('\n')
    current_id, current_seq = '', ''
    
    for line in lines:
        if line.startswith('>'):
            if current_id and current_seq:
                sequences.append(current_seq)
                ids.append(current_id)
                current_seq = ''
            current_id = line[1:].strip()
        else:
            current_seq += line.strip()
    
    if current_id and current_seq:
        sequences.append(current_seq)
        ids.append(current_id)
    
    df = pd.DataFrame({'ID': ids, 'Sequence': sequences})
    df['Sequence_Length'] = df['Sequence'].str.len()
    
    print(f"  Initial count: {len(df)}")
    return df

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
        return pd.Series({  # Return NaN if calculation fails
            'Net_Charge': np.nan, 'Hydrophobicity_GRAVY': np.nan,
            'Molecular_Weight': np.nan, 'Instability_Index': np.nan,
            'Fraction_Positive_AA': np.nan, 'Fraction_Negative_AA': np.nan,
            'Fraction_Hydrophobic_AA': np.nan
        })

print("=== DRAMP Dataset Merging and Curating ===\n")

# 1. Load both datasets
natural_df = load_and_clean_dataset("data/natural_amps.fasta", "Natural AMPs")
gram_neg_df = load_and_clean_dataset("data/Anti-Gram-_amps.fasta", "Anti-Gram-negative AMPs")

# 2. Find the intersection: Natural peptides with Gram-negative activity
print("\nFinding intersection...")
merged_df = pd.merge(natural_df, gram_neg_df, on=['Sequence'], how='inner', suffixes=('_natural', '_gramneg'))
merged_df = merged_df[['Sequence', 'ID_natural']].rename(columns={'ID_natural': 'ID'})
print(f"Natural peptides with Gram-negative activity: {len(merged_df)}")

# 3. Apply rigorous filtering
print("\nApplying filters...")
# A. Remove duplicates (based on sequence)
merged_df = merged_df.drop_duplicates(subset=['Sequence'])
print(f"After removing duplicates: {len(merged_df)}")

# B. Filter by length: 10 <= length <= 80 (based on our analysis)
merged_df['Length'] = merged_df['Sequence'].str.len()
initial_count = len(merged_df)
merged_df = merged_df[(merged_df['Length'] >= 10) & (merged_df['Length'] <= 80)]
print(f"After length filtering (10-80 AA): {len(merged_df)} ({initial_count - len(merged_df)} removed)")

# C. Filter out non-standard amino acids
standard_aas = set('ACDEFGHIKLMNPQRSTVWY')
def is_valid_sequence(seq):
    return all(aa in standard_aas for aa in seq)

initial_count = len(merged_df)
merged_df = merged_df[merged_df['Sequence'].apply(is_valid_sequence)]
print(f"After removing non-standard AAs: {len(merged_df)} ({initial_count - len(merged_df)} removed)")

# 4. Calculate physicochemical properties
print("\nCalculating physicochemical properties...")
properties_df = merged_df['Sequence'].apply(calculate_properties)
final_df = pd.concat([merged_df, properties_df], axis=1)

# Remove any rows where property calculation failed
final_df = final_df.dropna()
print(f"Final dataset size after property calculation: {len(final_df)}")

# 5. Save the final curated dataset
output_path = "curated_dramp_natural_gramnegative_amps.csv"
final_df.to_csv(output_path, index=False)
print(f"\n✅ Final dataset saved to: {output_path}")
print(f"   Total sequences: {len(final_df)}")
print(f"   Sequence length range: {final_df['Length'].min()} - {final_df['Length'].max()} AA")

# 6. Show final dataset overview
print("\n=== Final Dataset Overview ===")
print(final_df[['Sequence', 'Length', 'Net_Charge', 'Hydrophobicity_GRAVY']].head().to_string())
print(f"\nKey statistics:")
print(f"  Average length: {final_df['Length'].mean():.1f} AA")
print(f"  Average net charge: {final_df['Net_Charge'].mean():.2f}")
print(f"  Average hydrophobicity (GRAVY): {final_df['Hydrophobicity_GRAVY'].mean():.3f}")

# 7. Plot distribution of key features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
final_df['Length'].hist(bins=30, ax=axes[0,0], edgecolor='black')
axes[0,0].set_title('Sequence Length Distribution')
axes[0,0].set_xlabel('Length (AA)')

final_df['Net_Charge'].hist(bins=30, ax=axes[0,1], edgecolor='black', color='green')
axes[0,1].set_title('Net Charge Distribution (pH 7.4)')
axes[0,1].set_xlabel('Net Charge')

final_df['Hydrophobicity_GRAVY'].hist(bins=30, ax=axes[1,0], edgecolor='black', color='orange')
axes[1,0].set_title('Hydrophobicity (GRAVY) Distribution')
axes[1,0].set_xlabel('GRAVY Index')

# Scatter plot: Charge vs Hydrophobicity
axes[1,1].scatter(final_df['Hydrophobicity_GRAVY'], final_df['Net_Charge'], alpha=0.6)
axes[1,1].set_title('Charge vs Hydrophobicity')
axes[1,1].set_xlabel('Hydrophobicity (GRAVY)')
axes[1,1].set_ylabel('Net Charge')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dataset_properties_overview.png', dpi=150, bbox_inches='tight')
print("\n📊 Visualization saved as 'dataset_properties_overview.png'")