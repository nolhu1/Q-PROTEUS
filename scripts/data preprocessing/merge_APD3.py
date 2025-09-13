import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def load_apd3_fasta(apd3_fasta_path):
    """
    Load and clean APD3 FASTA file specifically for Gram-negative AMPs
    """
    print("Loading APD3 Gram-negative AMP data...")
    
    sequences = []
    ids = []
    descriptions = []
    
    # Use robust parsing to handle encoding issues
    with open(apd3_fasta_path, 'rb') as f:
        content = f.read().decode('utf-8', errors='ignore')
    
    lines = content.split('\n')
    current_id, current_seq, current_desc = '', '', ''
    
    for line in lines:
        if line.startswith('>'):
            if current_id and current_seq:
                sequences.append(current_seq)
                ids.append(current_id)
                descriptions.append(current_desc)
                current_seq = ''
            header = line[1:].strip()
            current_id = header.split('|')[0] if '|' in header else header
            current_desc = header
        else:
            current_seq += line.strip()
    
    if current_id and current_seq:
        sequences.append(current_seq)
        ids.append(current_id)
        descriptions.append(current_desc)
    
    apd3_df = pd.DataFrame({
        'ID': ids,
        'Description': descriptions,
        'Sequence': sequences,
        'Source': 'APD3'  # Add source identifier
    })
    
    apd3_df['Sequence_Length'] = apd3_df['Sequence'].str.len()
    print(f"  Raw APD3 entries: {len(apd3_df)}")
    
    return apd3_df

def standardize_sequence(seq):
    """Standardize sequence representation for duplicate detection"""
    return seq.upper().replace(' ', '').replace('-', '')

def find_and_remove_duplicates(existing_df, new_df, source_name):
    """
    Identify duplicates between existing and new datasets
    Returns: (non_duplicate_df, duplicate_count)
    """
    # Standardize sequences for comparison
    existing_sequences = set(existing_df['Sequence'].apply(standardize_sequence))
    new_sequences = set(new_df['Sequence'].apply(standardize_sequence))
    
    # Find overlap
    overlapping_sequences = existing_sequences & new_sequences
    print(f"  Found {len(overlapping_sequences)} duplicate sequences already in existing dataset")
    
    # Filter out duplicates from new dataset
    new_df['Standardized_Seq'] = new_df['Sequence'].apply(standardize_sequence)
    non_duplicate_df = new_df[~new_df['Standardized_Seq'].isin(existing_sequences)]
    non_duplicate_df = non_duplicate_df.drop('Standardized_Seq', axis=1)
    
    return non_duplicate_df, len(overlapping_sequences)

def calculate_properties_for_apd3(apd3_df):
    """Calculate physicochemical properties for APD3 sequences"""
    print("Calculating properties for APD3 sequences...")
    
    properties_list = []
    valid_indices = []
    
    for idx, row in apd3_df.iterrows():
        try:
            analysed_seq = ProteinAnalysis(row['Sequence'])
            properties = {
                'Net_Charge': analysed_seq.charge_at_pH(7.4),
                'Hydrophobicity_GRAVY': analysed_seq.gravy(),
                'Molecular_Weight': analysed_seq.molecular_weight(),
                'Instability_Index': analysed_seq.instability_index(),
            }
            
            # Add amino acid composition features
            aa_comp = analysed_seq.get_amino_acids_percent()
            properties.update({
                'Fraction_Positive_AA': (aa_comp.get('R', 0) + aa_comp.get('H', 0) + aa_comp.get('K', 0)),
                'Fraction_Negative_AA': (aa_comp.get('D', 0) + aa_comp.get('E', 0)),
                'Fraction_Hydrophobic_AA': (aa_comp.get('A', 0) + aa_comp.get('V', 0) + aa_comp.get('L', 0) + 
                                           aa_comp.get('I', 0) + aa_comp.get('P', 0) + aa_comp.get('F', 0) + 
                                           aa_comp.get('M', 0) + aa_comp.get('W', 0))
            })
            
            properties_list.append(properties)
            valid_indices.append(idx)
            
        except Exception as e:
            # Skip sequences that can't be analyzed
            continue
    
    # Create properties DataFrame
    properties_df = pd.DataFrame(properties_list, index=valid_indices)
    
    # Merge with original APD3 data
    apd3_with_properties = apd3_df.loc[valid_indices].copy()
    apd3_with_properties = pd.concat([apd3_with_properties, properties_df], axis=1)
    
    print(f"  Successfully calculated properties for {len(apd3_with_properties)}/{len(apd3_df)} sequences")
    return apd3_with_properties

def main():
    # Configuration
    curated_dramp_path = "curated_dramp_natural_gramnegative_amps.csv"
    apd3_fasta_path = "data/apd3.fasta"  # Your APD3 Gram-negative FASTA file
    output_path = "curated_combined_gramnegative_amps.csv"
    
    print("=== Merging APD3 Data with Curated DRAMP Dataset ===\n")
    
    # 1. Load existing curated DRAMP dataset
    print("Loading existing curated DRAMP dataset...")
    dramp_df = pd.read_csv(curated_dramp_path)
    dramp_df['Source'] = 'DRAMP'  # Add source identifier
    print(f"  Existing DRAMP sequences: {len(dramp_df)}")
    
    # 2. Load and clean APD3 data
    apd3_df = load_apd3_fasta(apd3_fasta_path)
    
    # 3. Apply the same rigorous filtering to APD3 data
    print("\nFiltering APD3 data...")
    
    # A. Length filter (10-80 AA)
    initial_count = len(apd3_df)
    apd3_df = apd3_df[(apd3_df['Sequence_Length'] >= 10) & (apd3_df['Sequence_Length'] <= 80)]
    print(f"  After length filtering: {len(apd3_df)} ({initial_count - len(apd3_df)} removed)")
    
    # B. Remove non-standard amino acids
    standard_aas = set('ACDEFGHIKLMNPQRSTVWY')
    def is_valid_sequence(seq):
        return all(aa in standard_aas for aa in seq.upper())
    
    initial_count = len(apd3_df)
    apd3_df = apd3_df[apd3_df['Sequence'].apply(is_valid_sequence)]
    print(f"  After removing non-standard AAs: {len(apd3_df)} ({initial_count - len(apd3_df)} removed)")
    
    # 4. Remove duplicates between APD3 and existing DRAMP dataset
    print("\nChecking for duplicates between APD3 and DRAMP...")
    apd3_non_duplicate, duplicate_count = find_and_remove_duplicates(dramp_df, apd3_df, "APD3")
    print(f"  Unique APD3 sequences to add: {len(apd3_non_duplicate)}")
    
    if len(apd3_non_duplicate) == 0:
        print("No new sequences to add from APD3.")
        dramp_df.to_csv(output_path, index=False)
        return
    
    # 5. Calculate properties for APD3 sequences
    apd3_with_properties = calculate_properties_for_apd3(apd3_non_duplicate)
    
    # 6. Ensure column consistency between datasets
    print("\nEnsuring column consistency...")
    
    # Get all unique columns from both datasets
    all_columns = set(dramp_df.columns) | set(apd3_with_properties.columns)
    
    # Add missing columns to each dataset with NaN values
    for col in all_columns:
        if col not in dramp_df.columns:
            dramp_df[col] = np.nan
        if col not in apd3_with_properties.columns:
            apd3_with_properties[col] = np.nan
    
    # Reorder columns to match DRAMP dataset
    apd3_with_properties = apd3_with_properties[dramp_df.columns]
    
    # 7. Merge the datasets
    print("Merging datasets...")
    combined_df = pd.concat([dramp_df, apd3_with_properties], ignore_index=True)
    
    # 8. Final cleanup - remove any potential duplicates that might have been missed
    combined_df = combined_df.drop_duplicates(subset=['Sequence'], keep='first')
    
    # 9. Save the final combined dataset
    combined_df.to_csv(output_path, index=False)
    
    # 10. Generate summary report
    print(f"\n✅ Merge completed successfully!")
    print(f"   Final dataset saved to: {output_path}")
    print(f"   Total sequences: {len(combined_df)}")
    print(f"   - DRAMP sequences: {len(combined_df[combined_df['Source'] == 'DRAMP'])}")
    print(f"   - APD3 sequences: {len(combined_df[combined_df['Source'] == 'APD3'])}")
    print(f"   - Duplicates prevented: {duplicate_count}")
    
    # Show overview of the combined dataset
    print(f"\n=== Combined Dataset Overview ===")
    print(f"Sequence length range: {combined_df['Length'].min()} - {combined_df['Length'].max()} AA")
    print(f"Average length: {combined_df['Length'].mean():.1f} AA")
    print(f"Average net charge: {combined_df['Net_Charge'].mean():.2f}")
    print(f"Average hydrophobicity (GRAVY): {combined_df['Hydrophobicity_GRAVY'].mean():.3f}")
    
    # Show first few rows of APD3 data that was added
    apd3_added = combined_df[combined_df['Source'] == 'APD3'].head(3)
    if len(apd3_added) > 0:
        print(f"\nSample of APD3 sequences added:")
        print(apd3_added[['ID', 'Sequence', 'Length', 'Net_Charge']].to_string(index=False))

if __name__ == "__main__":
    main()