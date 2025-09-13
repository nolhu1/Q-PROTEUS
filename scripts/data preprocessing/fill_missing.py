import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def calculate_missing_properties(df):
    """Calculate missing properties for sequences that need them."""
    print("Calculating missing properties...")
    
    # Identify sequences with missing Length (these are the APD3 sequences)
    missing_mask = df['Length'].isna()
    sequences_to_process = df[missing_mask]['Sequence'].tolist()
    indices_to_process = df[missing_mask].index.tolist()
    
    print(f"Found {len(sequences_to_process)} sequences with missing properties")
    
    properties_list = []
    success_count = 0
    
    for idx, seq in zip(indices_to_process, sequences_to_process):
        try:
            analysed_seq = ProteinAnalysis(seq)
            
            properties = {
                'Length': len(seq),
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
            
            # Update the dataframe
            for prop, value in properties.items():
                df.at[idx, prop] = value
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing sequence {idx}: {e}")
            continue
    
    print(f"Successfully calculated properties for {success_count}/{len(sequences_to_process)} sequences")
    
    # Remove Sequence_Length column as it's redundant with Length
    if 'Sequence_Length' in df.columns:
        df = df.drop('Sequence_Length', axis=1)
    
    return df

def clean_dataframe(df):
    """Final cleaning and validation."""
    print("Performing final cleanup...")
    
    # Remove any rows that still have missing essential properties
    essential_cols = ['Length', 'Net_Charge', 'Hydrophobicity_GRAVY']
    initial_count = len(df)
    df = df.dropna(subset=essential_cols)
    print(f"Removed {initial_count - len(df)} rows with missing essential properties")
    
    # Remove duplicates one more time
    df = df.drop_duplicates(subset=['Sequence'])
    print(f"Final dataset size: {len(df)} sequences")
    
    return df

def main():
    # Load the dataset with missing values
    input_path = "data/curated_combined_gramnegative_amps.csv"
    output_path = "data/complete_curated_combined_gramnegative_amps.csv"
    
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")
    
    # Calculate missing properties
    df = calculate_missing_properties(df)
    
    # Final cleaning
    df = clean_dataframe(df)
    
    # Save the complete dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Complete dataset saved to {output_path}")
    
    # Show final summary
    print("\n📊 FINAL DATASET SUMMARY:")
    print(f"Total sequences: {len(df)}")
    
    if 'Source' in df.columns:
        source_counts = df['Source'].value_counts()
        for source, count in source_counts.items():
            print(f"{source}: {count} sequences ({count/len(df)*100:.1f}%)")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\n❌ Remaining missing values:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  {col}: {count}")
    else:
        print("✅ No missing values remaining")

if __name__ == "__main__":
    main()