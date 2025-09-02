import pandas as pd
import numpy as np
from Bio import SeqIO
import requests
from io import StringIO

def download_human_proteome():
    """Download human protein sequences from UniProt as non-AMPs."""
    print("Downloading human proteome as non-AMP examples...")
    
    # UniProt API query for human reviewed proteins
    url = "https://www.uniprot.org/uniprot/?query=reviewed:yes+organism:9606&format=fasta&limit=2000"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Parse FASTA
            sequences = []
            ids = []
            for record in SeqIO.parse(StringIO(response.text), "fasta"):
                seq_str = str(record.seq)
                # Filter for appropriate length (similar to AMPs)
                if 10 <= len(seq_str) <= 80:
                    sequences.append(seq_str)
                    ids.append(record.id)
            
            print(f"Downloaded {len(sequences)} human proteins as non-AMPs")
            return sequences, ids
        else:
            print("Failed to download from UniProt")
            return None, None
    except Exception as e:
        print(f"Error downloading: {e}")
        return None, None

def calculate_properties_for_sequences(sequences, ids, label=0):
    """Calculate properties for a list of sequences."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    
    data = []
    valid_sequences = []
    valid_ids = []
    
    for seq, seq_id in zip(sequences, ids):
        try:
            analysed_seq = ProteinAnalysis(seq)
            
            properties = {
                'ID': seq_id,
                'Sequence': seq,
                'Length': len(seq),
                'Net_Charge': analysed_seq.charge_at_pH(7.4),
                'Hydrophobicity_GRAVY': analysed_seq.gravy(),
                'Molecular_Weight': analysed_seq.molecular_weight(),
                'Instability_Index': analysed_seq.instability_index(),
                'Fraction_Positive_AA': (analysed_seq.get_amino_acids_percent().get('R', 0) + 
                                        analysed_seq.get_amino_acids_percent().get('H', 0) + 
                                        analysed_seq.get_amino_acids_percent().get('K', 0)),
                'Fraction_Negative_AA': (analysed_seq.get_amino_acids_percent().get('D', 0) + 
                                        analysed_seq.get_amino_acids_percent().get('E', 0)),
                'Fraction_Hydrophobic_AA': (analysed_seq.get_amino_acids_percent().get('A', 0) + 
                                           analysed_seq.get_amino_acids_percent().get('V', 0) + 
                                           analysed_seq.get_amino_acids_percent().get('L', 0) + 
                                           analysed_seq.get_amino_acids_percent().get('I', 0) + 
                                           analysed_seq.get_amino_acids_percent().get('P', 0) + 
                                           analysed_seq.get_amino_acids_percent().get('F', 0) + 
                                           analysed_seq.get_amino_acids_percent().get('M', 0) + 
                                           analysed_seq.get_amino_acids_percent().get('W', 0)),
                'is_amp': label
            }
            
            data.append(properties)
            valid_sequences.append(seq)
            valid_ids.append(seq_id)
            
        except Exception as e:
            # Skip sequences that can't be analyzed
            continue
    
    df = pd.DataFrame(data)
    print(f"Successfully processed {len(df)} sequences")
    return df

def main():
    """Main function to get real non-AMP data."""
    # Download human proteins as non-AMPs
    non_amp_sequences, non_amp_ids = download_human_proteome()
    
    if non_amp_sequences:
        # Calculate properties for non-AMPs
        non_amp_df = calculate_properties_for_sequences(non_amp_sequences, non_amp_ids, label=0)
        non_amp_df.to_csv('real_non_amp_data.csv', index=False)
        print("✅ Real non-AMP data saved to 'real_non_amp_data.csv'")
        
        # Load your AMP data
        amp_df = pd.read_csv("data/complete_curated_combined_gramnegative_amps.csv")
        amp_df['is_amp'] = 1  # Label as AMPs
        
        # Combine datasets
        combined_df = pd.concat([amp_df, non_amp_df], ignore_index=True)
        combined_df.to_csv('data/amp_vs_nonamp_training_data.csv', index=False)
        print("✅ Combined training data saved to 'amp_vs_nonamp_training_data.csv'")
        print(f"Total sequences: {len(combined_df)}")
        print(f"AMPs: {len(amp_df)}, Non-AMPs: {len(non_amp_df)}")

if __name__ == "__main__":
    main()