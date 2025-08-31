import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration - UPDATE THESE PATHS TO MATCH WHERE YOU SAVED THE FILES
natural_fasta_path = "data/natural_amps.fasta"  # or "raw_data/dramp_natural_amps.fasta"
gram_neg_fasta_path = "data/Anti-Gram-_amps.fasta"  # or "raw_data/dramp_anti_gram_negative.fasta"

# 0. First, let's just check if the files exist and are readable
print("Initial File Check")
print("=" * 50)
for file_path in [natural_fasta_path, gram_neg_fasta_path]:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"✓ Found '{file_path}' ({file_size:.2f} MB)")
    else:
        print(f"✗ Missing: '{file_path}'")
        print("Please download the files from DRAMP and place them in the correct location.")
        exit()

print("\n\033[1m=== Beginning Data Analysis ===\033[0m")

# 1. Robust FASTA parser that handles encoding issues
def robust_fasta_parser(file_path):
    """
    A more robust FASTA parser that handles encoding issues and provides diagnostics.
    """
    print(f"Attempting to parse: {file_path}")
    
    sequences = []
    ids = []
    descriptions = []
    problematic_entries = []
    
    try:
        for i, record in enumerate(SeqIO.parse(file_path, "fasta")):
            try:
                seq_str = str(record.seq)
                # Basic validation: check if sequence contains only expected characters
                if len(seq_str) == 0:
                    problematic_entries.append({'id': record.id, 'issue': 'Empty sequence'})
                    continue
                    
                sequences.append(seq_str)
                ids.append(record.id)
                descriptions.append(record.description)
                
            except UnicodeDecodeError as e:
                problematic_entries.append({'id': f'record_{i}', 'issue': f'Unicode error: {str(e)}'})
            except Exception as e:
                problematic_entries.append({'id': record.id if hasattr(record, 'id') else f'record_{i}', 'issue': f'Other error: {str(e)}'})
                
    except UnicodeDecodeError:
        print(f"Major encoding issue with {file_path}. Trying alternative approach...")
        # Fallback: read as binary and decode with error handling
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        # Simple FASTA parser for emergency use
        lines = content.split('\n')
        current_id, current_seq = '', ''
        for line in lines:
            if line.startswith('>'):
                if current_id:
                    sequences.append(current_seq)
                    ids.append(current_id)
                    descriptions.append(current_id)  # Simplified
                    current_seq = ''
                current_id = line[1:].strip()
            else:
                current_seq += line.strip()
        if current_id and current_seq:
            sequences.append(current_seq)
            ids.append(current_id)
            descriptions.append(current_id)
    
    return pd.DataFrame({'ID': ids, 'Description': descriptions, 'Sequence': sequences}), problematic_entries

# 2. Analyze the FASTA file with comprehensive diagnostics
def analyze_fasta_file(file_path, dataset_name):
    """
    Performs a comprehensive analysis of a FASTA file.
    Returns a DataFrame and a dictionary of statistics.
    """
    print(f"\n\033[1m=== Analyzing {dataset_name} Dataset ===\033[0m")
    
    # Parse the file with our robust parser
    df, problematic_entries = robust_fasta_parser(file_path)
    
    if len(df) == 0:
        print("❌ No valid sequences found in the file!")
        return pd.DataFrame(), {}
    
    df['Sequence_Length'] = df['Sequence'].str.len()
    
    # Calculate basic statistics
    stats = {
        'total_entries': len(df),
        'unique_sequences': df['Sequence'].nunique(),
        'unique_ids': df['ID'].nunique(),
        'avg_sequence_length': df['Sequence_Length'].mean(),
        'min_sequence_length': df['Sequence_Length'].min(),
        'max_sequence_length': df['Sequence_Length'].max(),
        'duplicate_sequences': len(df) - df['Sequence'].nunique(),
        'problematic_entries': len(problematic_entries),
    }
    
    # Print the statistics
    print(f"✅ Successfully parsed entries: {stats['total_entries']}")
    print(f"✅ Unique sequences: {stats['unique_sequences']}")
    print(f"⚠️  Duplicate sequences: {stats['duplicate_sequences']}")
    print(f"❌ Problematic entries: {stats['problematic_entries']}")
    print(f"📏 Average sequence length: {stats['avg_sequence_length']:.2f}")
    print(f"📏 Min length: {stats['min_sequence_length']}")
    print(f"📏 Max length: {stats['max_sequence_length']}")
    
    # Show problematic entries if any
    if problematic_entries:
        print(f"\n❌ Problematic entries found:")
        for entry in problematic_entries[:5]:  # Show first 5 problems
            print(f"  - {entry['id']}: {entry['issue']}")
        if len(problematic_entries) > 5:
            print(f"  ... and {len(problematic_entries) - 5} more")
    
    # Show the first 5 rows to understand the format
    print(f"\nFirst 5 valid entries:")
    print(df[['ID', 'Sequence_Length']].head().to_string())
    
    return df, stats, problematic_entries

# 3. Analyze amino acid composition
def analyze_aa_composition(seq_series, title):
    """Analyzes and plots the amino acid frequency distribution."""
    from collections import Counter
    import numpy as np
    
    all_sequences = ''.join(seq_series.tolist())
    aa_count = Counter(all_sequences)
    total_aas = len(all_sequences)
    
    # Calculate frequencies
    aa_freq = {aa: count/total_aas for aa, count in aa_count.items()}
    
    # Plot
    plt.figure(figsize=(12, 6))
    aa_names = list(aa_freq.keys())
    frequencies = list(aa_freq.values())
    
    plt.bar(aa_names, frequencies)
    plt.title(f'Amino Acid Frequency Distribution - {title}')
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print summary of non-standard AAs
    standard_aas = set('ACDEFGHIKLMNPQRSTVWY')
    found_aas = set(aa_count.keys())
    non_standard = found_aas - standard_aas
    
    if non_standard:
        print(f"⚠️ Non-standard amino acids found: {non_standard}")
        for ns_aa in non_standard:
            ns_count = aa_count[ns_aa]
            print(f"  {ns_aa}: {ns_count} occurrences ({ns_count/total_aas*100:.4f}%)")
    else:
        print("✅ No non-standard amino acids found.")
    
    return aa_freq

# 4. Main analysis execution
try:
    # Analyze each dataset individually
    natural_df, natural_stats, natural_problems = analyze_fasta_file(natural_fasta_path, "Natural AMPs")
    gram_neg_df, gram_neg_stats, gram_neg_problems = analyze_fasta_file(gram_neg_fasta_path, "Anti-Gram-negative AMPs")
    
    if len(natural_df) > 0 and len(gram_neg_df) > 0:
        # Analyze potential overlap between datasets
        print(f"\n\033[1m=== Dataset Overlap Analysis ===\033[0m")
        natural_seqs = set(natural_df['Sequence'])
        gram_neg_seqs = set(gram_neg_df['Sequence'])

        overlap_seqs = natural_seqs & gram_neg_seqs
        print(f"✅ Sequences in both Natural and Anti-Gram-negative datasets: {len(overlap_seqs)}")
        print(f"📊 Natural sequences not in Anti-Gram-negative: {len(natural_seqs - gram_neg_seqs)}")
        print(f"📊 Anti-Gram-negative sequences not in Natural: {len(gram_neg_seqs - natural_seqs)}")

        # Analyze amino acid composition for each dataset
        print(f"\n\033[1m=== Amino Acid Analysis ===\033[0m")
        natural_aa_freq = analyze_aa_composition(natural_df['Sequence'], "Natural AMPs")
        gram_neg_aa_freq = analyze_aa_composition(gram_neg_df['Sequence'], "Anti-Gram-negative AMPs")

        # Analyze length distributions
        print(f"\n\033[1m=== Length Distribution Analysis ===\033[0m")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(natural_df['Sequence_Length'], bins=50, edgecolor='black', alpha=0.7, label='Natural')
        plt.title('Natural AMPs Length Distribution')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(gram_neg_df['Sequence_Length'], bins=50, edgecolor='black', alpha=0.7, label='Gram-negative', color='orange')
        plt.title('Anti-Gram-negative AMPs Length Distribution')
        plt.xlabel('Sequence Length')
        plt.tight_layout()
        plt.show()

        # Show detailed info about the longest sequences
        print(f"\n\033[1m=== Longest Sequences Analysis ===\033[0m")
        print("Top 5 longest natural AMPs:")
        print(natural_df.nlargest(5, 'Sequence_Length')[['ID', 'Sequence_Length']].to_string())
        print("\nTop 5 longest anti-Gram-negative AMPs:")
        print(gram_neg_df.nlargest(5, 'Sequence_Length')[['ID', 'Sequence_Length']].to_string())

except Exception as e:
    print(f"❌ Unexpected error during analysis: {str(e)}")
    print("This might be due to file encoding issues or corrupted downloads.")
    print("Try re-downloading the files or checking their integrity.")

print(f"\n\033[1m=== ANALYSIS COMPLETE ===\033[0m")