import requests
import csv
import time
from datetime import datetime

def safe_get(data, *keys, default=''):
    """Safely get nested dictionary values with error handling."""
    try:
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, default)
            elif isinstance(data, list) and len(data) > 0:
                data = data[0].get(key, default) if isinstance(data[0], dict) else default
            else:
                return default
        return data if data is not None else default
    except (AttributeError, IndexError, TypeError):
        return default

def fetch_peptide_data(peptide_id):
    """Fetch peptide data from DBAASP API with error handling."""
    url = f"https://dbaasp.org/peptides/DBAASPS_{peptide_id}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ID {peptide_id}: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON for ID {peptide_id}: {e}")
        return None

def extract_toxicity_data(peptide_id, data):
    """Extract toxicity-relevant data from API response with robust error handling."""
    if not data:
        return None
        
    try:
        # Extract sequence
        sequence = safe_get(data, 'sequence', default='')
        
        # Extract hemolytic/cytotoxic activities (first entry if multiple)
        hemolytic_data = safe_get(data, 'hemoliticCytotoxicActivities', default=[])
        if hemolytic_data and isinstance(hemolytic_data, list) and len(hemolytic_data) > 0:
            first_hemolytic = hemolytic_data[0]
            hemolytic_info = {
                'hemolytic_concentration': safe_get(first_hemolytic, 'concentration'),
                'hemolytic_unit': safe_get(first_hemolytic, 'unit', 'name'),
                'hemolytic_activity': safe_get(first_hemolytic, 'activity'),
                'hemolytic_target_cell': safe_get(first_hemolytic, 'targetCell', 'name')
            }
        else:
            hemolytic_info = {
                'hemolytic_concentration': '',
                'hemolytic_unit': '',
                'hemolytic_activity': '',
                'hemolytic_target_cell': ''
            }
        
        # Extract target activities (first entry if multiple)
        target_data = safe_get(data, 'targetActivities', default=[])
        if target_data and isinstance(target_data, list) and len(target_data) > 0:
            first_target = target_data[0]
            target_info = {
                'target_species': safe_get(first_target, 'targetSpecies', 'name'),
                'target_concentration': safe_get(first_target, 'concentration'),
                'target_unit': safe_get(first_target, 'unit', 'name'),
                'target_activity': safe_get(first_target, 'activity')
            }
        else:
            target_info = {
                'target_species': '',
                'target_concentration': '',
                'target_unit': '',
                'target_activity': ''
            }
        
        return {
            'id': peptide_id,
            'sequence': sequence,
            **hemolytic_info,
            **target_info
        }
        
    except Exception as e:
        print(f"Error processing data for ID {peptide_id}: {e}")
        return None

def main(input_csv, output_csv):
    """Main processing function."""
    # Read input CSV
    try:
        with open(input_csv, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            peptide_ids = [row['ID'] for row in reader if row.get('ID')]
            # peptide_ids = [491]
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    if not peptide_ids:
        print("No peptide IDs found in the input file.")
        return
    
    # Prepare output CSV
    fieldnames = [
        'id', 'sequence',
        'hemolytic_concentration', 'hemolytic_unit', 'hemolytic_activity', 'hemolytic_target_cell',
        'target_species', 'target_concentration', 'target_unit', 'target_activity'
    ]
    
    successful_count = 0
    error_count = 0
    
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process each peptide ID
            for count, peptide_id in enumerate(peptide_ids, 1):
                print(f"Processing ID {peptide_id} ({count}/{len(peptide_ids)})...")
                
                # Fetch data with rate limiting
                data = fetch_peptide_data(peptide_id)
                if data is None:
                    error_count += 1
                    continue
                    
                # Extract relevant data
                result = extract_toxicity_data(peptide_id, data)
                if result:
                    writer.writerow(result)
                    successful_count += 1
                else:
                    error_count += 1
                
                # Delay between requests (5 seconds)
                if count < len(peptide_ids):
                    time.sleep(5)
    
    except Exception as e:
        print(f"Error writing output file: {e}")
        return
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {successful_count} peptides")
    print(f"Errors encountered: {error_count} peptides")
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    input_filename = "data/peptides.csv"  # Change to your input file name
    output_filename = f"toxicity_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    main(input_filename, output_filename)