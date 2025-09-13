import requests
import pandas as pd
import time
import logging
import argparse
from typing import Dict, List, Optional
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dbaasp_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fetch_peptide_data(peptide_id: int, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch peptide data from DBAASP API with retry logic
    """
    url = f"https://dbaasp.org/peptides/DBAASPS_{peptide_id}"
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1} for ID {peptide_id}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for ID {peptide_id}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed for ID {peptide_id}")
                return None

def extract_toxicity_data(peptide_data: Dict) -> Dict:
    """
    Extract relevant toxicity data from the API response
    """
    toxicity_info = {}
    
    # Extract basic information
    toxicity_info['id'] = peptide_data.get('id', '')
    toxicity_info['dbaaspId'] = peptide_data.get('dbaaspId', '')
    toxicity_info['name'] = peptide_data.get('name', '')
    toxicity_info['sequence'] = peptide_data.get('sequence', '')
    toxicity_info['sequenceLength'] = peptide_data.get('sequenceLength', '')
    
    # Extract hemolytic/cytotoxic activities
    hemolytic_activities = peptide_data.get('hemoliticCytotoxicActivities', [])
    
    # Initialize toxicity columns
    toxicity_info['target_cell'] = ''
    toxicity_info['activity_measure'] = ''
    toxicity_info['concentration'] = ''
    toxicity_info['unit'] = ''
    toxicity_info['activity_value'] = ''
    
    # Extract the first hemolytic/cytotoxic activity if available
    if hemolytic_activities:
        activity = hemolytic_activities[0]
        toxicity_info['target_cell'] = activity.get('targetCell', {}).get('name', '')
        toxicity_info['activity_measure'] = activity.get('activityMeasureForLysisValue', '')
        toxicity_info['concentration'] = activity.get('concentration', '')
        toxicity_info['unit'] = activity.get('unit', {}).get('name', '')
        toxicity_info['activity_value'] = activity.get('activity', '')
    
    return toxicity_info

def process_peptide_ids(input_csv_path: str, output_csv_path: str, start_index: int = 0):
    """
    Process peptide IDs with resume capability
    """
    # Read input CSV
    try:
        input_df = pd.read_csv(input_csv_path)
        peptide_ids = input_df['ID'].astype(int).tolist()
        logger.info(f"Loaded {len(peptide_ids)} peptide IDs from {input_csv_path}")
    except Exception as e:
        logger.error(f"Error reading input CSV: {e}")
        return
    
    # Try to load existing output to resume
    try:
        existing_df = pd.read_csv(output_csv_path)
        processed_ids = set(existing_df['id'].astype(int).tolist())
        logger.info(f"Resuming from existing file with {len(processed_ids)} processed peptides")
    except FileNotFoundError:
        processed_ids = set()
        existing_df = pd.DataFrame()
        logger.info("Starting new output file")
    
    results = []
    processed_count = 0
    failed_count = 0
    
    for idx, peptide_id in enumerate(peptide_ids[start_index:], start=start_index):
        if peptide_id in processed_ids:
            logger.debug(f"Skipping already processed ID {peptide_id}")
            continue
            
        logger.info(f"Processing ID {peptide_id} ({idx + 1}/{len(peptide_ids)})")
        
        peptide_data = fetch_peptide_data(peptide_id)
        
        if peptide_data:
            toxicity_data = extract_toxicity_data(peptide_data)
            results.append(toxicity_data)
            processed_count += 1
            
            # Save incrementally every 10 peptides
            if len(results) >= 10:
                incremental_df = pd.DataFrame(results)
                if not existing_df.empty:
                    final_df = pd.concat([existing_df, incremental_df], ignore_index=True)
                else:
                    final_df = incremental_df
                final_df.to_csv(output_csv_path, index=False)
                logger.info(f"Incrementally saved {len(final_df)} peptides")
                results = []  # Reset results list
        else:
            logger.error(f"Failed to fetch data for ID {peptide_id}")
            failed_count += 1
        
        # Respect the 5-second delay
        time.sleep(5)
        
        # Progress update every 100 peptides
        if (idx + 1) % 100 == 0:
            logger.info(f"Progress: {idx + 1}/{len(peptide_ids)} peptides processed")
    
    # Save any remaining results
    if results:
        incremental_df = pd.DataFrame(results)
        if not existing_df.empty:
            final_df = pd.concat([existing_df, incremental_df], ignore_index=True)
        else:
            final_df = incremental_df
        final_df.to_csv(output_csv_path, index=False)
    
    logger.info(f"Processing complete! Total: {processed_count} succeeded, {failed_count} failed")

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='DBAASP Peptide Toxicity Data Scraper')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with peptide IDs')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file for toxicity data')
    parser.add_argument('--start', '-s', type=int, default=0, help='Start index (for resuming)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting DBAASP scraper with input: {args.input}, output: {args.output}")
    process_peptide_ids(args.input, args.output, args.start)

if __name__ == "__main__":
    main()