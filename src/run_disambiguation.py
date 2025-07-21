import os
import argparse
from disambiguation import Disambiguator

def run_disambiguation(input_file, ground_truth, cache_path, max_cache_size=10000):
    """
    Run disambiguation on a CSV file of predictions.
    
    Args:
        input_file: Path to the predictions CSV file
        ground_truth: Path to the ground truth file for evaluation
        cache_path: Path to the cache file
        max_cache_size: Maximum number of entries to keep in the cache
    """
    # Set up disambiguator
    disambiguator = Disambiguator(cache_path=cache_path, max_cache_size=max_cache_size)
    
    print(f"Running disambiguation for {os.path.basename(input_file)}...")
    
    # Run disambiguation
    disambiguator.disambiguate_csv(input_file)
    
    # Run evaluation
    print(f"Running evaluation against {os.path.basename(ground_truth)}...")
    disambiguator.run_evaluation(input_file, ground_truth)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disambiguate entity references in a predictions CSV")
    
    parser.add_argument("input_file", help="Path or filename of the predictions CSV file")
    parser.add_argument("--dataset", default="dataset2024", help="Dataset name")
    parser.add_argument("--split", default="val", help="Data split to use for ground truth")
    parser.add_argument("--cache", default="predictions/cache/wikidata_cache.json", 
                       help="Path to the cache file")
    parser.add_argument("--max_cache_size", type=int, default=10000,
                       help="Maximum number of entries to keep in the cache")
    
    args = parser.parse_args()
    
    # Ensure input_file is a full path if only a filename was provided
    if not os.path.isabs(args.input_file):
        input_file = os.path.join(f"predictions/{args.dataset}", args.input_file)
    else:
        input_file = args.input_file
    
    # Construct ground truth path based on dataset and split
    ground_truth = f"data/{args.dataset}/data/{args.split}.csv"
    
    # Verify files exist
    if not os.path.isfile(input_file):
        print(f"Error: Input file {input_file} does not exist")
        exit(1)
        
    if not os.path.isfile(ground_truth):
        print(f"Error: Ground truth file {ground_truth} does not exist")
        exit(1)
    
    print(f"Using ground truth from {ground_truth}")
    
    success = run_disambiguation(
        input_file,
        ground_truth,
        args.cache,
        args.max_cache_size
    )
    
    if success:
        print("\nDisambiguation completed successfully!")
    else:
        print("\nDisambiguation failed.")
