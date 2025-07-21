import os
import sys
import argparse
import yaml
from disambiguation import Disambiguator
from launch_utils import get_incomplete_experiments

def disambiguate_experiment_directory(exp_dir, cache_path, max_cache_size=10000):
    """
    Disambiguate experiment results in an experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        cache_path: Path to the cache file
        max_cache_size: Maximum number of entries to keep in the cache
        
    Returns:
        bool: True if disambiguation completed successfully
    """
    # Check if directory exists
    if not os.path.isdir(exp_dir):
        print(f"Error: Experiment directory {exp_dir} does not exist")
        return False
    
    # Check for predictions.csv file
    predictions_file = os.path.join(exp_dir, "predictions.csv")
    if not os.path.isfile(predictions_file):
        print(f"Error: Predictions file not found in {exp_dir}")
        return False
    
    # Check if results.json already exists (renamed from evaluation.json)
    results_file = os.path.join(exp_dir, "results.json")
    if os.path.isfile(results_file):
        print(f"Skipping {os.path.basename(exp_dir)} - already disambiguated")
        return True
    
    # Load config.yaml to get dataset name
    config_file = os.path.join(exp_dir, "config.yaml")
    if not os.path.isfile(config_file):
        print(f"Error: Config file not found in {exp_dir}")
        return False
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_name = config["dataset"]["name"]
    split = config["dataset"].get("split", "val")
    
    # Get ground truth path
    ground_truth = f"data/{dataset_name}/data/{split}.csv"
    if not os.path.isfile(ground_truth):
        print(f"Error: Ground truth file {ground_truth} does not exist")
        return False
    
    # Set up disambiguator
    disambiguator = Disambiguator(cache_path=cache_path, max_cache_size=max_cache_size)
    
    print(f"Running disambiguation for experiment: {os.path.basename(exp_dir)}")
    print(f"Predictions file: {predictions_file}")
    print(f"Ground truth: {ground_truth}")
    
    # Run disambiguation
    disambiguator.disambiguate_csv(predictions_file)
    
    # Run evaluation
    print(f"Running evaluation...")
    disambiguator.run_evaluation(predictions_file, ground_truth)
    
    print(f"\nDisambiguation completed for {os.path.basename(exp_dir)}")
    return True

def disambiguate_all_experiments(base_dir, cache_path, max_cache_size=10000, only_incomplete=True):
    """
    Disambiguate all experiments in the base directory, including subfolders.
    """
    # Check if base directory exists
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return 0, 0, 0

    # Recursively find all subdirectories containing predictions.csv
    exp_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "predictions.csv" in files:
            exp_dirs.append(root)

    if not exp_dirs:
        print(f"No experiment directories with predictions.csv found in {base_dir}")
        return 0, 0, 0

    print(f"Found {len(exp_dirs)} experiment directories with predictions.csv in {base_dir}")

    succeeded = 0
    failed = 0
    skipped = 0

    for exp_dir in sorted(exp_dirs):
        # Only print and process if not already disambiguated
        results_file = os.path.join(exp_dir, "results.json")
        if os.path.isfile(results_file):
            skipped += 1
            continue
        print(f"\n{'=' * 80}\nProcessing experiment {succeeded + failed + skipped + 1}/{len(exp_dirs)}\n{'=' * 80}")
        if disambiguate_experiment_directory(exp_dir, cache_path, max_cache_size):
            succeeded += 1
        else:
            failed += 1

    return succeeded, failed, skipped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disambiguate experiment results")
    
    parser.add_argument("--experiment", type=str, help="Path to specific experiment directory to disambiguate")
    parser.add_argument("--dataset", type=str, default="dataset2024", 
                       help="Dataset name (for disambiguating all experiments for a dataset)")
    parser.add_argument("--cache", type=str, default="/Users/edouardalbert-roulhac/Documents/Montreal/recherche/code/soft_knowledge_retrieval/predictions/cache/wikidata_cache.json",
                       help="Path to the cache file")
    parser.add_argument("--max-cache-size", type=int, default=10000,
                       help="Maximum number of entries to keep in the cache")
    parser.add_argument("--all", action="store_true",
                       help="Process all experiments, not just incomplete ones")
    
    args = parser.parse_args()
    
    if args.experiment:
        # Disambiguate a specific experiment
        if disambiguate_experiment_directory(args.experiment, args.cache, args.max_cache_size):
            print("\nDisambiguation completed successfully!")
        else:
            print("\nDisambiguation failed.")
            sys.exit(1)
    else:
        # Disambiguate experiments for a dataset
        base_dir = f"predictions/{args.dataset}"
        succeeded, failed, skipped = disambiguate_all_experiments(
            base_dir, 
            args.cache, 
            args.max_cache_size,
            not args.all  # Process only incomplete experiments unless --all is specified
        )
        print(f"\nDisambiguation summary: {succeeded} succeeded, {failed} failed, {skipped} skipped")
        if failed > 0:
            sys.exit(1)
