import pandas as pd
import ast
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from consistency import aggregate_consistency_predictions, evaluate_with_ground_truth, relation_categories


def get_detailed_consistency_results(predictions_file: str = None, dataset: str = "dataset2025", 
                                   experiment_name: str = None, strategy: str = "threshold", 
                                   threshold: float = 0.5) -> pd.DataFrame:
    """
    Get detailed results by relation for all consistency levels.
    
    Args:
        predictions_file: Path to predictions CSV file (optional if experiment_name provided)
        dataset: Dataset name for finding ground truth
        experiment_name: Experiment name to auto-construct path (optional if predictions_file provided)
        strategy: Aggregation strategy ("majority", "threshold", or "relation-threshold")
        threshold: Threshold for threshold strategy
        
    Returns:
        DataFrame with consistency levels, relations, and their F1 scores
    """
    if predictions_file is None and experiment_name is not None:
        predictions_file = f"predictions/{dataset}/{experiment_name}/predictions.csv"
    elif predictions_file is None:
        raise ValueError("Either predictions_file or experiment_name must be provided")
    
    if not Path(predictions_file).exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    dummy_df = aggregate_consistency_predictions(
        predictions_file,
        strategy=strategy,
        threshold=threshold,
        partial_consistency=None,
        save_to_file=False
    )
    
    eval_results = evaluate_with_ground_truth(
        dummy_df,
        dataset,
        predictions_file,
        strategy=strategy,
        threshold=threshold,
        evaluate_all_levels=True,
        detailed_relation=True,
        save_results_to_file=False
    )
    
    rows = []
    for level, level_data in eval_results.items():
        if level.startswith("consistency_"):
            consistency_level = int(level.split("_")[1])
            relation_scores = level_data["relation_f1_scores"]
            
            for relation, f1_score in relation_scores.items():
                rows.append({
                    "consistency_level": consistency_level,
                    "relation": relation,
                    "macro_f1": f1_score,
                    "threshold": threshold
                })
    
    return pd.DataFrame(rows)


def get_threshold_range_results(predictions_file: str = None, dataset: str = "dataset2025", 
                               experiment_name: str = None, strategy: str = "threshold", 
                               threshold_range: np.ndarray = None) -> pd.DataFrame:
    """
    Get detailed results across a range of thresholds.
    
    Args:
        predictions_file: Path to predictions CSV file (optional if experiment_name provided)
        dataset: Dataset name for finding ground truth
        experiment_name: Experiment name to auto-construct path (optional if predictions_file provided)
        strategy: Aggregation strategy
        threshold_range: Array of thresholds to test (default: np.linspace(0, 1, 5))
        
    Returns:
        DataFrame with consistency levels, relations, thresholds, and their F1 scores
    """
    if threshold_range is None:
        threshold_range = np.linspace(0, 1, 5)
    
    all_results = []
    
    for threshold in threshold_range:
        try:
            df = get_detailed_consistency_results(
                predictions_file=predictions_file,
                dataset=dataset,
                experiment_name=experiment_name,
                strategy=strategy,
                threshold=threshold
            )
            all_results.append(df)
        except Exception as e:
            print(f"Error with threshold {threshold:.2f}: {e}")
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.concat(all_results, ignore_index=True)


def get_optimal_strategy_results(predictions_file: str = None, dataset: str = "dataset2025", 
                               experiment_name: str = None, n_threshold_points: int = 5) -> pd.DataFrame:
    """
    Get optimal strategy results for each relation type:
    - str_list relations: test threshold range [0, 0.25, 0.5, 0.75, 1.0] (or fewer points if n_threshold_points < 5)
    - numerical relations: test strategies ["majority", "mean", "median"]
    - none_or_single_str relations: use "majority" strategy only
    
    Args:
        predictions_file: Path to predictions CSV file (optional if experiment_name provided)
        dataset: Dataset name for finding ground truth
        experiment_name: Experiment name to auto-construct path (optional if predictions_file provided)
        n_threshold_points: Number of threshold points to test for str_list relations
        
    Returns:
        DataFrame with consistency levels, relations, strategies/thresholds, and their F1 scores
    """
    # Auto-construct predictions file path if experiment_name is provided
    if predictions_file is None and experiment_name is not None:
        predictions_file = f"predictions/{dataset}/{experiment_name}/predictions.csv"
        print(f"Using predictions file: {predictions_file}")
    elif predictions_file is None:
        raise ValueError("Either predictions_file or experiment_name must be provided")
    
    # Check if file exists
    if not Path(predictions_file).exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    all_results = []
    
    # Define strategies for each relation type
    str_list_thresholds = np.linspace(0, 1, n_threshold_points)  # Configurable number of points
    numerical_strategies = ["majority", "mean", "median"]
    
    print("Testing optimal strategies for each relation type...")
    
    # Test str_list relations with different thresholds
    print(f"\n--- Testing str_list relations with thresholds: {str_list_thresholds} ---")
    for threshold in str_list_thresholds:
        print(f"Testing threshold: {threshold:.2f}")
        try:
            df = get_detailed_consistency_results(
                predictions_file=predictions_file,
                dataset=dataset,
                experiment_name=None,  # Already have file path
                strategy="threshold",
                threshold=threshold
            )
            # Filter for str_list relations and add strategy info
            str_list_relations = [rel for rel, rel_type in relation_categories.items() if rel_type == "str_list"]
            df_filtered = df[df['relation'].isin(str_list_relations)].copy()
            df_filtered['strategy'] = f"threshold_{threshold:.2f}"
            df_filtered['strategy_type'] = "threshold"
            df_filtered['strategy_value'] = threshold
            all_results.append(df_filtered)
        except Exception as e:
            print(f"Error with threshold {threshold:.2f}: {e}")
            continue
    
    # Test numerical relations with different strategies
    print(f"\n--- Testing numerical relations with strategies: {numerical_strategies} ---")
    for strategy in numerical_strategies:
        print(f"Testing strategy: {strategy}")
        try:
            df = get_detailed_consistency_results(
                predictions_file=predictions_file,
                dataset=dataset,
                experiment_name=None,  # Already have file path
                strategy=strategy,
                threshold=0.5  # Not used for numerical relations
            )
            # Filter for numerical relations and add strategy info
            numerical_relations = [rel for rel, rel_type in relation_categories.items() if rel_type == "numerical"]
            df_filtered = df[df['relation'].isin(numerical_relations)].copy()
            df_filtered['strategy'] = strategy
            df_filtered['strategy_type'] = "numerical"
            df_filtered['strategy_value'] = strategy
            all_results.append(df_filtered)
        except Exception as e:
            print(f"Error with strategy {strategy}: {e}")
            continue
    
    # Test none_or_single_str relations with majority strategy only
    print(f"\n--- Testing none_or_single_str relations with majority strategy ---")
    try:
        df = get_detailed_consistency_results(
            predictions_file=predictions_file,
            dataset=dataset,
            experiment_name=None,  # Already have file path
            strategy="majority",
            threshold=0.5  # Not used for none_or_single_str relations
        )
        # Filter for none_or_single_str relations and add strategy info
        single_str_relations = [rel for rel, rel_type in relation_categories.items() if rel_type == "none_or_single_str"]
        df_filtered = df[df['relation'].isin(single_str_relations)].copy()
        df_filtered['strategy'] = "majority"
        df_filtered['strategy_type'] = "single_value"
        df_filtered['strategy_value'] = "majority"
        all_results.append(df_filtered)
    except Exception as e:
        print(f"Error with majority strategy for none_or_single_str: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: Optimal Strategy Results by Relation Type")
    print(f"{'='*80}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Consistency levels: {sorted(combined_df['consistency_level'].unique())}")
    print(f"Relations: {sorted(combined_df['relation'].unique())}")
    print(f"Strategies tested: {sorted(combined_df['strategy'].unique())}")
    
    # Show relation type breakdown
    for rel_type in ["str_list", "numerical", "none_or_single_str"]:
        relations = [rel for rel, rt in relation_categories.items() if rt == rel_type]
        rel_data = combined_df[combined_df['relation'].isin(relations)]
        if not rel_data.empty:
            strategies = sorted(rel_data['strategy'].unique())
            print(f"\n{rel_type} relations ({len(relations)}): {relations}")
            print(f"  Strategies tested: {strategies}")
    
    return combined_df


def analyze_optimal_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the optimal strategy results to find the best strategy for each relation.
    
    Args:
        df: DataFrame from get_optimal_strategy_results()
        
    Returns:
        DataFrame with best strategy for each relation and consistency level
    """
    if df.empty:
        return pd.DataFrame()
    
    # Find best strategy for each relation and consistency level
    best_strategies = []
    
    for relation in df['relation'].unique():
        for consistency_level in df['consistency_level'].unique():
            rel_data = df[(df['relation'] == relation) & (df['consistency_level'] == consistency_level)]
            if not rel_data.empty:
                best_row = rel_data.loc[rel_data['macro_f1'].idxmax()]
                best_strategies.append(best_row)
    
    result_df = pd.DataFrame(best_strategies)
    
    print(f"\n{'='*60}")
    print("BEST STRATEGIES BY RELATION AND CONSISTENCY LEVEL")
    print(f"{'='*60}")
    
    # Show summary by relation
    for relation in sorted(result_df['relation'].unique()):
        rel_data = result_df[result_df['relation'] == relation]
        rel_type = relation_categories.get(relation, "unknown")
        print(f"\n{relation} ({rel_type}):")
        
        if rel_type == "str_list":
            # Show best thresholds
            best_thresholds = rel_data['strategy_value'].value_counts()
            print(f"  Best thresholds: {dict(best_thresholds)}")
        elif rel_type == "numerical":
            # Show best strategies
            best_strategies_count = rel_data['strategy_value'].value_counts()
            print(f"  Best strategies: {dict(best_strategies_count)}")
        
        # Show F1 range
        f1_min, f1_max = rel_data['macro_f1'].min(), rel_data['macro_f1'].max()
        print(f"  F1 range: {f1_min:.3f} - {f1_max:.3f}")
    
    return result_df