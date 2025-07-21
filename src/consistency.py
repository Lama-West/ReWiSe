import pandas as pd
import ast
import re
from collections import Counter
from typing import List, Dict, Any, Union
import json
import os
from pathlib import Path
import tempfile
import sys

from evaluate import evaluate


relation_categories = {
    "countryLandBordersCountry": "str_list",
    "personHasCityOfDeath": "none_or_single_str",
    "seriesHasNumberOfEpisodes": "numerical",
    "awardWonBy": "str_list",
    "companyTradesAtStockExchange": "str_list",
    "hasCapacity": "numerical",
    "hasArea": "numerical",
    }

# Optimal strategies per relation (from consistency threshold optimization analysis)
optimal_strategy = {
    "awardWonBy": 0.05,
    "companyTradesAtStockExchange": 0.3,
    "countryLandBordersCountry": 0.5,
    "hasCapacity": "median",
    "hasArea": "median", 
    "seriesHasNumberOfEpisodes": "majority",
    "personHasCityOfDeath": "majority",
}

def majority_vote(predictions: List[str]) -> str:
    """
    Simple majority voting for list of string predictions.
    
    Args:
        predictions: List of prediction strings
        
    Returns:
        Most common prediction, or first one if tie
    """
    if not predictions:
        return ""
        
    counter = Counter(predictions)
    # Return most common (first in case of tie)
    return counter.most_common(1)[0][0]


def threshold_aggregation(predictions: List[str], threshold: float = 0.5) -> List[str]:
    """
    Keep answers that appear in at least threshold fraction of predictions.
    
    Args:
        predictions: List of prediction strings  
        threshold: Minimum fraction to keep an answer (0.0 to 1.0)
        
    Returns:
        List of answers that meet the threshold
    """
    if not predictions:
        return []
    
    counter = Counter(predictions)
    total = len(predictions)
    
    result = []
    for answer, count in counter.items():
        if count / total >= threshold:
            result.append(answer)
    
    return result


def parse_predictions(predictions: List[str]) -> List[List[str]]:
    parsed_predictions = []
    for pred in predictions:
        pred = str(pred).strip()
        if pred.lower() in ['none', '[]']:
            parsed_predictions.append([])
            continue

        try:
            entities = ast.literal_eval(pred)
            if isinstance(entities, list):
                valid_entities = []
                for e in entities:
                    e_str = str(e).strip()
                    if e_str.lower() == 'none':
                        valid_entities.append(None)
                    else:
                        valid_entities.append(e_str)
                parsed_predictions.append(valid_entities)
            else:
                entity_str = str(entities).strip()
                if entity_str.lower() == 'none':
                    parsed_predictions.append([None])
                else:
                    parsed_predictions.append([entity_str])
        except (ValueError, SyntaxError):
            if pred.lower() == 'none' or pred.lower() == '[]':
                parsed_predictions.append([])
            else:
                entities = []
                for e in pred.split(","):
                    e = e.strip()
                    if e.lower() == 'none':
                        entities.append(None)
                    else:
                        entities.append(e)
                parsed_predictions.append(entities)
    return parsed_predictions

def aggregate_single_value(parsed_predictions: List[List[str]]) -> str:
    single_answers = []
    for pred in parsed_predictions:
        if not pred:
            single_answers.append(None)
        else:
            single_answers.append(pred[0])
    
    counter = Counter(single_answers)
    most_common = counter.most_common(1)[0][0]
    return "[]" if most_common is None else str([most_common])

def aggregate_numerical(parsed_predictions: List[List[str]], strategy: str = "median") -> str:
    # Extract numerical values from all predictions
    numerical_values = []
    for pred in parsed_predictions:
        if not pred:
            continue
        for item in pred:
            if item is None:
                continue
            try:
                # Try to cast as float first, then int
                if '.' in str(item):
                    numerical_values.append(float(item))
                else:
                    numerical_values.append(int(item))
            except (ValueError, TypeError):
                # Skip values that can't be cast to numbers
                continue
    
    if not numerical_values:
        return "[]"
    
    if strategy == "mean":
        result = sum(numerical_values) / len(numerical_values)
        # Convert back to int if all input values were integers
        if all(isinstance(x, int) for x in numerical_values):
            result = int(round(result))
        return str([str(result)])
    elif strategy == "median":
        numerical_values.sort()
        n = len(numerical_values)
        if n % 2 == 0:
            result = (numerical_values[n//2 - 1] + numerical_values[n//2]) / 2
        else:
            result = numerical_values[n//2]
        # Convert back to int if all input values were integers
        if all(isinstance(x, int) for x in numerical_values):
            result = int(round(result))
        return str([str(result)])
    else:  # majority (default)
        counter = Counter(numerical_values)
        most_common = counter.most_common(1)[0][0]
        return str([str(most_common)])

def aggregate_str_list(parsed_predictions: List[List[str]], strategy: str, threshold: float) -> str:
    all_entities = []
    none_count = 0
    
    for pred in parsed_predictions:
        if not pred:
            none_count += 1
        else:
            valid_entities = [e for e in pred if e is not None]
            all_entities.extend(valid_entities)
    
    if strategy == "majority":
        if none_count > len(parsed_predictions) / 2:
            return "[]"
        elif all_entities:
            most_common = Counter(all_entities).most_common(1)[0][0]
            return str([most_common])
        else:
            return "[]"
    
    elif strategy == "threshold":
        if not all_entities:
            return "[]"
        
        counter = Counter(all_entities)
        total_predictions = len(parsed_predictions)
        kept_entities = []
        
        for entity, count in counter.items():
            if count / total_predictions >= threshold:
                kept_entities.append(entity)
        
        return str(sorted(kept_entities)) if kept_entities else "[]"

def entity_list_aggregation(predictions: List[str], strategy: str = "relation-threshold", 
                          threshold: float = 0.5, relation_type: str = "str_list", 
                          relation: str = None) -> str:
    if not predictions:
        return "[]"
    
    parsed_predictions = parse_predictions(predictions)
    if not parsed_predictions:
        return "[]"
    
    # For relation-threshold strategy, use optimal strategies per relation
    if strategy == "relation-threshold" and relation is not None:
        optimal_value = optimal_strategy.get(relation, 0.5)
        if isinstance(optimal_value, float):
            # It's a threshold value
            threshold = optimal_value
            actual_strategy = "threshold"
        else:
            # It's a strategy name (like "median", "mean", "majority")
            actual_strategy = optimal_value
    else:
        actual_strategy = strategy
        
    if relation_type != "numerical" and actual_strategy in ["mean", "median"]:
        actual_strategy = "threshold"

    if relation_type == "none_or_single_str":
        return aggregate_single_value(parsed_predictions)
    elif relation_type == "numerical":
        return aggregate_numerical(parsed_predictions, actual_strategy)
    elif relation_type == "str_list":
        return aggregate_str_list(parsed_predictions, actual_strategy, threshold)
    else:
        return aggregate_str_list(parsed_predictions, actual_strategy, threshold)


def aggregate_consistency_predictions(
        predictions_file: str,
        strategy: str = "majority",
        threshold: float = 0.5,
        partial_consistency: int = None,
        save_to_file: bool = False
        ) -> pd.DataFrame:
    """
    Aggregate consistency predictions from a predictions CSV file.
    
    Args:
        predictions_file: Path to CSV file with consistency predictions
        strategy: Aggregation strategy ("majority" or "threshold")
        threshold: For threshold strategy, minimum fraction to keep
        partial_consistency: If specified, use only first N predictions from each example
        save_to_file: If True, save aggregated predictions back to the original file
        
    Returns:
        DataFrame with aggregated predictions
    """
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Check for existing aggregated columns to determine which column to use
    if "object_entities_list" in df.columns:
        prediction_column = "object_entities_list"
    elif "generated_list" not in df.columns:
        raise ValueError("Expected 'generated_list' or 'object_entities_list' column in predictions file")
    else:
        prediction_column = "generated_list"
    
    aggregated_results = []
    
    for _, row in df.iterrows():
        try:
            if prediction_column == "object_entities_list":
                # Use already-parsed entity list
                prediction_list = ast.literal_eval(row[prediction_column])
            else:
                # Parse from generated_list
                prediction_list = ast.literal_eval(row[prediction_column])
        except (ValueError, SyntaxError):
            # If it's already a single prediction, wrap in list
            if prediction_column == "object_entities_list":
                prediction_list = [row[prediction_column]]
            else:
                prediction_list = [row[prediction_column]]
        
        if partial_consistency is not None and partial_consistency > 0:
            prediction_list = prediction_list[:partial_consistency]
        
        relation = row.get("Relation")
        relation_type = relation_categories.get(relation)
        
        aggregated = entity_list_aggregation(prediction_list, strategy=strategy, threshold=threshold, 
                                           relation_type=relation_type, relation=relation)
        
        # Ensure aggregated result is always a string
        if not isinstance(aggregated, str):
            aggregated = str(aggregated)
        
        # Create aggregated row
        agg_row = row.copy()
        agg_row["ObjectEntities"] = aggregated
        agg_row["ObjectEntitiesID"] = aggregated  # In this dataset, IDs and entities are the same
        agg_row["aggregation_strategy"] = strategy
        agg_row["relation_type"] = relation_type
        if strategy in ["threshold", "relation-threshold"]:
            if strategy == "relation-threshold" and relation is not None:
                optimal_value = optimal_strategy.get(relation, 0.5)
                if isinstance(optimal_value, float):
                    agg_row["threshold"] = optimal_value
                else:
                    agg_row["strategy"] = optimal_value
            else:
                agg_row["threshold"] = threshold
        if partial_consistency is not None:
            agg_row["partial_consistency"] = partial_consistency
        agg_row["n_predictions"] = len(prediction_list)
        
        aggregated_results.append(agg_row)
    
    result_df = pd.DataFrame(aggregated_results)
    
    if save_to_file:
        # Update the generated column to match ObjectEntities format
        for i, row in result_df.iterrows():
            obj_entities = row["ObjectEntities"]
            try:
                # Ensure obj_entities is a string
                if not isinstance(obj_entities, str):
                    obj_entities = str(obj_entities)
                
                entities_list = ast.literal_eval(obj_entities) if obj_entities != "[]" else []
                if entities_list:
                    result_df.at[i, "generated"] = ", ".join(str(e) for e in entities_list)
                else:
                    # Empty list means None
                    result_df.at[i, "generated"] = None
            except (ValueError, SyntaxError, TypeError):
                # If ObjectEntities is malformed, return None
                result_df.at[i, "generated"] = None
                pass
        
        # Save the aggregated predictions back to the original file
        output_path = predictions_file
        print(f"Saving aggregated predictions to: {output_path}")
        result_df.to_csv(output_path, index=False)
        
        # Delete existing results files since predictions have changed
        predictions_dir = Path(predictions_file).parent
        results_string_path = predictions_dir / "results_string.json"
        results_path = predictions_dir / "results.json"
        
        if results_string_path.exists():
            os.unlink(results_string_path)
            print(f"Deleted existing results_string.json")
        if results_path.exists():
            os.unlink(results_path)
            print(f"Deleted existing results.json")
    
    return result_df

def evaluate_with_ground_truth(
        aggregated_df: pd.DataFrame, 
        dataset: str,
        predictions_file: str,
        strategy: str = "threshold",
        threshold: float = 0.5,
        evaluate_all_levels: bool = False,
        detailed_relation: bool = False,
        save_results_to_file: bool = False
        ) -> Dict[str, Any]:
    """
    Evaluate aggregated predictions against ground truth using evaluate.py.
    
    Args:
        aggregated_df: DataFrame with aggregated predictions
        dataset: Dataset name for finding ground truth
        predictions_file: Path to original predictions file
        evaluate_all_levels: If True, evaluate for all levels 1 to max_consistency
        
    Returns:
        Dictionary with evaluation results
    """
    ground_truth_path = f"data/{dataset}/data/val.csv"
    
    results = {}
    
    if evaluate_all_levels:
        # Infer max consistency from the data
        original_df = pd.read_csv(predictions_file)
        max_consistency = 0
        for _, row in original_df.iterrows():
            try:
                prediction_list = ast.literal_eval(row["generated_list"])
                max_consistency = max(max_consistency, len(prediction_list))
            except (ValueError, SyntaxError):
                max_consistency = max(max_consistency, 1)
                
        # Evaluate for each consistency level from 1 to max_consistency
        for n in range(1, max_consistency + 1):
            
            # Create aggregated predictions for this level
            level_df = aggregate_consistency_predictions(
                predictions_file,
                strategy=strategy,
                threshold=threshold,
                partial_consistency=n
            )
            
            # Save to temporary file for evaluation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                # Ensure ObjectEntities column is always string type when saving
                level_df_copy = level_df.copy()
                level_df_copy['ObjectEntities'] = level_df_copy['ObjectEntities'].astype(str)
                level_df_copy.to_csv(tmp_file.name, index=False)
                
                eval_results = evaluate(
                    predictions=tmp_file.name,
                    ground_truth=ground_truth_path,
                    mode="string",
                    verbose=False
                )
                
                # Extract macro F1 for all relations or detailed relation scores
                if detailed_relation:
                    # Get F1 scores for each relation in alphabetical order
                    relation_f1_scores = {}
                    relations = [rel for rel in eval_results.index if rel != "*** All Relations ***"]
                    for relation in sorted(relations):
                        relation_f1_scores[relation] = eval_results.loc[relation, "macro-f1"]
                    
                    results[f"consistency_{n}"] = {
                        "relation_f1_scores": relation_f1_scores,
                        "full_results": eval_results
                    }
                else:
                    macro_f1 = eval_results.loc["*** All Relations ***", "macro-f1"]
                    results[f"consistency_{n}"] = {
                        "macro_f1": macro_f1,
                        "full_results": eval_results
                    }
                os.unlink(tmp_file.name)
    else:
        # Single evaluation with current aggregation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            # Ensure ObjectEntities column is always string type when saving
            aggregated_df_copy = aggregated_df.copy()
            aggregated_df_copy['ObjectEntities'] = aggregated_df_copy['ObjectEntities'].astype(str)
            aggregated_df_copy.to_csv(tmp_file.name, index=False)
            
            eval_results = evaluate(
                predictions=tmp_file.name,
                ground_truth=ground_truth_path,
                mode="string",
                verbose=False
            )
            
            # Extract macro F1 for all relations or detailed relation scores
            if detailed_relation:
                # Get F1 scores for each relation in alphabetical order
                relation_f1_scores = {}
                relations = [rel for rel in eval_results.index if rel != "*** All Relations ***"]
                for relation in sorted(relations):
                    relation_f1_scores[relation] = eval_results.loc[relation, "macro-f1"]
                
                results["current_aggregation"] = {
                    "relation_f1_scores": relation_f1_scores,
                    "full_results": eval_results
                }
            else:
                macro_f1 = eval_results.loc["*** All Relations ***", "macro-f1"]
                results["current_aggregation"] = {
                    "macro_f1": macro_f1,
                    "full_results": eval_results
                }
            
            # If save_results_to_file is True, save the evaluation results
            if save_results_to_file:
                predictions_dir = Path(predictions_file).parent
                results_string_path = predictions_dir / "results_string.json"
                
                # Convert eval_results to string format for saving
                results_dict = {}
                for relation in eval_results.index:
                    results_dict[relation] = {
                        "macro-p": float(eval_results.loc[relation, "macro-p"]),
                        "macro-r": float(eval_results.loc[relation, "macro-r"]), 
                        "macro-f1": float(eval_results.loc[relation, "macro-f1"]),
                        "micro-p": float(eval_results.loc[relation, "micro-p"]),
                        "micro-r": float(eval_results.loc[relation, "micro-r"]),
                        "micro-f1": float(eval_results.loc[relation, "micro-f1"]),
                        "avg. #preds": float(eval_results.loc[relation, "avg. #preds"]),
                        "#empty preds": int(eval_results.loc[relation, "#empty preds"])
                    }
                
                with open(results_string_path, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                print(f"Saved evaluation results to: {results_string_path}")
            
            os.unlink(tmp_file.name)
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate consistency predictions")
    parser.add_argument("predictions_file", help="Path to predictions CSV file")
    parser.add_argument("--dataset", default="dataset2024", help="Dataset")
    parser.add_argument("--strategy", choices=["majority", "threshold", "relation-threshold"], 
                       default="relation-threshold",
                       help="Aggregation strategy. 'relation-threshold' uses optimal thresholds per relation.")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for threshold strategy")
    parser.add_argument("--partial-consistency", type=int, default=None,
                       help="Use only first N predictions from each example")
    parser.add_argument("--save-to-file", action="store_true",
                       help="Save aggregated predictions back to the original file")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate aggregated predictions against ground truth")
    parser.add_argument("--evaluate-all-levels", action="store_true",
                       help="Evaluate all consistency levels from 1 to max_consistency")
    parser.add_argument("--detailed-relation", action="store_true",
                       help="Show detailed F1 scores for each relation instead of overall macro F1")

    args = parser.parse_args()
    
    # Auto-detect dataset from predictions file path if not specified
    if args.dataset == "dataset2024" and "predictions/" in args.predictions_file:
        path_parts = args.predictions_file.split("predictions/")
        if len(path_parts) > 1:
            dataset_part = path_parts[1].split("/")[0]
            if dataset_part.startswith("dataset"):
                args.dataset = dataset_part
                print(f"Auto-detected dataset: {args.dataset}")
    
    # Run aggregation
    result_df = aggregate_consistency_predictions(
        args.predictions_file,
        args.strategy,
        args.threshold,
        args.partial_consistency,
        args.save_to_file
    )
        
    # Run evaluation if requested
    if args.evaluate or args.evaluate_all_levels:
        try:
            eval_results = evaluate_with_ground_truth(
                result_df,
                args.dataset,
                args.predictions_file,
                args.strategy,
                args.threshold,
                args.evaluate_all_levels,
                args.detailed_relation,
                save_results_to_file=args.save_to_file  # Save results if --save-to-file was used
            )
            
            if args.evaluate_all_levels:
                if args.detailed_relation:
                    # Create a structured table for detailed relation F1 scores
                    first_level = list(eval_results.keys())[0]
                    relations = sorted(eval_results[first_level]["relation_f1_scores"].keys())
                    
                    # Build table data
                    table_data = []
                    for level, results in eval_results.items():
                        level_num = int(level.replace("consistency_", ""))
                        row = {"consistency_level": level_num}
                        for relation in relations:
                            row[relation] = results["relation_f1_scores"][relation]
                        table_data.append(row)
                    
                    # Convert to DataFrame and return
                    results_df = pd.DataFrame(table_data)
                    results_df = results_df.sort_values("consistency_level")
                    
                    # Return the DataFrame for programmatic use
                    return results_df
                else:
                    print("\nMacro F1 scores by consistency level:")
                    for level, results in eval_results.items():
                        print(f"{level}: {results['macro_f1']:.3f}")
            else:
                if args.detailed_relation:
                    # Create a single-row table for current aggregation
                    relation_scores = eval_results["current_aggregation"]["relation_f1_scores"]
                    row_data = {"consistency_level": args.partial_consistency or "all"}
                    row_data.update(relation_scores)
                    
                    results_df = pd.DataFrame([row_data])
                    
                    # Return the DataFrame for programmatic use
                    return results_df
                else:
                    print(f"\nMacro F1 score: {eval_results['current_aggregation']['macro_f1']:.3f}")
                
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Make sure ground truth exists at: data/{args.dataset}/data/val.csv")

if __name__ == "__main__":
    main()
