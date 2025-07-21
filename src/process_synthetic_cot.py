import argparse
import pandas as pd
from models.processing import process_synthetic_cot_results
import os

def infer_input_path(input_arg, dataset):
    """Infer full input path from partial path or filename."""
    if os.path.exists(input_arg):
        return input_arg
    
    if '/' in input_arg:
        if input_arg.startswith('data/'):
            return input_arg
        else:
            return input_arg
    
    filename = input_arg
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    inferred_path = f"data/{dataset}/cot/nohelp_synthetic_cot/raw/{filename}"
    
    if os.path.exists(inferred_path):
        return inferred_path
    
    alt_patterns = [
        f"data/{dataset}/cot/{filename}",
        f"data/{dataset}/cot/synthetic_cot/{filename}",
        f"data/{dataset}/cot/raw/{filename}"
    ]
    
    for pattern in alt_patterns:
        if os.path.exists(pattern):
            return pattern
    
    return inferred_path

def infer_output_path(input_path, output_arg, quality_filter):
    """Infer output path from input path and quality filter."""
    if output_arg and ('/' in output_arg or output_arg.endswith('.csv')):
        return output_arg
    
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    base_name = input_filename.replace('.csv', '')
    
    output_dir = input_dir.replace('/raw/', '/clean/')
    output_dir = output_dir.replace('/raw', '/clean')
    
    if 'raw' not in input_dir:
        if '/cot/' in input_dir:
            parts = input_dir.split('/cot/')
            output_dir = f"{parts[0]}/cot/clean"
        else:
            output_dir = os.path.join(input_dir, 'clean')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if quality_filter:
        if quality_filter == "correct,incomplete":
            suffix = "_correct_incomplete"
        elif quality_filter == "correct":
            suffix = "_correct"
        else:
            suffix = f"_{quality_filter.replace(',', '_')}"
        output_filename = f"{base_name}{suffix}.csv"
    else:
        output_filename = f"{base_name}.csv"
    
    return os.path.join(output_dir, output_filename)

def process_cot_data(args):
    """Process and analyze synthetic CoT data."""
    input_path = infer_input_path(args.input, args.dataset)
    print(f"Using input path: {input_path}")
    
    if args.output:
        output_path = args.output
    else:
        output_path = infer_output_path(input_path, None, args.quality_filter)
    
    print(f"Using output path: {output_path}")
    
    print(f"Loading synthetic CoT data from {input_path}")
    raw_df = pd.read_csv(input_path)
    
    # Remove examples from human_cot if specified
    if not args.keep_human_examples:
        human_cot_path = args.human_cot if args.human_cot else f"data/{args.dataset}/cot/human_cot_single.csv"
        if os.path.exists(human_cot_path):
            print(f"Loading human CoT data from {human_cot_path} to filter out existing examples")
            human_df = pd.read_csv(human_cot_path)
            
            human_examples = set()
            for _, row in human_df.iterrows():
                key = (row["Relation"], row.get("SubjectEntityID", row.get("SubjectEntity", "")))
                human_examples.add(key)
            
            original_len = len(raw_df)
            raw_df = raw_df[~raw_df.apply(lambda row: 
                (row["Relation"], row.get("SubjectEntityID", row.get("SubjectEntity", ""))) in human_examples, 
                axis=1)]
            
            filtered_count = original_len - len(raw_df)
            print(f"Removed {filtered_count} examples that appear in human CoT data")
        else:
            print(f"Warning: Human CoT file not found at {human_cot_path}. No examples removed.")
    
    print(f"Processing {len(raw_df)} entries in {args.mode} mode...")
    
    # Process the data - this adds metrics columns and evaluates quality
    processed_df = process_synthetic_cot_results(
        train_df=raw_df,
        generations=raw_df["CoT"].tolist(),
        mode=args.mode
    )
    
    if args.quality_filter:
        quality_values = args.quality_filter.split(',')
        original_len = len(processed_df)
        processed_df = processed_df[processed_df['quality'].isin(quality_values)]
        filtered_count = original_len - len(processed_df)
        print(f"Filtered out {filtered_count} examples that didn't match quality criteria {quality_values}, {len(processed_df)} examples remaining")
    
    processed_df = processed_df.sort_values(by=['Relation', 'SubjectEntity'], 
                                           ascending=[False, True])
    
    columns = list(processed_df.columns)
    base_columns = ['SubjectEntityID', 'SubjectEntity', 'Relation']
    quality_columns = ['quality', 'precision', 'recall', 'f1']
    answer_columns = ['extracted_answer', 'tp_items', 'fp_items', 'fn_items']
    end_columns = ['CoT', 'ObjectEntitiesID', 'ObjectEntities']
    
    new_columns = []
    
    # Add base columns first
    for col in base_columns:
        if col in columns:
            new_columns.append(col)
    
    # Add quality metrics
    for col in quality_columns:
        if col in columns:
            new_columns.append(col)
    
    # Add answer details
    for col in answer_columns:
        if col in columns:
            new_columns.append(col)
    
    for col in columns:
        if col not in new_columns and col not in end_columns:
            new_columns.append(col)
    
    for col in end_columns:
        if col in columns:
            new_columns.append(col)
    
    processed_df = processed_df[new_columns]
    
    print(f"Saving processed data to {output_path}")
    processed_df.to_csv(output_path, index=False)
    
    quality_counts = processed_df["quality"].value_counts()
    print("\nQuality Distribution Summary:")
    for quality, count in quality_counts.items():
        percentage = 100 * count / len(processed_df)
        print(f"  {quality}: {count} ({percentage:.1f}%)")
    
    if not args.no_relation_stats:
        print("\nRelation Statistics:")
        
        # Get quality counts by relation
        relation_stats = processed_df.groupby("Relation")["quality"].value_counts().unstack().fillna(0)
        
        relation_total = processed_df.groupby("Relation").size()
        relation_correct = processed_df[processed_df["quality"] == "correct"].groupby("Relation").size()
        
        relation_f1 = processed_df.groupby("Relation")["f1"].apply(
            lambda x: pd.to_numeric(x, errors='coerce').mean()
        ).round(3)
        
        # Create a comprehensive stats dataframe
        comprehensive_stats = pd.DataFrame({
            "Total": relation_total,
            "Correct": relation_correct.reindex(relation_total.index).fillna(0).astype(int),
            "% Correct": (100 * relation_correct.reindex(relation_total.index).fillna(0) / relation_total).round(1),
            "Avg F1": relation_f1
        })
        
        for col in relation_stats.columns:
            if col in relation_stats:
                comprehensive_stats[col] = relation_stats[col].reindex(comprehensive_stats.index).fillna(0).astype(int)
        
        comprehensive_stats = comprehensive_stats.sort_values("% Correct", ascending=False)
        
        comprehensive_stats["% Correct"] = comprehensive_stats["% Correct"].astype(str) + '%'
        
        print(comprehensive_stats)
    
    return processed_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and evaluate synthetic CoT data")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file (can be full path, relative path, or just filename)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the processed output CSV file (auto-inferred if not provided)")
    parser.add_argument("--mode", type=str, choices=["help", "nohelp"], default="nohelp",
                        help="The mode used for generation: 'help' (answer provided, not recommended) or 'nohelp' (answer generated)")
    parser.add_argument("--human-cot", type=str, default=None,
                        help="Path to human CoT data (default: data/{dataset}/cot/human_cot_single.csv)")
    parser.add_argument("--dataset", type=str, default="dataset2025",
                        help="Dataset name (used to locate human_cot.csv if --human-cot not provided)")
    parser.add_argument("--keep-human-examples", action="store_true",
                        help="Keep examples that appear in human CoT data (not recommended)")
    parser.add_argument("--quality-filter", type=str, default='correct,incomplete',
                        help="Comma-separated list of quality values to keep (e.g., 'correct,incomplete')")
    parser.add_argument("--no-relation-stats", action="store_true",
                        help="Skip printing relation-specific statistics")
    
    args = parser.parse_args()
    process_cot_data(args)
