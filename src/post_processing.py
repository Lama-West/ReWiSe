import pandas as pd
import argparse
import re
from tqdm import tqdm
from prompt_utils import format_answer

def clean_cot(cot, expected_answers):
    """
    Clean the CoT by extracting the thinking part and replacing anything 
    after </think> with the formatted answer.
    
    Args:
        cot: The Chain-of-Thought text to clean
        expected_answers: List of expected answers
    
    Returns:
        str: Cleaned CoT text with original thinking but standardized answer format,
             or None if the format is invalid
    """
    # Check if the CoT has the proper format
    if not (cot.strip().startswith('<think>') and '</think>' in cot):
        return None
    
    if '\n' in cot:
        return None
    
    # Extract the thinking part
    thinking_match = re.search(r'<think>(.*?)</think>', cot, re.DOTALL)
    if not thinking_match:
        return None
    
    thinking_part = thinking_match.group(1).strip()
    
    # Format the expected answers
    formatted_answer = format_answer(expected_answers)
    
    # Combine thinking part with formatted answer
    return f"<think> {thinking_part} </think> {formatted_answer}"

def process_cot_file(input_path, output_path):
    """
    Process a synthetic CoT file by:
    1. Filtering out entries without proper <think>...</think> tags
    2. Replacing content after </think> with formatted answers
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the processed CSV file
    
    Returns:
        tuple: (original_count, filtered_count, formatted_count)
    """
    print(f"Reading input file: {input_path}")
    df = pd.read_csv(input_path)
    original_count = len(df)
    
    print("Processing CoT entries...")
    formatted_rows = []
    dropped_rows = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cot = row.get('CoT', '')
        object_entities = eval(row.get('ObjectEntities', '[]'))
        
        cleaned_cot = clean_cot(cot, object_entities)
        
        if cleaned_cot:
            # Update the CoT field with the cleaned version
            row_dict = row.to_dict()
            row_dict['CoT'] = cleaned_cot
            formatted_rows.append(row_dict)
        else:
            dropped_rows.append(row.to_dict())
    
    # Create DataFrames from the processed rows
    formatted_df = pd.DataFrame(formatted_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    
    formatted_count = len(formatted_df)
    filtered_count = original_count - formatted_count
    
    # Save the cleaned data
    print(f"Writing {formatted_count} formatted entries to: {output_path}")
    formatted_df.to_csv(output_path, index=False)
    
    # Also save dropped entries if there are any
    if filtered_count > 0:
        dropped_path = output_path.replace('.csv', '_dropped.csv')
        print(f"Writing {filtered_count} dropped entries to: {dropped_path}")
        dropped_df.to_csv(dropped_path, index=False)
    
    return original_count, filtered_count, formatted_count

def main():
    parser = argparse.ArgumentParser(description="Post-process synthetic CoT data")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--dataset", type=str, default="dataset2024", help="Dataset name (e.g., dataset2024, dataset2025)")
    
    args = parser.parse_args()
    
    original_count, filtered_count, formatted_count = process_cot_file(args.input, args.output)
    formatted_df = pd.read_csv(args.output)
    
    print("\nSummary:")
    print(f"Original entries: {original_count}")
    print(f"Entries filtered out: {filtered_count} ({filtered_count/original_count*100:.2f}%)")
    print(f"Entries formatted and kept: {formatted_count} ({formatted_count/original_count*100:.2f}%)")
    print("Statistics on kept entries:")
    print("  Average CoT length per relation:")
    print(formatted_df.groupby('Relation')['CoT'].apply(lambda x: x.str.len().mean()))

    # Statistics for human CoT file based on dataset
    human_cot_path = f"data/{args.dataset}/cot/human_cot_single.csv"
    human_cot_df = pd.read_csv(human_cot_path)
    print("\nHuman CoT statistics:")
    print("  Average CoT length per relation:")
    print(human_cot_df.groupby('Relation')['CoT'].apply(lambda x: x.str.len().mean()))


if __name__ == "__main__":
    main()
