import argparse
import pandas as pd
import json
import os
import ast
from pathlib import Path

def convert_csv_to_submission_jsonl(predictions_csv_path, output_jsonl_path):
    """
    Convert predictions CSV to the submission JSONL format.
    
    Args:
        predictions_csv_path: Path to the predictions CSV file
        output_jsonl_path: Path where to save the submission JSONL file
    """
    # Load predictions
    df = pd.read_csv(predictions_csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # Convert to submission format
    submission_entries = []
    
    for _, row in df.iterrows():
        # Parse ObjectEntities - handle both string representation of list and direct strings
        object_entities = row['ObjectEntities']
        
        if pd.isna(object_entities) or object_entities == '' or object_entities == 'None':
            # Handle empty/None cases
            object_entities_list = []
        elif isinstance(object_entities, str):
            try:
                # Try to parse as Python list representation
                if object_entities.startswith('[') and object_entities.endswith(']'):
                    object_entities_list = ast.literal_eval(object_entities)
                else:
                    # Treat as comma-separated string
                    object_entities_list = [e.strip() for e in object_entities.split(',') if e.strip()]
            except (ValueError, SyntaxError):
                # If parsing fails, treat as single string
                object_entities_list = [object_entities.strip()] if object_entities.strip() else []
        else:
            object_entities_list = []
        
        # Create submission entry
        submission_entry = {
            "SubjectEntity": row['SubjectEntity'],
            "SubjectEntityID": row['SubjectEntityID'],
            "ObjectEntities": object_entities_list,
            "ObjectEntitiesID": [],  # We don't predict entity IDs, only names
            "Relation": row['Relation']
        }
        
        submission_entries.append(submission_entry)
    
    # Write to JSONL file
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in submission_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Successfully converted {len(submission_entries)} predictions to {output_jsonl_path}")
    return output_jsonl_path

def create_submission_zip(jsonl_path, zip_path=None):
    """
    Create a ZIP file containing the predictions.jsonl for submission.
    
    Args:
        jsonl_path: Path to the predictions.jsonl file
        zip_path: Optional path for the ZIP file (defaults to same directory)
    """
    import zipfile
    
    if zip_path is None:
        zip_path = str(jsonl_path) + '.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(jsonl_path, 'predictions.jsonl')
    
    print(f"Created submission ZIP file: {zip_path}")
    return zip_path

def main():
    parser = argparse.ArgumentParser(description="Convert predictions CSV to submission format")
    parser.add_argument("predictions_csv", help="Path to predictions CSV file")
    parser.add_argument("--zip-path", help="Custom path for ZIP file (optional)")
    
    args = parser.parse_args()
    
    # Auto-generate output path by replacing .csv with .jsonl
    predictions_csv_path = Path(args.predictions_csv)
    output_jsonl_path = predictions_csv_path.with_suffix('.jsonl')
    
    # Convert CSV to JSONL
    jsonl_path = convert_csv_to_submission_jsonl(args.predictions_csv, str(output_jsonl_path))
    
    # Always create ZIP file
    zip_path = create_submission_zip(jsonl_path, args.zip_path)
    print(f"Submission ready: {zip_path}")
    
if __name__ == "__main__":
    main()
