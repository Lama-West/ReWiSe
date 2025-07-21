import os
import argparse
import pandas as pd
from datetime import datetime
import json
import yaml
import requests  # Add this for API calls
from openai import OpenAI

from data_utils import load_dataset
from models import (
    prepare_prompts
)
from config_utils import get_system_message, get_model_path, load_config
from launch_utils import create_experiment_directory, save_config_to_directory
from prompt_utils import validate_cot_data
from consistency import aggregate_consistency_predictions
from tracking import Experiment

# Add Lambda API helper function
def generate_with_lambda_api(prompts, temperature=1.0, max_tokens=100, model_name="llama3.3-70b-instruct-fp8"):
    """
    Generate text using Lambda API.
    
    Args:
        prompts: List of prompts (strings or message lists)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        model_name: Model name for Lambda API
        
    Returns:
        list: Generated texts
    """
    API_KEY = open("ideas/lambda_api").read().strip()
    API_URL = "https://api.lambdalabs.com/v1"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_URL
        )
    model = 'llama3.3-70b-instruct-fp8'
    
    outputs = []

    from tqdm import tqdm
    
    for i, prompt in tqdm(enumerate(prompts)):
        # Format prompt for Lambda API
        if isinstance(prompt, list):  # It's a structured chat message
            messages = prompt
        else:
            raise ValueError("Prompts must be structured chat messages for Lambda API")
        
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        response = chat_completion.choices[0].message.content
        outputs.append(response)
        
    return outputs

def run_from_config(config_path):
    """
    Run inference using parameters from a YAML config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        str: Path to predictions file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    dataset_name = config['dataset']['name']
    split = config['dataset'].get('split', 'val')  # Get split from config
    model_alias = config['model']['alias']
    few_shot = config['prompting'].get('few_shot', 0)
    few_shot_other = config['prompting'].get('few_shot_other', 0)
    use_cot = config['prompting'].get('cot', False)
    cot_source = config['prompting'].get('cot_source', 'human')
    n_consistency = config['generation'].get('n_consistency', 1)
    temperature = config['generation'].get('temperature', 1.)
    max_tokens = config['generation'].get('max_tokens', 100)
    limit = config['dataset'].get('limit', 0)
    experiment_name = config['experiment']['name']
    debug_mode = config['generation'].get('debug', False)
    
    # Set up experiment directory
    experiment_dir = os.path.dirname(os.path.abspath(config_path))
    in_predictions_dir = "predictions" in experiment_dir and os.path.basename(config_path) == "config.yaml"
    
    if in_predictions_dir:
        # Use the existing experiment directory
        output_dir = experiment_dir
        print(f"Using existing experiment directory: {output_dir}")
    else:
        # Create a new directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = create_experiment_directory(dataset_name, timestamp)
        
        # Copy config to output directory
        save_config_to_directory(config_path, output_dir)
    
    # Load dataset
    # Use the prompt_file from config if specified, otherwise default
    prompt_file = config['prompting'].get('prompt_file', 'question_prompts.csv')
    train_df, val_df, question_prompts = load_dataset(dataset_name, prompt_file, split=split)

    # Robust CoT file path resolution (for use_cot)
    if use_cot and cot_source != 'train':
        cot_path = cot_source
        # If no .csv extension, add it
        if not cot_path.endswith('.csv'):
            cot_path += '.csv'
        # If not an absolute or relative path, assume data/{dataset}/cot/
        if not os.path.exists(cot_path):
            candidate_path = os.path.join('data', dataset_name, 'cot', cot_path)
            if os.path.exists(candidate_path):
                cot_path = candidate_path
        try:
            cot_df = pd.read_csv(cot_path)
            print(f"Loaded {len(cot_df)} CoT examples from {cot_path}")
            train_df = cot_df
        except Exception as e:
            print(f"Warning: Failed to load CoT file '{cot_path}': {e}")
            print("Proceeding with original train_df.")

    # Determine system prompt type based on prompt file
    if prompt_file and 'relation_prompts' in prompt_file:
        system_prompt_type = 'direct'
    else:
        system_prompt_type = 'cot' if use_cot else 'qa'
    system_message = get_system_message(system_prompt_type)
    
    # Limit examples if requested
    if limit > 0:
        print(f"Limiting to {limit} examples for testing")
        val_df = val_df.head(limit)
    
    # Set up paths
    predictions_csv = f"{output_dir}/predictions.csv"
    
    # Prepare for generation
    subjects = val_df["SubjectEntity"].tolist()
    relations = val_df["Relation"].tolist()
    question_templates = [
        question_prompts[question_prompts["Relation"] == rel]["PromptTemplate"].iloc[0]
        for rel in relations
    ]
    
    # Prepare prompts (no need for local model loading)
    print(f"Generating responses for {len(subjects)} examples...")
    
    # We still need model_path for the is_instruct check in prepare_prompts
    # But we don't actually load the model
    cfg = load_config()
    model_path = get_model_path(model_alias, False, cfg)
    
    raw_prompts = prepare_prompts(
        model_name_or_path=model_path,
        subjects=subjects,
        relations=relations,
        question_templates=question_templates,
        train_df=train_df,
        question_prompts=question_prompts,
        few_shot=few_shot,
        few_shot_other=few_shot_other,
        use_cot=use_cot,
        system_message=system_message
    )
    
    # Print the first example's full conversation
    if raw_prompts:
        print("\nFIRST EXAMPLE (full conversation):\n" + "="*60)
        for turn in raw_prompts[0]:
            print(f"[{turn['role'].upper()}]: {turn['content']}")
        print("="*60)
    
    # Generate outputs using Lambda API
    lambda_model_name = "llama3.3-70b-instruct-fp8"  # Adjust if needed for Lambda's naming
    
    # Handle consistency runs
    all_predictions = []
    
    for consistency_run in range(n_consistency):
        print(f"Running consistency iteration {consistency_run + 1}/{n_consistency}")
        
        outputs = generate_with_lambda_api(
            prompts=raw_prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=lambda_model_name
        )
        
        # Create predictions dataframe for this run
        predictions = []
        for i, (subject, relation, output) in enumerate(zip(subjects, relations, outputs)):
            row_id = val_df.iloc[i]["SubjectEntityID"] if "SubjectEntityID" in val_df.columns else str(i)
            
            predictions.append({
                "SubjectEntityID": row_id, 
                "SubjectEntity": subject,
                "Relation": relation,
                "ObjectEntitiesID": None,
                "ObjectEntities": None,
                "generated": output,
                "consistency_run": consistency_run
            })
        
        all_predictions.extend(predictions)
    
    # Save all predictions
    all_predictions_df = pd.DataFrame(all_predictions)
    predictions_csv = f"{output_dir}/predictions.csv"
    all_predictions_df.to_csv(predictions_csv, index=False)
    print(f"All predictions saved to {predictions_csv}")
    
    # If using consistency, aggregate the predictions
    if n_consistency > 1:
        print(f"Aggregating {n_consistency} consistency predictions...")
        aggregated_df = aggregate_consistency_predictions(
            predictions_csv, 
            strategy="relation-threshold",
            debug=debug_mode
        )
        
        # Save aggregated predictions
        aggregated_csv = f"{output_dir}/predictions_aggregated.csv"
        aggregated_df.to_csv(aggregated_csv, index=False)
        print(f"Aggregated predictions saved to {aggregated_csv}")
        
        return aggregated_csv
    else:
        return predictions_csv

def run_inference(args):
    """
    Legacy function for running inference from command-line args.
    Now delegates to run_from_config if config file is provided.
    """
    # If config file is provided, use it
    if args.config:
        return run_from_config(args.config)
    
    # Otherwise, show a deprecation warning
    print("WARNING: Running inference from command line args is deprecated.")
    print("Please use a YAML configuration file instead.")
    
    # Create a simple config dict from args
    config = {
        "experiment": {
            "name": args.experiment_name
        },
        "model": {
            "alias": args.model,
            "use_vllm": args.vllm,
            "quantize": args.quantize,
            "use_peft": args.use_peft
        },
        "dataset": {
            "name": args.dataset,
            "limit": args.limit
        },
        "prompting": {
            "few_shot": args.few_shot,
            "cot": args.cot,
            "cot_source": "train"
        },
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": 1.
        }
    }
    
    # Save to a temporary config file
    tmp_config = f"tmp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Run with the temporary config file
        result = run_from_config(tmp_config)
        # Clean up
        os.remove(tmp_config)
        return result
    except Exception as e:
        # Clean up on error too
        os.remove(tmp_config)
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with any model")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    # Legacy arguments (for backward compatibility)
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="dataset2024", help="Dataset name")
    parser.add_argument("--relation", type=str, help="Optional relation filter")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT model")
    parser.add_argument("--vllm", action="store_true", help="Use vLLM for generation if available")
    parser.add_argument("--quantize", action="store_true", help="Use quantization for HF models")
    parser.add_argument("--few_shot", type=int, default=2, help="Number of few-shot examples")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--limit", type=int, default=0, help="Limit examples (0 = no limit)")
    
    args = parser.parse_args()
    
    # Enforce experiment name if not using config file
    if not args.config and not args.experiment_name:
        parser.error("--experiment_name is required when not using --config")
    
    run_inference(args)
