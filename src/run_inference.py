import os
import argparse
import pandas as pd
import yaml

from data_utils import load_dataset
from models import (
    load_model, 
    load_peft_model, 
    load_vllm_model, 
    generate_text, 
    is_vllm_available, 
    prepare_prompts,
    format_for_display
)
from config_utils import get_system_message, get_model_path, load_config, generate_unique_experiment_id
from launch_utils import create_experiment_directory, save_config_to_directory
from consistency import aggregate_consistency_predictions

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
    use_vllm = config['model'].get('use_vllm', True)
    use_quantize = config['model'].get('quantize', False)
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
    
    # Get custom prompt file and system message type if specified
    prompt_file = config['prompting'].get('prompt_file', 'question_prompts.csv')
    system_message_type = config['prompting'].get('system_message', 'qa' if not use_cot else 'cot')
    
    # Resolve model path from alias
    cfg = load_config()
    model_path = get_model_path(model_alias, use_quantize, cfg)
    
    # Check if this is from an existing experiment directory
    experiment_dir = os.path.dirname(os.path.abspath(config_path))
    in_predictions_dir = "predictions" in experiment_dir and os.path.basename(config_path) == "config.yaml"
    
    if in_predictions_dir:
        output_dir = experiment_dir
        print(f"Using existing experiment directory: {output_dir}")
    else:
        job_id = os.environ.get('SLURM_JOB_ID')
        unique_id = generate_unique_experiment_id(job_id)
        output_dir = create_experiment_directory(dataset_name, unique_id, job_id)
        
        save_config_to_directory(config_path, output_dir)
    
    # Load dataset with the specified prompt file and split
    train_df, val_df, question_prompts = load_dataset(dataset_name, prompt_file, split=split)
    print(f"Loaded prompt templates from: {prompt_file}")
    print(f"Using data split: {split}")

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
    
    # Load model based on settings
    if use_vllm and is_vllm_available():
        # Pass the quantize parameter to vLLM - consistent with synthetic_cot.py
        model, tokenizer = load_vllm_model(model_path, quantized=use_quantize)
        backend = "vllm"
    else:
        model, tokenizer = load_model(model_path, use_quantization=use_quantize)
        backend = "hf"
    
    # Get system message based on specified type, falling back to standard types if not provided
    system_message = get_system_message(system_message_type)
    print(f"Using system message type: {system_message_type}")
    
    # Generate responses
    print(f"Generating responses for {len(subjects)} examples...")
    if n_consistency > 1:
        print(f"Using self-consistency with {n_consistency} predictions per example")
    
    # For consistency, we need to generate multiple times
    all_outputs = []
    for consistency_run in range(n_consistency):
        # Prepare prompts for this consistency run (different few-shot examples each time)
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
        
        # Determine if Qwen3-8B for enable_thinking
        model_name_lower = str(model_path).split('/')[-1].lower()
        is_qwen3 = "qwen3-" in model_name_lower
        enable_thinking = use_cot if is_qwen3 else None

        # Generate outputs for this run
        outputs = generate_text(
            model=model,
            prompts=raw_prompts,
            tokenizer=tokenizer,
            backend=backend,
            generation_params={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            debug=debug_mode and consistency_run == 0,  # Only debug first run
            debug_samples=min(5, len(raw_prompts)),
            enable_thinking=enable_thinking
        )
        
        all_outputs.append(outputs)
    
    # Create predictions dataframe
    predictions = []
    for i, (subject, relation) in enumerate(zip(subjects, relations)):
        row_id = val_df.iloc[i]["SubjectEntityID"] if "SubjectEntityID" in val_df.columns else str(i)

        # Extract ObjectEntities (string answers) immediately after generation
        # Use the same logic as in Disambiguator._extract_entities_from_generated
        def extract_entities_from_generated(raw_generated):
            if "<think>" in raw_generated and "</think>" in raw_generated:
                try:
                    answer_part = raw_generated.split("</think>", 1)[1].strip()
                except IndexError:
                    answer_part = raw_generated
            else:
                answer_part = raw_generated
            return answer_part

        # Collect all outputs for this example across consistency runs
        all_generated = []
        all_clean_outputs = []
        for consistency_run in range(n_consistency):
            output = all_outputs[consistency_run][i]
            all_generated.append(output)
            
            clean_output = extract_entities_from_generated(output)
            object_entities = [e.strip() for e in clean_output.split(",") if e.strip()]
            all_clean_outputs.append(str(object_entities))

        # For compatibility, use first output as main "generated" field
        main_generated = all_generated[0]
        main_clean_output = all_clean_outputs[0]

        predictions.append({
            "SubjectEntityID": row_id, 
            "SubjectEntity": subject,
            "Relation": relation,
            "ObjectEntitiesID": None,
            "ObjectEntities": main_clean_output,
            "generated": main_generated,
            "object_entities_list": str(all_clean_outputs),
            "generated_list": str(all_generated),
            "n_consistency": n_consistency
        })

    # Save predictions
    pd.DataFrame(predictions).to_csv(predictions_csv, index=False)
    print(f"Predictions saved to {predictions_csv}")
    
    # If multiple consistency runs, aggregate the results automatically
    if n_consistency > 1:
        print(f"Running consistency aggregation for {n_consistency} predictions per example...")
        try:
            aggregated_df = aggregate_consistency_predictions(
                predictions_csv,
                strategy="relation-threshold",
                save_to_file=True
            )
            print(f"Consistency aggregation completed. ObjectEntities field now contains aggregated results.")
        except Exception as e:
            print(f"Warning: Consistency aggregation failed with error: {e}")
    
    # After saving predictions, run string-based evaluation and save results as results_string.json
    # Only run evaluation for non-test splits as test sets don't have ground truth answers
    if split != "test":
        from evaluate import evaluate
        ground_truth_path = f"data/{dataset_name}/data/{split}.csv"
        if os.path.exists(ground_truth_path):
            # Check if this is a test set by looking for ObjectEntities column in ground truth
            try:
                gt_df = pd.read_csv(ground_truth_path)
                has_answers = "ObjectEntities" in gt_df.columns and not gt_df["ObjectEntities"].isna().all()
                
                if has_answers:
                    print("Running string-based evaluation (2025 method)...")
                    results_df = evaluate(predictions_csv, ground_truth_path, mode="string")
                    out_json = os.path.join(output_dir, "results_string.json")
                    results_df.to_json(out_json, orient="index", indent=2)
                    print(f"String-based evaluation results saved to {out_json}")
                else:
                    print(f"Skipping evaluation: Ground truth file exists but appears to be a test set (no answers)")
            except Exception as e:
                print(f"Warning: Could not check ground truth file format: {e}")
        else:
            print(f"Ground truth file not found for string evaluation: {ground_truth_path}")
    else:
        print(f"Skipping evaluation for test split - no ground truth available")
    
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
        },
        "dataset": {
            "name": args.dataset,
            "limit": args.limit
        },
        "prompting": {
            "few_shot": args.few_shot,
            "cot": args.cot,
            "cot_source": "train",
            "prompt_file": args.prompt_file,
            "system_message": args.system_message,
            "n_consistency": getattr(args, 'n_consistency', 1)
        },
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": 1.
        }
    }
    
    job_id = os.environ.get('SLURM_JOB_ID')
    unique_id = generate_unique_experiment_id(job_id)
    tmp_config = f"tmp_config_{unique_id}.yaml"
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
    parser.add_argument("--vllm", action="store_true", help="Use vLLM for generation if available")
    parser.add_argument("--quantize", action="store_true", help="Use quantization for HF models")
    parser.add_argument("--few_shot", type=int, default=2, help="Number of few-shot examples")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--limit", type=int, default=0, help="Limit examples (0 = no limit)")
    parser.add_argument("--prompt_file", type=str, default="question_prompts.csv", 
                        help="Prompt template file to use (e.g., relation_prompts.csv)")
    parser.add_argument("--system_message", type=str, default="qa",
                        help="System message type (qa, cot, direct)")
    parser.add_argument("--n_consistency", type=int, default=1,
                        help="Number of consistency predictions per example")
    
    args = parser.parse_args()
    
    # Enforce experiment name if not using config file
    if not args.config and not args.experiment_name:
        parser.error("--experiment_name is required when not using --config")
    
    run_inference(args)
