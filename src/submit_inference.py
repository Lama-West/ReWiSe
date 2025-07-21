import os
import sys
import json
import argparse
import yaml
import subprocess
from config_utils import generate_unique_experiment_id, create_unique_config_path

def determine_hardware_config(model_alias, trained_model_dir=None):
    """
    Determine SLURM hardware configuration based on model.
    
    Args:
        model_alias: Model alias (e.g., 'l_8i', 'q_8i')
        trained_model_dir: Path to trained model directory (for PEFT models)
        
    Returns:
        dict: Hardware configuration for SLURM
    """
    # For PEFT models, check the base model from config
    if trained_model_dir and os.path.exists(os.path.join(trained_model_dir, "config.json")):
        with open(os.path.join(trained_model_dir, "config.json"), 'r') as f:
            config = json.load(f)
            base_model = config.get("model", "")
    
    # For now, assume 8B models for which can use single A100
    return {
        "gres": "gpu:a100:1", 
        "cpus_per_task": 8,
        "mem": "64G",
        "time": "3:00:00"
    }

def determine_system_message(use_cot, prompt_file):
    """Determine system message type based on CoT usage and prompt file."""
    if prompt_file in ["empty_prompts.csv", "relation_prompts.csv"]:
        return "direct"
    elif use_cot:
        return "cot"
    else:
        return "qa"

def determine_max_tokens(use_cot, max_tokens_override=None):
    """Determine max tokens based on CoT usage."""
    if max_tokens_override is not None:
        return max_tokens_override
    return 1000 if use_cot else 100

def get_synthetic_cot_path(dataset, cot_source):
    """Get path to synthetic CoT data."""
    if cot_source == "synthetic":
        # Default synthetic
        return f"data/{dataset}/cot/nohelp_synthetic_cot/clean/2shot_temp1.0_lambda70b_group_correct_incomplete.csv"
    
    if not cot_source.endswith(".csv"):
        cot_source = f"{cot_source}.csv"
    
    # Check if it's an absolute path and exists
    if cot_source.startswith("/") and os.path.exists(cot_source):
        return cot_source
    
    # Try multiple possible locations in order of preference
    possible_paths = [
        # In case exact path is provided
        cot_source,
        # Should be there in most cases
        f"data/{dataset}/cot/nohelp_synthetic_cot/clean/{cot_source}",
        f"data/{dataset}/cot/nohelp_synthetic_cot/{cot_source}",
        f"data/{dataset}/cot/{cot_source}",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If none found, return the preferred path (will cause error later if doesn't exist)
    return f"data/{dataset}/cot/nohelp_synthetic_cot/clean/{cot_source}"

def generate_config_yaml(args, job_id=None):
    """Generate YAML configuration for the experiment."""
    unique_id = generate_unique_experiment_id(job_id)
    
    if not args.prompt_file.startswith("/") and not args.prompt_file.startswith("data/"):
        prompt_file_path = f"data/{args.dataset}/{args.prompt_file}"
    else:
        prompt_file_path = args.prompt_file
    
    use_peft = args.trained_model_dir is not None
    system_message = determine_system_message(args.cot, args.prompt_file)
    max_tokens = determine_max_tokens(args.cot, args.max_tokens)
    
    if use_peft:
        model_name = os.path.basename(os.path.normpath(args.trained_model_dir))
        
        config_path = os.path.join(args.trained_model_dir, "config.json")
        base_model = None
        peft_method = "prompt_tuning"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                train_config = json.load(f)
                base_model = train_config.get("model")
                peft_method = train_config.get("peft_method", "prompt_tuning")
        
        if not base_model:
            raise ValueError(f"Could not determine base model from {config_path}")
        
        experiment_name = f"peft_{unique_id}"
        
        config = {
            "experiment": {
                "name": experiment_name
            },
            "model": {
                "use_peft": True,
                "trained_model_dir": args.trained_model_dir,
                "base": base_model,
                "quantize": args.quantize,
                "epoch": args.epoch
            },
            "dataset": {
                "name": args.dataset,
                "split": args.split,
                "limit": args.limit
            },
            "prompting": {
                "few_shot": 0,  # PEFT models don't use few-shot as of 03/06
                "cot": args.cot,
                "prompt_file": prompt_file_path,
                "system_message": system_message
            },
            "generation": {
                "temperature": args.temperature,
                "max_tokens": max_tokens,
                "n_consistency": args.n_consistency
            }
        }
        
        # Add optional report field if provided
        if args.report:
            config["experiment"]["report"] = args.report
        
        # Add CoT source if using CoT
        if args.cot:
            config["prompting"]["cot_source"] = get_synthetic_cot_path(args.dataset, args.cot_source)
                
    else:
        # Pretrained model configuration
        experiment_name = unique_id
        
        config = {
            "experiment": {
                "name": experiment_name
            },
            "model": {
                "alias": args.model,
                "use_vllm": args.use_vllm,
                "quantize": args.quantize,
                "use_peft": False
            },
            "dataset": {
                "name": args.dataset,
                "split": args.split,
                "limit": args.limit
            },
            "prompting": {
                "few_shot": args.few_shot,
                "cot": args.cot,
                "prompt_file": prompt_file_path,
                "system_message": system_message
            },
            "generation": {
                "temperature": args.temperature,
                "max_tokens": max_tokens,
                "n_consistency": args.n_consistency
            }
        }
        
        # Add optional report field if provided
        if args.report:
            config["experiment"]["report"] = args.report
        
        # Add CoT source if using CoT
        if args.cot:
            config["prompting"]["cot_source"] = get_synthetic_cot_path(args.dataset, args.cot_source)
    
    return config, experiment_name

def submit_slurm_job(config_path, hardware_config, experiment_name, args):
    """Submit SLURM job for the experiment."""
    
    # Determine model to copy
    if args.trained_model_dir:
        # For PEFT models get base model from config
        try:
            config_json_path = os.path.join(args.trained_model_dir, "config.json")
            with open(config_json_path, 'r') as f:
                peft_config = json.load(f)
                base_model = peft_config.get("model")
        except:
            raise ValueError(f"Could not load PEFT model config from {config_json_path}")
        model_to_copy = base_model
    else:
        model_to_copy = args.model
    
    # Create SLURM script content
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={experiment_name[:20]}  # Truncate for SLURM
#SBATCH --output=out_jobs/%x-%j.out
#SBATCH --error=out_jobs/%x-%j.err
#SBATCH --time={hardware_config['time']}
#SBATCH --account=def-azouaq
#SBATCH --gres={hardware_config['gres']}
#SBATCH --cpus-per-task={hardware_config['cpus_per_task']}
#SBATCH --mem={hardware_config['mem']}

# Setup environment
source $HOME/.bashrc
source $HOME/myenv/bin/activate

# Navigate to project directory
cd /home/edarsem/projects/def-azouaq/edarsem/soft_knowledge_retrieval

# Create output directory if it doesn't exist
mkdir -p out_jobs

# Copy model to SLURM_TMPDIR if available
if [ -n "$SLURM_TMPDIR" ]; then
    echo "Copying model to SLURM_TMPDIR..."
    if [ -f scripts/copy_model.sh ]; then
        ./scripts/copy_model.sh "{model_to_copy}" "{'quantized' if args.quantize else ''}"
        echo "Model copied to SLURM_TMPDIR, proceeding with inference at $(date)"
    else
        echo "Warning: copy_model.sh not found, proceeding without model copy"
    fi
else
    echo "Warning: SLURM_TMPDIR not available, proceeding without model copy"
fi

# Run inference
echo "Starting inference with config: {config_path}"
"""
    
    # Choose the correct inference script based on whether we're using PEFT
    if args.trained_model_dir:
        # Use inference_peft.py for PEFT models
        slurm_script += f"""python src/inference_peft.py \\
    --trained-model-dir {args.trained_model_dir} \\
    --dataset {args.dataset} \\
    --max-tokens {determine_max_tokens(args.cot, args.max_tokens)} \\
    --temperature {args.temperature} \\
    --n-consistency {args.n_consistency}"""
        
        # Add optional arguments
        if args.quantize:
            slurm_script += " \\\n    --quantize"
        if args.epoch is not None:
            slurm_script += f" \\\n    --epoch {args.epoch}"
        if args.limit > 0:
            slurm_script += f" \\\n    --limit {args.limit}"
        if args.cot:
            slurm_script += " \\\n    --use-cot"
            
        slurm_script += "\n"
    else:
        # Use run_inference.py for base models  
        slurm_script += f"python src/run_inference.py --config {config_path}\n"
    
    slurm_script += """
"""
    
    slurm_script += """
echo "Job completed at $(date)"
"""

    # Write SLURM script to temporary file
    slurm_script_path = f"tmp_slurm_{experiment_name}.sh"
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script)
    
    try:
        # Submit the job
        if args.dry_run:
            print(f"DRY RUN: Would submit SLURM job with script:")
            print(slurm_script)
            print(f"Config file: {config_path}")
            return True, None
        else:
            result = subprocess.run(['sbatch', slurm_script_path], 
                                  capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            print(f"Submitted SLURM job {job_id} for experiment {experiment_name}")
            print(f"Config: {config_path}")
            print(f"Hardware: {hardware_config}")
            return True, job_id
            
    except subprocess.CalledProcessError as e:
        print(f"Error submitting SLURM job: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, None
    finally:
        # Clean up temporary SLURM script
        if os.path.exists(slurm_script_path):
            os.remove(slurm_script_path)

def main():
    parser = argparse.ArgumentParser(description="Submit inference experiments to SLURM")
    
    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, 
                           help="Model alias for pretrained models (e.g., l_8i, q_8i)")
    model_group.add_argument("--trained-model-dir", type=str,
                           help="Path to trained PEFT model directory")
    
    # Dataset and split options
    parser.add_argument("--dataset", type=str, default="dataset2024",
                       help="Dataset name (default: dataset2024)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                       help="Data split to use (default: val)")
    
    # Prompting options
    parser.add_argument("--few-shot", type=int, default=5,
                       help="Number of few-shot examples (default: 5, ignored for PEFT)")
    parser.add_argument("--cot", action="store_true",
                       help="Use chain-of-thought prompting")
    parser.add_argument("--cot-source", type=str, default="synthetic",
                       help="CoT data source: 'synthetic', 'train', or specific filename (default: synthetic)")
    parser.add_argument("--prompt-file", type=str, default="question_prompts.csv",
                       help="Prompt template file (default: question_prompts.csv)")
    
    # Generation options  
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Generation temperature (default: 1.0)")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Max tokens to generate (default: auto-determined from CoT)")
    parser.add_argument("--n-consistency", type=int, default=1,
                       help="Number of consistency predictions (default: 1)")
    
    # Model options
    parser.add_argument("--use-vllm", action="store_true", default=True,
                       help="Use vLLM for generation (default: True)")
    parser.add_argument("--quantize", action="store_true",
                       help="Use quantization")
    parser.add_argument("--epoch", type=int, default=None,
                       help="Specific epoch for PEFT models (default: final)")
    
    # Experiment options
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of examples (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print configuration and SLURM script without submitting")
    
    # Output options
    parser.add_argument("--config-dir", type=str, default="configs",
                       help="Directory to save config files (default: configs)")
    parser.add_argument("--report", type=str, default=None,
                       help="Optional experiment report description for analysis")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.trained_model_dir and not os.path.exists(args.trained_model_dir):
        print(f"Error: Trained model directory does not exist: {args.trained_model_dir}")
        sys.exit(1)
    
    if args.cot and not args.cot_source:
        print("Error: --cot-source is required when using --cot")
        sys.exit(1)
    
    # Warning for PEFT + few-shot
    if args.trained_model_dir and args.few_shot > 0:
        print(f"Warning: Few-shot examples ({args.few_shot}) are ignored for PEFT models")
    
    # Create config directory
    os.makedirs(args.config_dir, exist_ok=True)
    
    # Get the current launcher job ID from SLURM environment
    launcher_job_id = os.environ.get('SLURM_JOB_ID')
    
    # Generate configuration
    try:
        config, experiment_name = generate_config_yaml(args, launcher_job_id)
    except Exception as e:
        print(f"Error generating configuration: {e}")
        sys.exit(1)
    
    # Create unique config path
    config_path, final_experiment_name = create_unique_config_path(args.config_dir, job_id=launcher_job_id)
    
    # Update config with final experiment name if it changed
    if final_experiment_name != experiment_name:
        config["experiment"]["name"] = final_experiment_name
        experiment_name = final_experiment_name
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated config: {config_path}")
    
    # Determine hardware configuration
    model_ref = args.model if args.model else args.trained_model_dir
    hardware_config = determine_hardware_config(model_ref, args.trained_model_dir)
    
    # Submit SLURM job
    success, job_id = submit_slurm_job(config_path, hardware_config, experiment_name, args)
    
    # Update config with job information if job was submitted successfully
    if success and job_id and not args.dry_run:
        # Get the current launcher job ID from SLURM environment
        launcher_job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        
        # Add job names to config
        config["job_name"] = {
            "launcher": launcher_job_id,
            "experiment": job_id
        }
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Updated config with job IDs: launcher={launcher_job_id}, experiment={job_id}")
    
    if success and not args.dry_run:
        print(f"Experiment submitted successfully!")
    elif args.dry_run:
        print("Dry run completed. Use --no-dry-run to actually submit the job.")
    else:
        print("Failed to submit experiment.")
        sys.exit(1)

if __name__ == "__main__":
    main()
