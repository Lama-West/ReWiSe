import os
import yaml
import json
from config_utils import generate_unique_experiment_id

def load_yaml_config(yaml_path):
    """
    Load a YAML configuration file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration parameters
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_experiment_directory(dataset, timestamp=None, job_id=None):
    """Create a timestamped directory for experiment outputs, ensuring uniqueness."""
    if timestamp is None:
        timestamp = generate_unique_experiment_id(job_id)
    
    base_dir = f"predictions/{dataset}"
    output_dir = f"{base_dir}/{timestamp}"
    
    counter = 1
    while os.path.exists(output_dir):
        new_timestamp = f"{timestamp}_{counter}"
        output_dir = f"{base_dir}/{new_timestamp}"
        counter += 1
        
        if counter > 100:
            timestamp = generate_unique_experiment_id(job_id)
            output_dir = f"{base_dir}/{timestamp}"
            break
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_trained_model_directory(model_alias, use_cot=False, base_dir="trained_models", timestamp=None, job_id=None):
    """Create a unique directory for trained models: trained_models/timestamp_modelalias{_cot}"""
    if timestamp is None:
        timestamp = generate_unique_experiment_id(job_id)
    
    run_name = f"{timestamp}_{model_alias}{'_cot' if use_cot else ''}"
    output_dir = os.path.join(base_dir, run_name)
    counter = 1
    while os.path.exists(output_dir):
        new_timestamp = f"{timestamp}_{counter}"
        run_name = f"{new_timestamp}_{model_alias}{'_cot' if use_cot else ''}"
        output_dir = os.path.join(base_dir, run_name)
        counter += 1
        if counter > 100:
            timestamp = generate_unique_experiment_id(job_id)
            run_name = f"{timestamp}_{model_alias}{'_cot' if use_cot else ''}"
            output_dir = os.path.join(base_dir, run_name)
            break
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, run_name.split('_')[0]

def save_config_to_directory(config, directory):
    """
    Save config to the experiment directory.
    
    Args:
        config: Configuration dictionary or path to YAML file
        directory: Directory to save the config
    """
    if isinstance(config, str):
        # It's a path, load it first
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        config_path = config
    else:
        config_data = config
        config_path = None
    
    # Save as YAML in the directory
    output_path = os.path.join(directory, "config.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # If original path was provided, also copy the file directly
    if config_path:
        with open(config_path, 'r') as f_in:
            with open(output_path, 'w') as f_out:
                f_out.write(f_in.read())

def get_ground_truth_path(dataset, split="val"):
    """
    Get path to ground truth data for a dataset.
    
    Args:
        dataset: Dataset name
        split: Data split (train, val, test)
        
    Returns:
        str: Path to the ground truth file
    """
    return f"data/{dataset}/data/{split}.csv"

def make_config_from_template(template, params):
    """
    Create a configuration by filling a template with parameters.
    
    Args:
        template: Template configuration dictionary
        params: Parameters to substitute in the template
        
    Returns:
        dict: Filled configuration
    """
    # Convert the template to a string to perform replacements
    config_str = yaml.dump(template)
    
    # Perform replacements
    for key, value in params.items():
        placeholder = "{" + key + "}"
        config_str = config_str.replace(placeholder, str(value))
    
    # Convert back to dictionary
    return yaml.safe_load(config_str)

def generate_experiment_configs(base_config, parameter_grid):
    """
    Generate multiple configurations from a base config and a grid of parameters.
    
    Args:
        base_config: Base configuration dictionary
        parameter_grid: Dictionary where keys are parameter names and values are lists of values
        
    Returns:
        list: List of configuration dictionaries
    """
    import itertools
    
    # Get all combinations of parameters
    keys = parameter_grid.keys()
    values = parameter_grid.values()
    configs = []
    
    for combination in itertools.product(*values):
        # Create parameter dictionary for this combination
        params = dict(zip(keys, combination))
        
        # Create a name for this configuration
        exp_name = params.get('experiment_name', '_'.join([f"{k}-{v}" for k, v in params.items()]))
        params['experiment_name'] = exp_name
        
        # Make a new config with these parameters
        config = make_config_from_template(base_config, params)
        configs.append(config)
    
    return configs

def summarize_experiment(experiment_dir):
    """
    Create a summary of experiment results.
    
    Args:
        experiment_dir: Directory containing experiment results
        
    Returns:
        dict: Summary of the experiment
    """
    # Load config
    config_path = os.path.join(experiment_dir, "config.yaml")
    # Update path from evaluation.json to results.json
    results_path = os.path.join(experiment_dir, "results.json")
    
    summary = {
        "directory": experiment_dir,
        "config": None,
        "metrics": None
    }
    
    # Load config if available
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            summary["config"] = yaml.safe_load(f)
    
    # Load results metrics if available (renamed from evaluation)
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            summary["metrics"] = json.load(f)
    
    return summary

def list_experiments(dataset):
    """
    List all experiments for a dataset.
    
    Args:
        dataset: Dataset name
        
    Returns:
        list: List of experiment directories
    """
    base_dir = f"predictions/{dataset}"
    if not os.path.exists(base_dir):
        return []
    
    # Find all subdirectories
    return [d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]

def remove_launcher_file(yaml_path):
    """
    Remove a YAML file from the launcher directory after a successful run.
    
    Args:
        yaml_path: Path to the YAML file to remove
        
    Returns:
        bool: True if file was removed successfully
    """
    try:
        if os.path.exists(yaml_path):
            os.remove(yaml_path)
            return True
        return False
    except Exception as e:
        print(f"Warning: Failed to remove launcher file {yaml_path}: {e}")
        return False

def get_incomplete_experiments(dataset):
    """
    Find experiment directories that have predictions but haven't been disambiguated.
    
    Args:
        dataset: Dataset name
        
    Returns:
        list: List of experiment directories that need disambiguation
    """
    base_dir = f"predictions/{dataset}"
    if not os.path.exists(base_dir):
        return []
    
    incomplete = []
    
    for exp_dir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, exp_dir)
        
        # Skip if not a directory
        if not os.path.isdir(full_path):
            continue
            
        # Check if predictions.csv exists
        pred_file = os.path.join(full_path, "predictions.csv")
        if not os.path.isfile(pred_file):
            continue
            
        # Check if results.json doesn't exist (renamed from evaluation.json)
        results_file = os.path.join(full_path, "results.json")
        if os.path.isfile(results_file):
            continue
            
        incomplete.append(full_path)
    
    return incomplete
