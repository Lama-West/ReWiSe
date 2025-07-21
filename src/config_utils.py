import os
import configparser
from datetime import datetime

SYSTEM_MESSAGES = {
    # Standard question-answering system message
    "qa": "Given a question, your task is to provide the list of answers without any other context. "
          "If there are multiple answers, separate them with a comma. "
          "If there are no answers, type \"None\".",
    
    # Chain-of-thought system message
    "cot": "Answer the question after a brief chain of reasoning. "
           "First, write your reasoning between <think> and </think> tags. "
           "Then, directly list the answers, separated by commas. "
           "If there's no answer, type \"None\". Be concise.",
    
    # System message for CoT generation for synthetic data
    "synthetic_cot": "Given a question and its true answer, your task is to provide a short Chain-of-Thought "
                     "reasoning with context that leads from the question to the answer. "
                     "Write it between <think> and </think> tags and then repeat the provided list of answers. "
                     "Keep it short. Don't skip lines. Always keep the same format.",
                     
    # Direct/basic system message for minimal prompting
    "direct": "Given a subject entity name and a relation name, your task is to provide the list of object entity names without any other context. "
              "If there are multiple answers, separate them with a comma. "
              "If there are no answers, type \"None\".",
}

def get_system_message(message_type):
    """
    Get the appropriate system message based on the type.
    
    Args:
        message_type: Type of system message to retrieve ('qa', 'cot', 'synthetic_cot', 'direct')
    
    Returns:
        str: The appropriate system message
    """
    if message_type in SYSTEM_MESSAGES:
        return SYSTEM_MESSAGES[message_type]
    else:
        raise ValueError(f"System message type '{message_type}' not found.")

def load_config(config_path="config.ini"):
    """
    Load configuration from config file.
    
    Args:
        config_path: Path to config file (defaults to config.ini)
        
    Returns:
        ConfigParser object with loaded configuration
    """
    config = configparser.ConfigParser()
    
    if os.path.exists(config_path):
        config.read(config_path)
    
    return config

def get_model_paths(config=None):
    """
    Get all model directory paths from the configuration.
    Priority order: 
    1. models_dir_priority (projects directory - faster access)
    2. models_dir (scratch directory - larger but slower)
    3. Other directories
    
    Args:
        config: Optional pre-loaded config
        
    Returns:
        list: List of model directory paths
    """
    if config is None:
        config = load_config()
    
    model_paths = []
    
    # First prioritize models_dir_priority (projects directory - faster access)
    if config.has_section('paths') and 'models_dir_priority' in config['paths']:
        model_paths.append(config['paths']['models_dir_priority'])
    
    # Then add main models_dir (scratch directory - larger but slower)
    if config.has_section('paths') and 'models_dir' in config['paths']:
        model_paths.append(config['paths']['models_dir'])
    
    # Add any remaining models_dirX entries (skipping the ones we've already added)
    if config.has_section('paths'):
        for key in sorted(config['paths'].keys()):
            if key.startswith('models_dir') and key != 'models_dir' and key != 'models_dir_priority':
                model_paths.append(config['paths'][key])
    
    return model_paths

def get_model_path(model_name, use_quantization=False, config=None):
    """
    Get the full path to a model, handling aliases and quantization.
    
    Args:
        model_name: Name or alias of model
        use_quantization: Whether to use quantized version if available
        config: Optional pre-loaded config
        
    Returns:
        str: Full model path or name
    """
    if config is None:
        config = load_config()
    
    # If empty config or no models section, return original name
    if not config.has_section('models') and not config.has_section('model_aliases'):
        return model_name
        
    # Check if it's an alias
    if config.has_section('model_aliases') and model_name in config['model_aliases']:
        model_name = config['model_aliases'][model_name]
    
    # # If quantized version requested and exists, use it
    # if use_quantization:
    #     quantized_name = f"{model_name}_quantized"
    #     if config.has_section('models') and quantized_name in config['models']:
    #         return config['models'][quantized_name]
    
    # Otherwise use regular model path if exists
    if config.has_section('models') and model_name in config['models']:
        return config['models'][model_name]
    
    # If not found in config, return the original model name
    return model_name

def generate_unique_experiment_id(job_id=None):
    """Generate a unique experiment ID using timestamp and microseconds, optionally with job ID."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    microseconds = now.microsecond // 1000
    base_id = f"{timestamp}_{microseconds:03d}"
    
    if job_id is not None:
        return f"{base_id}_{job_id}"
    return base_id

def create_unique_path(base_dir, prefix="", suffix="", job_id=None):
    """Create a unique directory path that doesn't exist yet."""
    unique_id = generate_unique_experiment_id(job_id)
    if prefix:
        unique_id = f"{prefix}_{unique_id}"
    if suffix:
        unique_id = f"{unique_id}_{suffix}"
    
    path = os.path.join(base_dir, unique_id)
    
    # Ensure uniqueness by incrementing if path exists
    counter = 1
    original_path = path
    while os.path.exists(path):
        path = f"{original_path}_{counter}"
        counter += 1
    
    return path, os.path.basename(path)

def create_unique_config_path(config_dir, prefix="", suffix="", job_id=None):
    """Create a unique config file path."""
    unique_id = generate_unique_experiment_id(job_id)
    if prefix:
        unique_id = f"{prefix}_{unique_id}"
    if suffix:
        unique_id = f"{unique_id}_{suffix}"
    
    filename = f"{unique_id}.yaml"
    path = os.path.join(config_dir, filename)
    
    # Ensure uniqueness
    counter = 1
    original_path = path
    while os.path.exists(path):
        path = f"{original_path[:-5]}_{counter}.yaml"
        counter += 1
    
    return path, unique_id

