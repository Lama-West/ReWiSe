import os
import time
import random
from datetime import datetime
from pathlib import Path

def generate_unique_id():
    """Generate a unique identifier using timestamp + microseconds + random component."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    microseconds = now.strftime("%f")[:3]
    random_component = random.randint(100, 999)
    
    return f"{timestamp}_{microseconds}_{random_component}"

def create_unique_path(base_dir, prefix="", suffix="", is_file=False, max_attempts=100):
    """
    Create a unique path that doesn't exist yet.
    
    Args:
        base_dir: Base directory path
        prefix: Optional prefix for the name
        suffix: Optional suffix for the name (e.g., '.yaml', '.csv')
        is_file: Whether this is a file path (True) or directory path (False)
        max_attempts: Maximum attempts to find unique path
        
    Returns:
        str: Unique path that doesn't exist
        
    Raises:
        RuntimeError: If unable to create unique path after max_attempts
    """
    os.makedirs(base_dir, exist_ok=True)
    
    for attempt in range(max_attempts):
        unique_id = generate_unique_id()
        name = f"{prefix}{unique_id}{suffix}" if prefix else f"{unique_id}{suffix}"
        path = os.path.join(base_dir, name)
        
        if not os.path.exists(path):
            if is_file:
                Path(path).touch()
            else:
                os.makedirs(path, exist_ok=True)
            return path
        
        time.sleep(0.001)
    
    raise RuntimeError(f"Unable to create unique path after {max_attempts} attempts")

def create_unique_config_path(config_dir="configs"):
    """Create unique config file path."""
    return create_unique_path(config_dir, suffix=".yaml", is_file=True)

def create_unique_experiment_dir(dataset_name, base_dir="predictions"):
    """Create unique experiment directory."""
    experiment_base = os.path.join(base_dir, dataset_name)
    return create_unique_path(experiment_base, is_file=False)

def create_unique_trained_model_dir(base_dir="trained_models", prefix=""):
    """Create unique trained model directory."""
    return create_unique_path(base_dir, prefix=prefix, is_file=False)

def extract_timestamp_from_path(path):
    """Extract just the timestamp part from a path for compatibility."""
    basename = os.path.basename(path)
    name_without_ext = os.path.splitext(basename)[0]
    
    if "_" in name_without_ext:
        timestamp_part = name_without_ext.split("_")[0] + "_" + name_without_ext.split("_")[1]
        return timestamp_part
    
    return name_without_ext
