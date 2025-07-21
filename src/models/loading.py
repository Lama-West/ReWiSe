import os
import json
import torch
import sys
import shutil

# Add parent directory to path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_utils import get_model_path, get_model_paths, load_config

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def get_resolved_model_path(model_name, use_quantization=False, slurm_tmpdir=None):
    """
    Resolve the model path checking multiple directories.
    
    Args:
        model_name: Name or path of the model 
        use_quantization: Whether to use quantization during loading (doesn't affect path)
        slurm_tmpdir: Optional SLURM_TMPDIR to check first
        
    Returns:
        str: The resolved model path
    """
    config = load_config()
    local_files_only = not torch.backends.mps.is_available()
    
    # Get the original model name without quantization suffix
    # We'll always use the base model path, quantization is applied at loading time
    resolved_model_name = get_model_path(model_name, False, config)
    
    # If we don't need local files, return huggingface name directly
    if not local_files_only:
        return resolved_model_name
    
    # Get all model directories to search
    search_paths = []
    
    # Check SLURM_TMPDIR first if provided
    if slurm_tmpdir:
        search_paths.append(slurm_tmpdir)
    
    # Add configured model directories
    search_paths.extend(get_model_paths(config))
    
    # Check if model exists in any of these directories
    for base_dir in search_paths:
        # Check both options: full model name and just the base name
        model_options = [
            os.path.join(base_dir, resolved_model_name),
            os.path.join(base_dir, resolved_model_name.split('/')[-1])
        ]
        
        for model_path in model_options:
            if os.path.exists(model_path):
                return model_path
    
    # If not found locally, return the huggingface model name
    return resolved_model_name

def copy_model_to_tmpdir(model_name, use_quantization=False):
    """
    Copy a model to SLURM_TMPDIR if available.
    
    Args:
        model_name: Name or path of the model to copy
        use_quantization: Whether we'll use quantization during loading (doesn't affect copy path)
        
    Returns:
        str: Path to the model in SLURM_TMPDIR, or original path if copy failed
    """
    # Check if SLURM_TMPDIR exists
    slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
    if not slurm_tmpdir:
        print("SLURM_TMPDIR not available, skipping model copy")
        return get_resolved_model_path(model_name, use_quantization)
    
    # Get resolved model path (always the base model, not a pre-quantized version)
    config = load_config()
    resolved_model_name = get_model_path(model_name, False, config)
    model_basename = resolved_model_name.split('/')[-1]
    
    # Find the original model path, checking directories in priority order
    original_path = None
    
    # Get model paths in priority order
    model_paths = get_model_paths(config)
    
    # Check if we have a priority directory configured and move it to the front
    priority_dir = None
    for key, value in config.items('paths'):
        if key == 'models_dir_priority' and value:
            priority_dir = value
            break
    
    # If priority directory exists, check it first
    if priority_dir:
        priority_path = os.path.join(priority_dir, model_basename)
        if os.path.exists(priority_path):
            print(f"Found model in priority directory (faster storage): {priority_path}")
            original_path = priority_path
    
    # If not found in priority directory, check the other directories
    if original_path is None:
        for base_dir in model_paths:
            test_path = os.path.join(base_dir, model_basename)
            if os.path.exists(test_path):
                original_path = test_path
                print(f"Found model in: {original_path}")
                break
    
    if not original_path:
        print(f"Model {model_name} not found locally, skipping copy")
        return get_resolved_model_path(model_name, use_quantization)
    
    # Prepare destination path
    dest_path = os.path.join(slurm_tmpdir, model_basename)
    if os.path.exists(dest_path):
        print(f"Model already exists in SLURM_TMPDIR: {dest_path}")
        return dest_path
    
    # Copy the model
    try:
        print(f"Copying model from {original_path} to {dest_path}...")
        shutil.copytree(original_path, dest_path)
        print(f"Model copied successfully to SLURM_TMPDIR")
        return dest_path
    except Exception as e:
        print(f"Error copying model to SLURM_TMPDIR: {e}")
        return get_resolved_model_path(model_name, use_quantization)

def load_model(model_name, use_quantization=False, use_slurm_tmpdir=False, force_hf=False):
    """
    Load a model and tokenizer with appropriate configuration.
    
    Args:
        model_name: Name or path of the model to load
        use_quantization: Whether to use 4-bit quantization (when available)
        use_slurm_tmpdir: Whether to copy the model to SLURM_TMPDIR first
        force_hf: Whether to force using Hugging Face implementation (needed for PEFT)
        
    Returns:
        tuple: (model, tokenizer)
    """
    if use_slurm_tmpdir and os.environ.get('SLURM_TMPDIR'):
        model_path = copy_model_to_tmpdir(model_name, use_quantization)
    else:
        model_path = get_resolved_model_path(model_name, use_quantization)
    
    # Check if we need to force using local files
    is_local_path = os.path.exists(model_path) and os.path.isdir(model_path) 
    local_files_only = is_local_path or not torch.backends.mps.is_available() or force_hf
    
    # Model
    print(f"Loading from {model_path}")
    if use_quantization and local_files_only:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            quantization_config=bnb_config, 
            torch_dtype=torch.float16, 
            local_files_only=local_files_only, 
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            local_files_only=local_files_only, 
            device_map="auto"
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=local_files_only)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

def load_peft_model(base_model, model_path, epoch=None):
    """
    Load a PEFT adapter onto a base model.
    Will throw an error if adapter cannot be loaded.
    
    Args:
        base_model: Base model to adapt
        model_path: Path to the PEFT model directory
        epoch: Optional epoch number to load, None for final model
        
    Returns:
        PeftModel: The adapted model
    """
    # If epoch is specified, look for the epoch directory
    if epoch is not None:
        epoch_dir = os.path.join(model_path, f"epoch{epoch}")
        if os.path.exists(epoch_dir):
            model_path = epoch_dir
            print(f"Loading model from epoch {epoch} checkpoint")
        else:
            print(f"Warning: Epoch {epoch} directory not found, falling back to default")
    else:
        # If no epoch specified, try to determine final epoch from relation_config.json
        relation_config_path = os.path.join(model_path, "relation_config.json")
        if os.path.exists(relation_config_path):
            try:
                with open(relation_config_path, 'r') as f:
                    relation_config = json.load(f)
                    final_epoch = relation_config.get("final_epoch")
                    if final_epoch:
                        epoch_dir = os.path.join(model_path, f"epoch{final_epoch}")
                        if os.path.exists(epoch_dir):
                            model_path = epoch_dir
                            print(f"Loading final model (epoch {final_epoch})")
            except Exception as e:
                print(f"Error reading relation_config.json: {e}")
    
    # Check for both possible configuration file names
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    peft_config_path = os.path.join(model_path, "peft_config.json")
    
    config_path = None
    if os.path.exists(adapter_config_path):
        config_path = adapter_config_path
    elif os.path.exists(peft_config_path):
        config_path = peft_config_path
    
    if not config_path:
        raise FileNotFoundError(
            f"No PEFT configuration found at {model_path}. "
            f"Checked for: adapter_config.json, peft_config.json"
        )
    
    try:
        # Try to detect PEFT method from config file
        peft_method = "unknown"
        try:
            with open(config_path, 'r') as f:
                peft_config = json.load(f)
                peft_type = peft_config.get("peft_type")
                if peft_type == "PROMPT_TUNING":
                    peft_method = "prompt_tuning"
                elif peft_type == "LORA":
                    peft_method = "lora"
                    # Print LoRA parameters
                    r = peft_config.get("r", "unknown")
                    lora_alpha = peft_config.get("lora_alpha", "unknown")
                    print(f"Loading LoRA adapter with r={r}, alpha={lora_alpha}")
        except Exception as e:
            print(f"Could not determine PEFT method from config: {e}")
        
        print(f"Loading PEFT model ({peft_method}) from {model_path} using configuration: {os.path.basename(config_path)}")
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        return peft_model
    except Exception as e:
        raise RuntimeError(f"Error loading PEFT model: {e}")

def load_vllm_model(model_name, quantized=False, tensor_parallel_size=None):
    """
    Load model using vLLM if available, otherwise fall back to standard loading.
    
    Args:
        model_name: Name or path of the model to load
        quantized: Whether to use quantization with bitsandbytes
        tensor_parallel_size: Number of GPUs to use for tensor parallelism (None=auto)
        
    Returns:
        tuple: (model, tokenizer)
    """
    from vllm import LLM

    # Look at available GPUs if tensor_parallel_size not specified
    if tensor_parallel_size is None:
        import torch
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            # Count GPU IDs
            tensor_parallel_size = len(visible_devices.split(","))
        else:
            tensor_parallel_size = torch.cuda.device_count()
    
    print(f"Loading {model_name} with vLLM..." + 
          (f" (with quantization, using {tensor_parallel_size} GPU(s))" if quantized else 
           f" (using {tensor_parallel_size} GPU(s))"))
    
    # Always resolve the model path using the robust function
    model_path = get_resolved_model_path(model_name, quantized)
    print(f"Resolved model path: {model_path}")
    
    # Tokenizer can be needed for chat template formatting
    local_files_only = not torch.backends.mps.is_available()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        padding_side="left", 
        local_files_only=local_files_only
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Base parameters for LLM
    llm_params = {
        "model": model_path,
        "tokenizer": model_path,  # Use same path for tokenizer
        "max_model_len": 2**15,
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
    }
    
    # Add quantization parameters only if requested
    if quantized:
        llm_params.update({
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes"
        })
    
    # Print the actual model path being used for clarity
    print(f"Initializing vLLM with model path: {model_path}")
    print(f"Using tensor parallel size: {tensor_parallel_size}")
    
    llm = LLM(**llm_params)
    
    return llm, tokenizer
