import torch
from typing import List, Dict, Union, Optional, Any
import os
import sys

from prompt_utils import format_chat_messages, prompt_few_shot, prompt_cot
from config_utils import get_system_message

def is_vllm_available():
    """
    Check if vLLM is installed and functional.
    This attempts to import a key vLLM class to verify the package works.
    """
    try:
        # Try to import a core vLLM class
        from vllm.sampling_params import SamplingParams
        return True
    except (ImportError, ModuleNotFoundError):
        return False
    except Exception:
        # Other exceptions might indicate vLLM is installed but can't run
        return False

def is_instruct_model(model_name_or_path):
    """
    Check if the model is an instruction model based on the actual model path.
    Looks for 'instruct' or known instruct model patterns (e.g., Qwen2 9b) in the resolved path rather than relying on aliases.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.config_utils import get_model_path, load_config
    
    # For vLLM models, extract the actual model path
    if hasattr(model_name_or_path, "model"):
        model_path = str(model_name_or_path.model)
    else:
        model_path = str(model_name_or_path)
    
    # If it's an alias, resolve it to the actual path using config
    config = load_config()
    try:
        resolved_path = get_model_path(model_path, False, config)
        model_path = resolved_path
    except:
        # Keep using original path if resolution fails
        pass
    
    model_basename = os.path.basename(model_path)
    model_name_lower = model_basename.lower()
    
    # Recognize instruct models by common patterns
    if 'instruct' in model_name_lower:
        return True
    # Recognize other instruct models (Qwen3-8B only for now)
    if model_name_lower == "qwen3-8b":
        return True
    return False

def prepare_prompts(
    model_name_or_path: str,
    subjects: List[str],
    relations: List[str],
    question_templates: List[str],
    train_df: Any,
    question_prompts: Any,
    few_shot: int = 0,
    few_shot_other: int = 0,
    use_cot: bool = False,
    system_message: Optional[str] = None,
) -> List[Union[str, List[Dict[str, str]]]]:
    """
    Prepare prompts based on model type (instruct vs non-instruct).
    
    Args:
        model_name_or_path: Model name or path (or vLLM model object)
        subjects: List of subject entities
        relations: List of relations
        question_templates: List of question templates
        train_df: Training dataframe for few-shot examples
        question_prompts: DataFrame with question templates for all relations
        few_shot: Number of examples from the target relation
        few_shot_other: Number of examples from other relations
        use_cot: Whether to use chain-of-thought prompting
        system_message: System message to include
        
    Returns:
        List of prompts (either strings or structured chat message lists)
    """
    if system_message is None:
        system_message = get_system_message("cot" if use_cot else "qa")
        
    prompts = []
    is_instruct = is_instruct_model(model_name_or_path)
    
    for subject, relation, question_template in zip(subjects, relations, question_templates):
        if is_instruct:
            # For instruct models, create structured chat messages
            prompt = format_chat_messages(
                subject, relation, question_template, train_df, question_prompts,
                few_shot=few_shot, few_shot_other=few_shot_other, 
                use_cot=use_cot, system_message=system_message
            )
        else:
            # For non-instruct models, use traditional format with system message
            prompt = prompt_few_shot(
                subject, relation, question_template, train_df, question_prompts,
                few_shot=few_shot, few_shot_other=few_shot_other, 
                use_cot=use_cot, system_message=system_message
            )
            if use_cot:
                prompt = prompt_cot(prompt)
                
        prompts.append(prompt)
        
    return prompts

def generate_with_vllm(
    model,
    prompts: List[Union[str, List[Dict[str, str]]]],
    tokenizer=None,
    generation_params: Optional[Dict[str, Any]] = None,
    enable_thinking = None
) -> List[str]:
    """
    Generate text using vLLM backend.
    
    Args:
        model: vLLM model
        prompts: List of prompts (strings or chat message lists)
        tokenizer: Tokenizer for applying chat templates
        generation_params: Parameters for generation
        
    Returns:
        List of generated texts
    """
    from vllm import SamplingParams
    
    default_params = {
        "max_tokens": 100,
        "temperature": 1.,
    }
    
    # Update with provided params
    if generation_params:
        default_params.update(generation_params)
        
    # Convert to SamplingParams
    sampling_params = SamplingParams(**default_params)
    
    formatted_prompts = []
    for prompt in prompts:
        if isinstance(prompt, list):  # Structured chat message
            if tokenizer is not None and enable_thinking is not None:
                formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            else:
                formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted_prompt)
        else:
            formatted_prompts.append(prompt)
    
    # Generate outputs
    outputs = model.generate(formatted_prompts, sampling_params)
    
    # Extract generated texts
    outputs = [output.outputs[0].text.strip() for output in outputs]
    
    return outputs

def generate_with_hf(
    model,
    tokenizer,
    prompts: List[Union[str, List[Dict[str, str]]]],
    generation_params: Optional[Dict[str, Any]] = None,
    batch_size: int = 8,
    enable_thinking = None
) -> List[str]:
    """
    Generate text using HuggingFace transformers backend.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of prompts (strings or chat message lists)
        generation_params: Parameters for generation
        batch_size: Size of batches for processing
        
    Returns:
        List of generated texts
    """
    from tqdm import tqdm
    
    default_params = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 1.,
    }
    
    # Update with provided params
    if generation_params:
        # Map max_tokens to max_new_tokens if provided
        if "max_tokens" in generation_params:
            generation_params["max_new_tokens"] = generation_params.pop("max_tokens")
            
        default_params.update(generation_params)
    # If temperature is 0, set do_sample to False for deterministic output
    if default_params.get("temperature", 1.) == 0:
        default_params["do_sample"] = False
    
    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, 'config'):
            model.config.pad_token_id = tokenizer.eos_token_id
    
    formatted_prompts = []
    for prompt in prompts:
        if isinstance(prompt, list):  # It's a structured chat message
            if enable_thinking is not None:
                formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            else:
                formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted_prompt)
        else:
            formatted_prompts.append(prompt)
    
    all_generated_texts = []
    
    # Process in batches to avoid memory issues
    print(f"Generating text for {len(formatted_prompts)} prompts in batches of {batch_size}...")
    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating"):
        batch_prompts = formatted_prompts[i:i+batch_size]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=False,
        )
        
        # Store the input lengths for later extraction
        input_lengths = [len(ids) for ids in inputs.input_ids]
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **default_params
            )
        
        # Extract only the new tokens by slicing the output tensors
        batch_generated_texts = []
        for j, output_ids in enumerate(outputs):
            # Get only the newly generated tokens (exclude input)
            new_tokens = output_ids[input_lengths[j]:]
            
            # Remove padding tokens if any
            if tokenizer.pad_token_id in new_tokens:
                new_tokens = new_tokens[:new_tokens.tolist().index(tokenizer.pad_token_id)]
            
            # Decode only the new tokens
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            batch_generated_texts.append(generated_text)
        
        all_generated_texts.extend(batch_generated_texts)
    
    return all_generated_texts

def detect_model_backend(model):
    """
    Detect whether a model is vLLM, HuggingFace, or PEFT.
    
    Args:
        model: The model to check
        
    Returns:
        str: 'vllm', 'hf', or 'peft'
    """
    # Check by class name first (most reliable)
    model_class = model.__class__.__name__
    
    if 'LLM' in model_class and 'vllm' in str(model.__class__):
        return 'vllm'
    elif 'PeftModel' in model_class:
        return 'peft'
    elif hasattr(model, 'peft_config'):
        return 'peft'
    elif 'PreTrainedModel' in str(model.__class__.mro()):
        return 'hf'
    
    # Fallback to attribute-based checks
    if hasattr(model, 'llm_engine'):
        return 'vllm'
    
    # Default to HF if we couldn't identify definitely
    return 'hf'

def generate_text(
    model,
    prompts: List[Union[str, List[Dict[str, str]]]],
    tokenizer=None,
    backend: str = "auto",
    generation_params: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    debug_samples: int = 2,
    batch_size: int = 8,
    enable_thinking = False
) -> List[str]:
    """
    Core text generation function for different model backends.
    
    Prompts must be pre-built.
    
    Args:
        model: The model to use for generation
        prompts: Pre-built prompts (strings for standard models, message lists for chat models)
        tokenizer: Tokenizer for the model (required for HF backend)
        backend: "auto", "vllm", or "hf"
        generation_params: Parameters for generation
        debug: Whether to print debug information
        debug_samples: Number of samples to show in debug mode. Only used if debug
        batch_size: Batch size for HuggingFace processing (ignored for vLLM)
        
    Returns:
        List of generated texts
    """
    # Auto-detect backend if requested
    if backend == "auto":
        vllm_available = is_vllm_available()
        
        detected_backend = detect_model_backend(model)
        
        if detected_backend == 'peft' or detected_backend == 'hf':
            backend = 'hf'
        elif detected_backend == 'vllm' and vllm_available:
            backend = 'vllm'
        else:
            backend = 'hf'
    
    # Validate inputs
    if backend == "hf" and tokenizer is None:
        raise ValueError("Tokenizer is required for HuggingFace backend")
    
    print(f"Using {backend.upper()} backend for text generation")
    
    # Prepare generation parameters for different backends
    hf_params = {}
    vllm_params = {}
    
    if generation_params:
        # Extract core parameters with consistent naming
        max_tokens = generation_params.get("max_tokens", 100)
        
        # Set backend-specific parameters
        hf_params["max_new_tokens"] = max_tokens
        vllm_params["max_tokens"] = max_tokens
        vllm_params["stop"] = generation_params.get("stop", [])
    
    # Generate text with appropriate backend
    if backend == "vllm":
        outputs = generate_with_vllm(model, prompts, tokenizer, vllm_params, enable_thinking=enable_thinking)
    else:  # hf
        outputs = generate_with_hf(model, tokenizer, prompts, hf_params, batch_size, enable_thinking=enable_thinking)
    
    if debug and prompts:
        print("\n===== DEBUG: COMPLETE PROMPT-RESPONSE EXAMPLES =====")
        for i in range(min(debug_samples, len(prompts))):
            print(f"\n\nEXAMPLE {i+1}/{debug_samples}")
            print("=" * 80)
            
            if isinstance(prompts[i], list):  # Chat format
                print("FORMAT: Chat Messages")
                print("-" * 80)
                
                # Print all messages in the prompt
                for msg in prompts[i]:
                    print(f"[{msg['role'].upper()}]: {msg['content']}")
                    print("-" * 40)
                
                # Print ONLY the generated response (not including the prompt)
                print(f"[ASSISTANT (GENERATED)]: {outputs[i]}")

            else:  # Text format
                print("FORMAT: Text prompt")
                print("-" * 80)
                print(prompts[i])
                print("\n" + "-" * 40 + " GENERATED " + "-" * 40)
                print(outputs[i])
            
            print("=" * 80)
    
    return outputs
