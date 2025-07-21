"""
Models module for handling language model loading, generation, and processing.
"""

from .generation import (
    generate_text, 
    prepare_prompts, 
    is_instruct_model,
    is_vllm_available,
    detect_model_backend,
)

from .loading import (
    load_model,
    load_peft_model,
    load_vllm_model,
)

from .processing import (
    format_for_display,
)

__all__ = [
    'generate_text',
    'prepare_prompts',
    'is_instruct_model',
    'is_vllm_available',
    'detect_model_backend',
    'load_model',
    'load_peft_model',
    'load_vllm_model',
    'format_for_display',
]
