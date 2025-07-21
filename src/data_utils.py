import pandas as pd
import os

def load_dataset(dataset_name='dataset2024', prompt_file=None, split='val'):
    """
    Loads train and test/val datasets and question prompts.
    
    Args:
        dataset_name: Name of the dataset to load
        prompt_file: Optional custom prompt template file to use (defaults to question_prompts.csv)
        split: Data split to load as evaluation set ('val' or 'test')
    
    Returns:
        train_df, eval_df, prompt_templates: DataFrames with train data, evaluation data, and prompt templates
    """
    data_dir = f'data/{dataset_name}/'
    train_data_path = f'{data_dir}data/train.csv'
    eval_data_path = f'{data_dir}data/{split}.csv'
    
    # Use custom prompt file if provided, otherwise default to question_prompts.csv
    prompt_file = prompt_file or 'question_prompts.csv'
    if not prompt_file.startswith(data_dir):
        prompt_file_path = f'{data_dir}{prompt_file}'
    else:
        prompt_file_path = prompt_file
    
    # Check if prompt file exists
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
    
    # Check if evaluation data file exists
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(f"Evaluation data file not found: {eval_data_path}")
    
    train_df = pd.read_csv(train_data_path)
    eval_df = pd.read_csv(eval_data_path)
    prompt_templates = pd.read_csv(prompt_file_path)
    
    return train_df, eval_df, prompt_templates