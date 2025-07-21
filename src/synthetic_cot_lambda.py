import pandas as pd
import argparse
from tqdm import tqdm
import os
import time
import requests
import json
from openai import OpenAI

from prompt_utils import format_answer
from config_utils import get_system_message

def _get_examples(row, cot_examples, question_prompts, few_shot, all_relations):
    """
    Helper function to gather few-shot examples with unique subject entity probability normalization.
    
    Args:
        row: Data row with relation info
        cot_examples: DataFrame with examples
        question_prompts: DataFrame with question templates
        few_shot: Number of examples to include
        all_relations: Whether to include examples from other relations if needed
    
    Returns:
        pd.DataFrame: Selected examples
    """
    relation = row["Relation"]
    
    # Get examples for this relation with unique subject entities
    relation_examples = cot_examples[cot_examples["Relation"] == relation]
    unique_subjects = relation_examples['SubjectEntity'].unique()
    relation_few_shot = min(few_shot, len(unique_subjects))
    
    # Sample examples with unique subject entities
    if relation_few_shot > 0:
        selected_subjects = pd.Series(unique_subjects).sample(relation_few_shot)
        relation_selected = relation_examples[relation_examples['SubjectEntity'].isin(selected_subjects)].groupby('SubjectEntity').sample(1)
    else:
        relation_selected = pd.DataFrame()
    
    # Get examples from other relations if needed
    if all_relations and relation_few_shot < few_shot:
        remaining = few_shot - relation_few_shot
        other_examples = cot_examples[cot_examples["Relation"] != relation]
        
        # Ensure unique subject entities for other relations too
        other_unique_subjects = other_examples['SubjectEntity'].unique()
        other_few_shot = min(remaining, len(other_unique_subjects))
        
        if other_few_shot > 0:
            selected_other_subjects = pd.Series(other_unique_subjects).sample(other_few_shot)
            other_selected = other_examples[other_examples['SubjectEntity'].isin(selected_other_subjects)].groupby('SubjectEntity').sample(1)
            examples = pd.concat([relation_selected, other_selected])
        else:
            examples = relation_selected
    else:
        examples = relation_selected
    
    # Shuffle examples
    return examples.sample(frac=1.0) if not examples.empty else examples

def build_synthetic_cot_prompt_help(row, cot_examples, question_prompts, few_shot=5, is_instruct=False, all_relations=False):
    """
    Build prompt for "help" mode synthetic CoT generation, where the model is given the answer.
    
    Args:
        row: Data row with subject, relation, and object entities
        cot_examples: DataFrame with human-written CoT examples
        question_prompts: DataFrame with question templates
        few_shot: Number of few-shot examples to include
        is_instruct: Whether to use instruct-model formatting
        all_relations: If True, include examples from other relations if needed
    
    Returns:
        str or list: Formatted prompt
    """
    relation = row["Relation"]
    subject = row["SubjectEntity"]
    question_template = question_prompts[question_prompts["Relation"] == relation].iloc[0]["PromptTemplate"]
    formatted_answer = format_answer(eval(row["ObjectEntities"]))
    
    # Get examples
    examples = _get_examples(row, cot_examples, question_prompts, few_shot, all_relations)
    
    # Get synthetic CoT system message
    system_message = get_system_message("synthetic_cot")
    
    if is_instruct:
        # Format for instruct models using chat template
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add examples as chat turns
        for _, ex in examples.iterrows():
            ex_relation = ex["Relation"]
            ex_question_template = question_prompts[question_prompts["Relation"] == ex_relation].iloc[0]["PromptTemplate"]
            ex_question = ex_question_template.format(subject_entity=ex["SubjectEntity"])
            ex_answer = format_answer(eval(ex["ObjectEntities"]))
            ex_cot = ex["CoT"]
            
            # Format with the answer provided
            example_text = (
                f"{ex_question}\n"
                f"The true answer is {ex_answer}."
            )
            
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": ex_cot})
        
        # Add the target question with the answer provided
        target_text = (
            f"{question_template.format(subject_entity=subject)}\n"
            f"The true answer is {formatted_answer}."
        )
        messages.append({"role": "user", "content": target_text})
        
        return messages
    else:
        # Format for standard models (non-instruct)
        prompt_parts = [system_message]
        
        # Add examples
        for _, ex in examples.iterrows():
            ex_relation = ex["Relation"]
            ex_question_template = question_prompts[question_prompts["Relation"] == ex_relation].iloc[0]["PromptTemplate"]
            ex_question = ex_question_template.format(subject_entity=ex["SubjectEntity"])
            ex_answer = format_answer(eval(ex["ObjectEntities"]))
            ex_cot = ex["CoT"]
            
            example_text = (
                f"{ex_question}\n"
                f"The true answer is {ex_answer}.\n"
                f"{ex_cot}"
            )
            prompt_parts.append(example_text)
        
        # Add target question with the answer provided
        target_text = (
            f"{question_template.format(subject_entity=subject)}\n"
            f"The true answer is {formatted_answer}."
        )
        prompt_parts.append(target_text)
        
        return "\n\n".join(prompt_parts)

def build_synthetic_cot_prompt_nohelp(row, cot_examples, question_prompts, few_shot=5, is_instruct=False, all_relations=False):
    """
    Build prompt for "nohelp" mode synthetic CoT generation, where the model must generate both reasoning and answer.
    
    Args:
        row: Data row with subject, relation, and object entities
        cot_examples: DataFrame with human-written CoT examples
        question_prompts: DataFrame with question templates
        few_shot: Number of few-shot examples to include
        is_instruct: Whether to use instruct-model formatting
        all_relations: If True, include examples from other relations if needed
    
    Returns:
        str or list: Formatted prompt
    """
    relation = row["Relation"]
    subject = row["SubjectEntity"]
    question_template = question_prompts[question_prompts["Relation"] == relation].iloc[0]["PromptTemplate"]
    
    # Get examples
    examples = _get_examples(row, cot_examples, question_prompts, few_shot, all_relations)
    
    # Get standard CoT system message
    system_message = get_system_message("cot")
    
    if is_instruct:
        # Format for instruct models using chat template
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add examples as chat turns
        for _, ex in examples.iterrows():
            ex_relation = ex["Relation"]
            ex_question_template = question_prompts[question_prompts["Relation"] == ex_relation].iloc[0]["PromptTemplate"]
            ex_question = ex_question_template.format(subject_entity=ex["SubjectEntity"])
            ex_cot = ex["CoT"]
            
            # Only provide the question, not the answer
            messages.append({"role": "user", "content": ex_question})
            messages.append({"role": "assistant", "content": ex_cot})
        
        # Add the target question without the answer
        messages.append({"role": "user", "content": question_template.format(subject_entity=subject)})
        
        return messages
    else:
        # Format for standard models (non-instruct)
        prompt_parts = [system_message]
        
        # Add examples
        for _, ex in examples.iterrows():
            ex_relation = ex["Relation"]
            ex_question_template = question_prompts[question_prompts["Relation"] == ex_relation].iloc[0]["PromptTemplate"]
            ex_question = ex_question_template.format(subject_entity=ex["SubjectEntity"])
            ex_cot = ex["CoT"]
            
            example_text = f"Q: {ex_question}\n{ex_cot}"
            prompt_parts.append(example_text)
        
        # Add target question
        target_text = f"Q: {question_template.format(subject_entity=subject)}"
        prompt_parts.append(target_text)
        
        return "\n\n".join(prompt_parts)

def build_synthetic_cot_prompt(row, cot_examples, question_prompts, few_shot=5, is_instruct=False, all_relations=False, mode=None):
    """
    Build prompt specifically for synthetic CoT generation.
    
    Args:
        row: Data row with subject, relation, and object entities
        cot_examples: DataFrame with human-written CoT examples
        question_prompts: DataFrame with question templates
        few_shot: Number of few-shot examples to include
        is_instruct: Whether to use instruct-model formatting
        all_relations: If True, include examples from other relations if needed
        mode: Prompt mode - "help" (provide answer) or "nohelp" (model must generate answer)
    
    Returns:
        str or list: Formatted prompt
    """
     
    if mode == "help":
        return build_synthetic_cot_prompt_help(row, cot_examples, question_prompts, few_shot, is_instruct, all_relations)
    elif mode == "nohelp":
        return build_synthetic_cot_prompt_nohelp(row, cot_examples, question_prompts, few_shot, is_instruct, all_relations)
    else:
        raise ValueError(f"Unknown prompt mode: {mode}. Use 'help' or 'nohelp'.")

def generate_text_with_lambda(prompts, model_name, temperature=1.0, max_tokens=500):
    """
    Generate text using Lambda API.
    
    Args:
        prompts: List of prompts (structured as chat messages)
        model_name: Model name for Lambda API
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        list: Generated texts
    """
    # Load API key from file
    try:
        with open("ideas/lambda_api", "r") as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("Lambda API key file 'lambda_api' not found. Please create this file with your API key.")
    
    # Initialize OpenAI client with Lambda endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.lambdalabs.com/v1"
    )
    
    outputs = []
    
    for i, prompt in tqdm(enumerate(prompts), desc="Lambda API calls", total=len(prompts)):
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                # Create chat completion
                response = client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # stop=["\n\n"] # For Qwen, we need this comment as the model skips lines
                )
                
                # Extract generated text
                text = response.choices[0].message.content
                outputs.append(text)
                
                # Add delay to avoid rate limiting
                time.sleep(0.5)
                break
            except Exception as e:
                retry_count += 1
                print(f"Error on prompt {i}, attempt {retry_count}: {e}")
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} attempts, using empty string")
                    outputs.append("")
                else:
                    # Exponential backoff
                    time.sleep(2 ** retry_count)
    
    return outputs

def generate_cots(args):
    """
    Generate CoT explanations for the training dataset using Lambda API.
    """
    
    # Load datasets
    if args.data_split == "train":
        data_df = pd.read_csv(args.train_data)
    elif args.data_split == "val":
        data_df = pd.read_csv(args.val_data)
    else:
        raise ValueError(f"Unknown data_split: {args.data_split}. Use 'train' or 'val'.")
        
    human_cot_df = pd.read_csv(args.human_cot)
    question_prompts = pd.read_csv(args.question_prompts)
    
    print(f"Loaded {len(data_df)} {args.data_split} examples")
    print(f"Loaded {len(human_cot_df)} human CoT examples")
    
    # Sample examples based on strategy
    if args.full_dataset_repeats > 0:
        # Use the full dataset repeated N times strategy
        print(f"Using full dataset sampling strategy with {args.full_dataset_repeats} repeats")
        # Create N copies of the dataset
        repeated_dfs = [data_df] * args.full_dataset_repeats
        sampled_df = pd.concat(repeated_dfs).reset_index(drop=True)
        
        # Shuffle the data
        sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)
        
        # Apply limit if specified (limit takes precedence over repeats)
        if args.limit > 0 and args.limit < len(sampled_df):
            print(f"Limiting to {args.limit} examples")
            sampled_df = sampled_df.head(args.limit)
        
        print(f"Using {len(sampled_df)} examples for CoT generation")
        data_df = sampled_df
    elif args.limit > 0:
        # Legacy random sampling with replacement; kept for debugging
        print(f"Using random sampling with limit of {args.limit} examples")
        data_df = data_df.sample(args.limit, replace=True)
    
    # Lambda API models are always instruct models
    is_instruct = True
    print(f"Using Lambda API with model: {args.model}")
    print(f"Using temperature: {args.temperature}")
    print(f"Using prompt mode: {args.mode}")
    
    # Generate prompts
    print("Building prompts...")
    all_prompts = []
    for i, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Building prompts"):
        prompt = build_synthetic_cot_prompt(
            row=row,
            cot_examples=human_cot_df,
            question_prompts=question_prompts,
            few_shot=args.few_shot,
            is_instruct=is_instruct,  # Always true for Lambda API models
            all_relations=args.all_relations,
            mode=args.mode
        )
        all_prompts.append(prompt)
    
    # Generate CoTs using Lambda API
    print("Generating CoTs with Lambda API...")
    generations = generate_text_with_lambda(
        prompts=all_prompts,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Add CoTs to the dataframe
    data_df["CoT"] = generations
    
    # If append is True and the file exists, merge with existing data
    if args.append and os.path.exists(args.output):
        print(f"Appending to existing file: {args.output}")
        existing_df = pd.read_csv(args.output)
        data_df = pd.concat([existing_df, data_df], ignore_index=True)
    
    # Save results
    print(f"Saving results to {args.output}")
    data_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Chain-of-Thought explanations using Lambda API")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="dataset2024", help="Dataset name (e.g., dataset2024, dataset2025)")
    
    # Data files
    parser.add_argument("--train-data", type=str, default=None, help="Path to training data CSV")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation data CSV")
    parser.add_argument("--data-split", type=str, choices=["train", "val"], default="train", help="Data split to use")
    parser.add_argument("--human-cot", type=str, default=None, help="Path to human CoT examples CSV")
    parser.add_argument("--question-prompts", type=str, default=None, help="Path to question templates CSV")
    parser.add_argument("--output", type=str, default=None, help="Output path for synthetic CoT data")
    
    # Model settings
    parser.add_argument("--model", type=str, default="llama3.3-70b-instruct-fp8", help="Lambda API model name") # Qwen3 big: qwen3-32b-fp8
    
    # Generation settings
    parser.add_argument("--few-shot", type=int, default=2, help="Number of few-shot examples")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of examples (0 = no limit)")
    parser.add_argument("--all-relations", action="store_true", help="Use examples from all relations if needed")
    parser.add_argument("--mode", type=str, choices=["help", "nohelp"], required=True,
                       help="Whether to provide the answers (help) or have the model generate them (nohelp)")
    
    # Append mode
    parser.add_argument("--append", action="store_true", help="Append to existing output file instead of overwriting")
    
    # Add new sampling strategy option
    parser.add_argument("--full-dataset-repeats", type=int, default=10, 
                       help="Number of times to repeat the full dataset (0 uses --limit instead)")
    
    args = parser.parse_args()
    
    # Set default paths based on dataset if not provided
    if args.train_data is None:
        args.train_data = f"data/{args.dataset}/data/train.csv"
    if args.val_data is None:
        args.val_data = f"data/{args.dataset}/data/val.csv"
    if args.human_cot is None:
        args.human_cot = f"data/{args.dataset}/cot/nohelp_synthetic_cot/human_cot_single.csv"
    if args.question_prompts is None:
        args.question_prompts = f"data/{args.dataset}/question_prompts.csv"
    if args.output is None:
        args.output = f"data/{args.dataset}/cot/nohelp_synthetic_cot/raw/synthetic_lambda.csv"
    
    generate_cots(args)
