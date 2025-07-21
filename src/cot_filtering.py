import argparse
import pandas as pd
import os
import sys
from typing import List, Dict

from models.generation import generate_with_vllm
from models.loading import load_vllm_model

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_question_prompts(dataset: str) -> pd.DataFrame:
    """Load question prompts for the given dataset."""
    question_prompts_path = f"data/{dataset}/question_prompts.csv"
    if not os.path.exists(question_prompts_path):
        raise FileNotFoundError(f"Question prompts file not found: {question_prompts_path}")
    return pd.read_csv(question_prompts_path)

def create_coherence_prompt(question: str, cot_responses: List[str]) -> List[Dict[str, str]]:
    """
    Create a prompt to evaluate if multiple CoT responses are contradicting each other.
    
    Args:
        question: The original question
        cot_responses: List of CoT reasoning texts
        
    Returns:
        Structured chat messages for coherence evaluation
    """
    system_message = """You are an expert evaluator tasked with determining if multiple reasoning chains about the same question are contradictory.

Your job is to identify cases where the reasoning chains contain clear contradictions or appear to be independent hallucinations rather than different valid approaches to the same problem.

You should answer "yes" (contradictory) only in extreme cases where:
- The reasoning chains make contradictory factual claims
- The chains appear to be completely unrelated hallucinations
- There are fundamental logical contradictions between the approaches

You should answer "no" (not contradictory) when:
- The chains use different but valid reasoning approaches
- The chains focus on different aspects of the question
- There are minor differences in phrasing or emphasis
- The chains are complementary rather than contradictory

Be conservative - only flag clear contradictions, not mere differences in approach."""

    # Format the CoT responses
    cot_text = ""
    for i, cot in enumerate(cot_responses, 1):
        cot_text += f"Chain-of-Thought {i}: {cot.strip()}\n\n"
    
    user_message = f"""Question: {question}

Here are multiple reasoning chains (CoT) for this question:

{cot_text.strip()}

Are these reasoning chains contradictory or do they appear to be independent hallucinations? Answer with "yes" if they are contradictory/hallucinations, or "no" if they represent valid different approaches.

Answer (yes/no):"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def evaluate_coherence_with_llm(
    question: str, 
    cot_responses: List[str],
    model_name: str = "q_8i"
) -> bool:
    """
    Use an LLM to evaluate if CoT responses are contradictory.
    
    Args:
        question: The original question
        cot_responses: List of CoT reasoning texts
        model_name: Model to use for evaluation
        
    Returns:
        True if responses are contradictory (should be filtered), False otherwise
    """
    # Create prompt
    prompt = create_coherence_prompt(question, cot_responses)
    
    model, tokenizer = load_vllm_model(model_name, quantized=False)
    
    generation_params = {
        "max_tokens": 10,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    responses = generate_with_vllm(model, [prompt], tokenizer, generation_params)
    response_text = responses[0].strip().lower()
    
    # Parse response - look for "yes" indicating contradiction
    return "yes" in response_text

def process_cot_filtering(input_file: str, dataset: str = "dataset2025") -> None:
    """
    Process CoT filtering on a cleaned synthetic CoT file.
    
    Args:
        input_file: Path to the input CSV file
        dataset: Dataset name for question prompts
    """
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Load question prompts
    question_prompts = load_question_prompts(dataset)
    question_dict = dict(zip(question_prompts['Relation'], question_prompts['PromptTemplate']))
    
    # Create output file paths
    input_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    base_name = input_filename.replace('.csv', '')
    
    kept_file = os.path.join(input_dir, f"cot_filtered_{base_name}.csv")
    removed_file = os.path.join(input_dir, f"OUT_cot_filtered_{base_name}.csv")
    
    # Separate dataframes for kept and removed entries
    kept_rows = []
    removed_rows = []
    
    # Group by subject entity and relation
    grouped = df.groupby(['SubjectEntity', 'Relation'])
    
    for (subject_entity, relation), group in grouped:
        # Always keep incomplete quality entries
        incomplete_entries = group[group['quality'] == 'incomplete']
        if len(incomplete_entries) > 0:
            kept_rows.extend(incomplete_entries.to_dict('records'))
        
        # Process correct quality entries
        correct_entries = group[group['quality'] == 'correct']
        
        if len(correct_entries) <= 1:
            # Single or no correct entries - keep them
            kept_rows.extend(correct_entries.to_dict('records'))
        else:
            # Multiple correct entries - check for coherence
            
            # Get question template
            if relation not in question_dict:
                print(f"Warning: No question template found for relation {relation}")
                kept_rows.extend(correct_entries.to_dict('records'))
                continue
            
            question_template = question_dict[relation]
            question = question_template.format(subject_entity=subject_entity)
            
            # Extract CoT responses
            cot_responses = []
            for _, entry in correct_entries.iterrows():
                cot_text = entry['CoT']
                # Extract thinking part if present
                if '<think>' in cot_text and '</think>' in cot_text:
                    thinking = cot_text.split('<think>')[1].split('</think>')[0].strip()
                    cot_responses.append(thinking)
                else:
                    cot_responses.append(cot_text)
            
            # Evaluate coherence
            are_contradictory = evaluate_coherence_with_llm(question, cot_responses)
            
            if are_contradictory:
                print(f"Removing contradictory CoT for {subject_entity}")
                removed_rows.extend(correct_entries.to_dict('records'))
            else:
                kept_rows.extend(correct_entries.to_dict('records'))
        
        # Keep all other quality entries (wrong, failed, etc.)
        other_entries = group[~group['quality'].isin(['correct', 'incomplete'])]
        if len(other_entries) > 0:
            kept_rows.extend(other_entries.to_dict('records'))
    
    # Create output dataframes
    kept_df = pd.DataFrame(kept_rows)
    removed_df = pd.DataFrame(removed_rows)
    
    # Ensure same column order as original
    if len(kept_df) > 0:
        kept_df = kept_df[df.columns]
    if len(removed_df) > 0:
        removed_df = removed_df[df.columns]
    
    # Save results
    print(f"Saving {len(kept_df)} kept entries to {kept_file}")
    kept_df.to_csv(kept_file, index=False)
    
    print(f"Saving {len(removed_df)} removed entries to {removed_file}")
    removed_df.to_csv(removed_file, index=False)
    
    # Verify the split
    total_original = len(df)
    total_kept = len(kept_df)
    total_removed = len(removed_df)
    
    print(f"\nSummary:")
    print(f"Original entries: {total_original}")
    print(f"Kept entries: {total_kept}")
    print(f"Removed entries: {total_removed}")
    print(f"Total: {total_kept + total_removed} (should equal {total_original})")
    
    if total_kept + total_removed != total_original:
        print("WARNING: Total does not match original count!")
    else:
        print("✓ Verification passed")

def infer_input_path(input_arg: str, dataset: str) -> str:
    """Infer full input path from partial path or filename."""
    if os.path.exists(input_arg):
        return input_arg
    
    if '/' in input_arg:
        return input_arg
    
    filename = input_arg
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Look in clean directory first
    inferred_path = f"data/{dataset}/cot/nohelp_synthetic_cot/clean/{filename}"
    
    if os.path.exists(inferred_path):
        return inferred_path
    
    # Alternative patterns
    alt_patterns = [
        f"data/{dataset}/cot/clean/{filename}",
        f"data/{dataset}/cot/{filename}",
    ]
    
    for pattern in alt_patterns:
        if os.path.exists(pattern):
            return pattern
    
    return inferred_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter synthetic CoT data based on coherence")
    
    parser.add_argument("input", type=str,
                        help="Input CSV file name or path (e.g., 'new_synthetic_llama_all' or full path)")
    parser.add_argument("--dataset", type=str, default="dataset2025",
                        help="Dataset name (default: dataset2025)")
    parser.add_argument("--model", type=str, default="q_8i",
                        help="Model to use for coherence evaluation (default: q_8i)")
    
    args = parser.parse_args()
    
    # Infer input path
    input_path = infer_input_path(args.input, args.dataset)
    print(f"Using input path: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Process the file
    process_cot_filtering(input_path, args.dataset)