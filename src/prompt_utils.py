import random
import pandas as pd

def atom_question(subject, question_template):
    """
    Returns a basic prompt given a subject and a question template.
    """
    return f"Q: {question_template.format(subject_entity=subject)}\nA:"

def format_answer(objects):
    """
    Formats a list of object entities as a string.
    """
    random.shuffle(objects)
    answer = ", ".join(objects) if len(objects) > 0 else "None"
    return answer

def validate_cot_data(train_df, use_cot=False):
    """
    Validate that CoT data is available if CoT mode is activated.
    Throws exceptions instead of just returning False.
    
    Args:
        train_df: Training dataframe
        use_cot: Whether CoT mode is active
    """
    # Skip validation during inference (train_df=None)
    if train_df is None:
        return True
        
    if use_cot and 'CoT' not in train_df.columns:
        raise ValueError("CoT mode is activated but 'CoT' column is missing in training data.")
    
    if use_cot and train_df['CoT'].isna().all():
        raise ValueError("CoT mode is activated but all 'CoT' values are empty in training data.")
        
    return True

def find_few_shot_prompting(relation, train_df, few_shot=5):
    """
    Returns a list of few-shot examples for the specified relation with unique subject entities.
    """
    relation_df = train_df[train_df["Relation"] == relation]
    
    # Ensure unique subject entities
    unique_subjects = relation_df['SubjectEntity'].unique()
    actual_few_shot = min(few_shot, len(unique_subjects))
    
    if actual_few_shot > 0:
        selected_subjects = pd.Series(unique_subjects).sample(actual_few_shot)
        examples = relation_df[relation_df['SubjectEntity'].isin(selected_subjects)].groupby('SubjectEntity').sample(1)
        return examples.to_dict("records")
    else:
        return []

def prompt_few_shot(subject, relation, question_template, train_df, question_prompts, few_shot=0, few_shot_other=0, use_cot=False, system_message=None):
    """
    Builds a prompt with few-shot examples + the new question at the end.
    If use_cot is True, includes CoT reasoning for examples.
    """
    # Validate that CoT data is available if CoT mode is activated
    if use_cot:
        validate_cot_data(train_df, use_cot)
    
    prompt_parts = []
    
    # Add system message if provided
    if system_message:
        prompt_parts.append(f"System: {system_message}\n")
    
    # Get examples
    examples = pd.DataFrame()
    if few_shot > 0 or few_shot_other > 0:
        # Target relation examples
        if few_shot > 0:
            # If using CoT, ensure we get examples with CoT
            if use_cot:
                # Filter to only examples that have CoT for target relation
                valid_examples = train_df[train_df['Relation'] == relation].dropna(subset=['CoT'])
                
                if len(valid_examples) < few_shot:
                    print(f"WARNING: Only {len(valid_examples)} CoT examples found for relation {relation}, requested {few_shot}.")
                    # Use all available examples or adjust few_shot
                    few_shot = min(few_shot, len(valid_examples))
                    
                if few_shot > 0:
                    # Sample examples with unique subject entities for CoT
                    unique_subjects = valid_examples['SubjectEntity'].unique()
                    if len(unique_subjects) < few_shot:
                        print(f"WARNING: Only {len(unique_subjects)} unique subjects found for relation {relation}, requested {few_shot}.")
                        few_shot = len(unique_subjects)
                    
                    if few_shot > 0:
                        selected_subjects = pd.Series(unique_subjects).sample(few_shot)
                        target_examples = valid_examples[valid_examples['SubjectEntity'].isin(selected_subjects)].groupby('SubjectEntity').sample(1)
                    else:
                        target_examples = pd.DataFrame()
                else:
                    target_examples = pd.DataFrame()  # Empty DataFrame
            else:
                # Get examples as DataFrame with unique subject entities
                rel_df = train_df[train_df["Relation"] == relation]
                if len(rel_df) > 0:
                    unique_subjects = rel_df['SubjectEntity'].unique()
                    if len(unique_subjects) < few_shot:
                        print(f"WARNING: Only {len(unique_subjects)} unique subjects found for relation {relation}, requested {few_shot}.")
                        few_shot = len(unique_subjects)
                    
                    if few_shot > 0:
                        selected_subjects = pd.Series(unique_subjects).sample(min(few_shot, len(unique_subjects)))
                        target_examples = rel_df[rel_df['SubjectEntity'].isin(selected_subjects)].groupby('SubjectEntity').sample(1)
                    else:
                        target_examples = pd.DataFrame()
                else:
                    target_examples = pd.DataFrame()
            
            examples = pd.concat([examples, target_examples])
        
        # Other relation examples
        if few_shot_other > 0:
            other_relations_df = train_df[train_df["Relation"] != relation]
            
            if use_cot:
                # Filter to only examples that have CoT for other relations
                other_relations_df = other_relations_df.dropna(subset=['CoT'])
            
            if len(other_relations_df) > 0:
                # Sample examples with unique subject entities for other relations
                unique_subjects = other_relations_df['SubjectEntity'].unique()
                if len(unique_subjects) < few_shot_other:
                    print(f"WARNING: Only {len(unique_subjects)} unique subjects found for other relations, requested {few_shot_other}.")
                    few_shot_other = len(unique_subjects)
                
                if few_shot_other > 0:
                    selected_subjects = pd.Series(unique_subjects).sample(few_shot_other)
                    other_examples = other_relations_df[other_relations_df['SubjectEntity'].isin(selected_subjects)].groupby('SubjectEntity').sample(1)
                    examples = pd.concat([examples, other_examples])
            else:
                print(f"WARNING: No examples found for other relations, requested {few_shot_other}.")
        
        # Shuffle all examples
        examples = examples.sample(frac=1.0)
        
        # Process each example
        for _, example in examples.iterrows():
            subject_example = example['SubjectEntity']
            rel_example = example['Relation']
            # Get the question template for this example's relation
            rel_question_template = question_prompts[question_prompts["Relation"] == rel_example]["PromptTemplate"].iloc[0]
            question = rel_question_template.format(subject_entity=subject_example)
            answer_list = eval(example['ObjectEntities'])
            formatted_answer = format_answer(answer_list)
            
            if use_cot and 'CoT' in example and pd.notna(example['CoT']):
                # Add the CoT from the example
                cot = example['CoT']
                prompt_parts.append(f"Q: {question}\n{cot}")
            else:
                prompt_parts.append(f"Q: {question}\nA: {formatted_answer}")
    
    # Add the current question
    current_question = atom_question(subject, question_template)
    prompt = "\n\n".join(prompt_parts) + f"\n\n{current_question}" if prompt_parts else current_question
    
    return prompt.strip()

def format_chat_messages(subject, relation, question_template, train_df, question_prompts=None, few_shot=0, few_shot_other=0, use_cot=False, system_message=None):
    """
    Format a conversation with few-shot examples for chat-based models.
    """
    # Validate CoT data availability - this will throw an exception if validation fails
    if train_df is not None and use_cot:
        validate_cot_data(train_df, use_cot)
    
    messages = []
    
    # Add system message if provided
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Format few-shot examples
    examples = pd.DataFrame()
    if few_shot > 0 or few_shot_other > 0:
        # Target relation examples
        if few_shot > 0:
            if use_cot:
                valid_examples = train_df[train_df['Relation'] == relation].dropna(subset=['CoT'])
                if len(valid_examples) < few_shot:
                    print(f"WARNING: Only {len(valid_examples)} CoT examples found for relation {relation}, requested {few_shot}.")
                    few_shot = min(few_shot, len(valid_examples))
                
                # Sample examples with unique subject entities
                if few_shot > 0:
                    unique_subjects = valid_examples['SubjectEntity'].unique()
                    if len(unique_subjects) < few_shot:
                        print(f"WARNING: Only {len(unique_subjects)} unique subjects found for relation {relation}, requested {few_shot}.")
                        few_shot = len(unique_subjects)
                    
                    if few_shot > 0:
                        selected_subjects = pd.Series(unique_subjects).sample(few_shot)
                        target_examples = valid_examples[valid_examples['SubjectEntity'].isin(selected_subjects)].groupby('SubjectEntity').sample(1)
                    else:
                        target_examples = pd.DataFrame()
                else:
                    target_examples = pd.DataFrame()
            else:
                rel_df = train_df[train_df["Relation"] == relation]
                if len(rel_df) > 0:
                    # Sample examples with unique subject entities for non-CoT
                    unique_subjects = rel_df['SubjectEntity'].unique()
                    if len(unique_subjects) < few_shot:
                        print(f"WARNING: Only {len(unique_subjects)} unique subjects found for relation {relation}, requested {few_shot}.")
                        few_shot = len(unique_subjects)
                    
                    if few_shot > 0:
                        selected_subjects = pd.Series(unique_subjects).sample(few_shot)
                        target_examples = rel_df[rel_df['SubjectEntity'].isin(selected_subjects)].groupby('SubjectEntity').sample(1)
                    else:
                        target_examples = pd.DataFrame()
                else:
                    target_examples = pd.DataFrame()
            
            examples = pd.concat([examples, target_examples])
        
        # Other relation examples
        if few_shot_other > 0:
            other_relations_df = train_df[train_df["Relation"] != relation]
            
            if use_cot:
                # Filter to only examples that have CoT for other relations
                other_relations_df = other_relations_df.dropna(subset=['CoT'])
            
            if len(other_relations_df) > 0:
                # Sample examples with unique subject entities for other relations
                unique_subjects = other_relations_df['SubjectEntity'].unique()
                if len(unique_subjects) < few_shot_other:
                    print(f"WARNING: Only {len(unique_subjects)} unique subjects found for other relations, requested {few_shot_other}.")
                    few_shot_other = len(unique_subjects)
                
                if few_shot_other > 0:
                    selected_subjects = pd.Series(unique_subjects).sample(few_shot_other)
                    other_examples = other_relations_df[other_relations_df['SubjectEntity'].isin(selected_subjects)].groupby('SubjectEntity').sample(1)
                    examples = pd.concat([examples, other_examples])
            else:
                print(f"WARNING: No examples found for other relations, requested {few_shot_other}.")
        
        # Shuffle all examples
        examples = examples.sample(frac=1.0)
        
        # Process each example
        for _, example in examples.iterrows():
            subject_example = example['SubjectEntity']
            rel_example = example['Relation']
            # Get the question template for this example's relation
            rel_question_template = question_prompts[question_prompts["Relation"] == rel_example]["PromptTemplate"].iloc[0]
            question = rel_question_template.format(subject_entity=subject_example)
            answer_list = eval(example['ObjectEntities'])
            formatted_answer = format_answer(answer_list)
            
            # Add question as user message
            messages.append({"role": "user", "content": question})
            
            # Add answer as assistant message, with CoT if applicable
            if use_cot and 'CoT' in example and pd.notna(example['CoT']):
                assistant_message = f"{example['CoT']}"  # The CoT already includes the answer and the think tags
                messages.append({"role": "assistant", "content": assistant_message})
            else:
                messages.append({"role": "assistant", "content": formatted_answer})
    
    # Add the current question
    current_question = question_template.format(subject_entity=subject)
    messages.append({"role": "user", "content": current_question})
    
    return messages

def prompt_cot(base_prompt):
    """
    Adds a simple chain-of-thought scaffold to the base prompt with start and end answer placeholders.
    """
    return f"{base_prompt} Let's reason step by step."

def parse_cot_answer(generated_text):
    """
    Parse the generated text to extract the CoT reasoning and the final answer.
    The CoT reasoning is enclosed within <think> and </think> tags.
    """
    start_seq = "<think>"
    end_seq = "</think>"
    
    cot_reasoning = ""
    final_answer = generated_text
    
    if start_seq in generated_text and end_seq in generated_text:
        cot_reasoning = generated_text.split(start_seq)[1].split(end_seq)[0].strip()
        final_answer = generated_text.split(end_seq)[-1].strip()
    
    return cot_reasoning, final_answer