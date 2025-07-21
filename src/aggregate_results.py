import os
import argparse
import json
import pandas as pd
import yaml
import re
import datetime
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

def clean_cot_source_name(cot_source):
    """Clean up the CoT source name to make it more readable"""
    if not cot_source or cot_source == "none":
        return "none"
    
    # For complex paths like nohelp_synthetic_cot/clean/2shot_temp1.0_lambda70b_correct_incomplete
    if '/' in cot_source:
        # Extract the first part before the slash
        primary_part = cot_source.split('/')[0]
        if primary_part:
            return primary_part
    
    return cot_source

def escape_latex_special_chars(text):
    """Escape special characters for LaTeX output"""
    if not isinstance(text, str):
        return text
    
    # Replace underscores with escaped underscores
    text = text.replace('_', '\\_')
    # Replace other special characters if needed
    text = text.replace('#', '\\#')
    text = text.replace('%', '\\%')
    text = text.replace('&', '\\&')
    
    return text

def is_recent_experiment(exp_dir, max_days=2):
    """Check if an experiment is recent based on its directory name or timestamp"""
    # Look for timestamp in directory name (format: YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_JOBID)
    timestamp_match = re.search(r'(\d{8}_\d{6})', exp_dir)
    
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
        try:
            # Parse the timestamp (ignoring any job ID suffix)
            exp_date = datetime.datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            # Calculate the age in days
            age_days = (datetime.datetime.now() - exp_date).days
            return age_days <= max_days
        except ValueError:
            pass  # Continue with other methods if parsing fails
    
    # If no timestamp in the name or parsing failed, use the directory's creation time
    try:
        exp_path = Path(exp_dir)
        creation_time = datetime.datetime.fromtimestamp(exp_path.stat().st_ctime)
        age_days = (datetime.datetime.now() - creation_time).days
        return age_days <= max_days
    except:
        # If all else fails, include the experiment
        return True

def ensure_save_dir(dataset_name):
    """
    Ensure the saved directory exists for the dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        str: Path to the saved directory
    """
    save_dir = os.path.join("saved", dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def create_relation_comparison_table(dataset_name, target_experiments, print_comparisons=True):
    """
    Create a comparison table showing performance by relation across target experiments
    
    Args:
        dataset_name: Name of the dataset
        target_experiments: List of (experiment_id, description) tuples to include in comparison
        print_comparisons: Whether to print the comparison tables
    """
    # Load experiment data
    base_dir = f"predictions/{dataset_name}"
    if not os.path.exists(base_dir):
        print(f"No experiments found for dataset: {dataset_name}")
        return None
    
    # Find all subdirectories
    exp_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    # Dictionary to store experiment data
    experiment_data = {}
    
    # Find and load the target experiments
    for target_id, description in target_experiments:
        for exp_dir in exp_dirs:
            full_path = os.path.join(base_dir, exp_dir)
            config_path = os.path.join(full_path, "config.yaml")
            if not os.path.exists(config_path):
                continue
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            experiment_name = config["experiment"].get("name", "")
            
            # Match by experiment name
            if target_id in experiment_name:
                # Special case for q_7i_0shot with few-shot=5
                if target_id == "q_7i_0shot" and config["prompting"].get("few_shot", 0) != 5:
                    continue
                
                results_path = os.path.join(full_path, "results.json")
                if not os.path.exists(results_path):
                    continue
                
                with open(results_path, 'r') as f:
                    metrics = json.load(f)
                
                experiment_data[description] = metrics
                break
    
    if not experiment_data:
        print("No experiments found for comparison")
        return None
    
    if not print_comparisons:
        # Skip printing if print_comparisons is False
        return experiment_data
    
    # Get all relations
    all_relations = set()
    for metrics in experiment_data.values():
        all_relations.update(metrics.keys())
    
    # Sort relations but keep "*** All Relations ***" first
    all_relations = sorted([r for r in all_relations if r != "*** All Relations ***"])

    all_relations = ["*** All Relations ***"] + all_relations
    
    # List of metrics to include - always use macro metrics
    metrics_to_show = ["macro-p", "macro-r", "macro-f1", "avg. #preds", "#empty preds"]
    
    # Print simple tabular output GROUPED BY RELATION
    print("\n" + "=" * 100)
    print("RELATION-BY-RELATION COMPARISON")
    print("=" * 100)
    
    # For each relation, print all metrics for all models
    for relation in all_relations:
        display_name = "Overall" if relation == "*** All Relations ***" else relation
        print(f"\n{display_name}:")
        
        # Print header row with metric names
        header_row = "Model".ljust(30)
        for metric in metrics_to_show:
            header_row += metric.ljust(15)
        print(header_row)
        print("-" * (30 + 15 * len(metrics_to_show)))
        
        # Print each model's metrics
        for model in experiment_data:
            model_row = model.ljust(30)
            if relation in experiment_data[model]:
                rel_metrics = experiment_data[model][relation]
                for metric in metrics_to_show:
                    value = rel_metrics.get(metric, "N/A")
                    if isinstance(value, float):
                        model_row += f"{value:.3f}".ljust(15)
                    else:
                        model_row += f"{value}".ljust(15)
            else:
                model_row += "N/A".ljust(15) * len(metrics_to_show)
            print(model_row)
        
        # Add an empty line for readability
        print("")
    
    # Create a more readable LaTeX table with stacked methods
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    
    # Enhanced caption with method descriptions
    caption = f"Macro Performance Comparison by Relation for {dataset_name}. Methods: "
    caption += "(1) Relation Tuning with CoT: fine-tuned model using chain-of-thought reasoning; "
    caption += "(2) Relation Tuning without CoT: fine-tuned model without reasoning steps; "
    caption += "(3) Prompting baseline: zero-shot model with 5 examples; "
    caption += "(4) Prompting with CoT: zero-shot model with chain-of-thought prompting and 2 examples."
    
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append("\\begin{tabular}{llccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Relation & Method & P & R & F1 \\\\")
    latex_lines.append("\\midrule")
    
    # Add data rows for each relation
    for relation in all_relations:
        # Format the relation name, with special handling for "Overall"
        display_name = relation.replace("_", "\\_")
        if relation == "*** All Relations ***":
            display_name = "\\textbf{Overall}"
        
        # Determine the number of models for multirow
        num_models = len(experiment_data)
        
        # Create multirow for relation name
        latex_lines.append(f"\\multirow{{{num_models}}}{{*}}{{{display_name}}} ")
        
        # Add each method as a separate row under this relation
        first_row = True
        for model in experiment_data:
            # Only add & for subsequent rows (not the first one in the multirow)
            prefix = "  & " if first_row else "  & "
            first_row = False
            
            escaped_model = model.replace("_", "\\_")
            row = prefix + escaped_model
            
            if relation in experiment_data[model]:
                # Always use macro metrics
                p_value = experiment_data[model][relation].get("macro-p", 0)
                r_value = experiment_data[model][relation].get("macro-r", 0)
                f1_value = experiment_data[model][relation].get("macro-f1", 0)
                row += f" & {p_value:.3f} & {r_value:.3f} & {f1_value:.3f}"
            else:
                row += " & --- & --- & ---"
                
            latex_lines.append(row + " \\\\")
        
        # Add midrule after each relation block
        latex_lines.append("\\midrule")
    
    # Replace last midrule with bottomrule
    latex_lines[-1] = "\\bottomrule"
    
    # Complete the table
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\label{tab:relation_comparison}")
    latex_lines.append("\\end{table}")
    
    # Save the LaTeX table
    save_dir = ensure_save_dir(dataset_name)
    latex_file = os.path.join(save_dir, f"relation_comparison_{dataset_name}.tex")
    with open(latex_file, 'w') as f:
        f.write("\n".join(latex_lines))
    print(f"\nMacro metric comparison table saved to {latex_file}")
    
    return experiment_data

def plot_f1_by_epoch(results_df, dataset_name):
    """
    Generate and save a plot showing F1 scores as a function of epoch,
    with different colors for different models and line styles for CoT usage
    
    Args:
        results_df: DataFrame with aggregated results
        dataset_name: Name of the dataset for the plot title and filename
    """
    # Filter for experiments with epochs > 0 (i.e., PEFT models)
    peft_df = results_df[results_df["peft_epochs"] > 0].copy()
    
    if len(peft_df) == 0:
        print("No experiments with epochs found, skipping plot generation")
        return
    
    # Ensure F1 is numeric and handle any inf values to avoid warnings
    peft_df["macro_f1"] = pd.to_numeric(peft_df["macro_f1"], errors='coerce')
    # Replace inf values with NaN to avoid seaborn warnings
    peft_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Set up plot style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 8))
    
    # Create a categorical color palette based on unique model names
    model_names = peft_df["model_name"].unique()
    palette = sns.color_palette("husl", len(model_names))
    color_map = dict(zip(model_names, palette))
    
    # Suppress the specific FutureWarning from seaborn
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, 
                               message="use_inf_as_na option is deprecated")
        
        # Create lineplot with different colors for models and line styles for CoT
        ax = sns.lineplot(
            data=peft_df,
            x="peft_epochs",
            y="macro_f1",
            hue="model_name",
            style="cot",
            markers=True,
            dashes=[(1, 0), (2, 2)],  # Solid line for no CoT, dashed line for CoT
            palette=color_map,
            linewidth=2.5,
            markersize=10
        )
    
    # Set plot labels and title
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Macro F1 Score", fontsize=14)
    ax.set_title(f"F1 Score by Epoch", fontsize=16)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    
    # Ensure x-axis shows integer values for epochs
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title_fontsize=12, fontsize=11, 
             title="Model & CoT Usage", loc='best', frameon=True, framealpha=0.9)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    save_dir = ensure_save_dir(dataset_name)
    plot_path = os.path.join(save_dir, f"f1_by_epoch_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nF1 score by epoch plot saved to {plot_path}")
    
    # Close the plot to free memory
    plt.close()

def detect_relation_in_prompt(experiment_dir):
    """
    Detect if an experiment uses relation in prompt based on the date
    
    Args:
        experiment_dir: Experiment directory name
        
    Returns:
        bool: True if relation in prompt was used
    """
    # Check if the experiment is from 20250415 or 20250416
    date_match = re.search(r'(20250415|20250416)', experiment_dir)
    return bool(date_match)

def detect_lora_usage(config):
    """
    Detect if an experiment uses LoRA based on the config
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        bool: True if LoRA was used
    """
    # Check for explicit lora method in peft_method field
    if "peft_method" in config["model"] and config["model"]["peft_method"] == "lora":
        return True
        
    # Check for LoRA in the config as a fallback
    if "lora" in str(config).lower():
        return True
        
    # Check for specific LoRA parameters in the model config
    if "peft_config" in config["model"] and "r" in config["model"]["peft_config"]:
        return True
        
    return False

def plot_model_comparison(results_df, dataset_name):
    """
    Generate plot comparing l_8i and q_7i models
    
    Args:
        results_df: DataFrame with aggregated results
        dataset_name: Name of the dataset for the plot title and filename
    """
    # Filter for l_8i and q_7i models only
    model_df = results_df[
        (results_df["model_name"].isin(["l_8i", "q_7i"])) & 
        (results_df["peft_epochs"] > 0)
    ].copy()
    
    # Exclude LoRA and relation in prompt experiments
    model_df = model_df[~model_df["experiment_dir"].apply(detect_relation_in_prompt)]
    
    if len(model_df) == 0:
        print("No valid experiments for model comparison plot")
        return
    
    # Print debug info about experiments being used
    print("\nExperiments used for MODEL COMPARISON plot:")
    for idx, row in model_df.iterrows():
        print(f"  - {row['model_name']} - CoT: {row['cot']} - Epoch: {row['peft_epochs']} - Dir: {row['experiment_dir']}")
    
    # Ensure F1 is numeric
    model_df["macro_f1"] = pd.to_numeric(model_df["macro_f1"], errors='coerce')
    model_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create the plot
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 8))
    
    # Create consistent CoT labels across all plots
    model_df["cot_str"] = model_df["cot"].apply(lambda x: "With CoT" if x else "Without CoT")
    
    # Create a mapping for display names
    model_display_names = {
        "l_8i": "Llama 8b",
        "q_7i": "Qwen 7b"
    }
    
    # Add a column with display names
    model_df["model_display"] = model_df["model_name"].map(model_display_names)
    
    # Create color palette
    palette = {"Llama 8b": "blue", "Qwen 7b": "red"}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        ax = sns.lineplot(
            data=model_df,
            x="peft_epochs",
            y="macro_f1",
            hue="model_display",  # Use display names
            style="cot_str",
            markers=True,
            dashes=[(1, 0), (2, 2)],
            palette=palette,
            linewidth=2.5,
            markersize=10
        )
    
    # Set plot labels and title
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Macro F1 Score", fontsize=14)
    ax.set_title(f"Model Comparison: Llama 8b vs Qwen 7b", fontsize=16)
    
    # Format y-axis and set x-axis to integers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title_fontsize=12, fontsize=11,
             title="Model & CoT Usage", loc='best', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    save_dir = ensure_save_dir(dataset_name)
    plot_path = os.path.join(save_dir, f"model_comparison_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nModel comparison plot saved to {plot_path}")
    
    plt.close()

def plot_peft_comparison(results_df, dataset_name):
    """
    Generate plot comparing LoRA vs prompt tuning for l_8i model
    
    Args:
        results_df: DataFrame with aggregated results
        dataset_name: Name of the dataset for the plot title and filename
    """
    # Filter for l_8i model only
    peft_df = results_df[
        (results_df["model_name"] == "l_8i") & 
        (results_df["peft_epochs"] > 0)
    ].copy()
    
    if len(peft_df) == 0:
        print("No valid experiments for PEFT comparison plot")
        return
    
    # Add LoRA flag and relation-in-prompt flag to the dataframe
    base_dir = f"predictions/{dataset_name}"
    peft_df["lora"] = False
    peft_df["relation_in_prompt"] = peft_df["experiment_dir"].apply(detect_relation_in_prompt)
    
    for idx, row in peft_df.iterrows():
        exp_dir = row["experiment_dir"]
        full_path = os.path.join(base_dir, exp_dir)
        config_path = os.path.join(full_path, "config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                peft_df.at[idx, "lora"] = detect_lora_usage(config)
    
    # Filter the dataframe to exclude relation-in-prompt experiments that are not LoRA
    # For Non-LoRA: only include the standard prompt (not relation-in-prompt)
    # For LoRA: include all LoRA experiments
    peft_df = peft_df[
        (peft_df["lora"]) |  # Include all LoRA experiments
        (~peft_df["relation_in_prompt"])  # For non-LoRA, only include standard prompt
    ]
    
    # Print debug info about experiments being used
    print("\nExperiments used for PEFT COMPARISON plot:")
    for idx, row in peft_df.iterrows():
        print(f"  - LoRA: {row['lora']} - CoT: {row['cot']} - Epoch: {row['peft_epochs']} - Dir: {row['experiment_dir']}")
    
    # Ensure F1 is numeric
    peft_df["macro_f1"] = pd.to_numeric(peft_df["macro_f1"], errors='coerce')
    peft_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create the plot
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 8))
    
    # Need to convert boolean to string for the palette to work correctly
    peft_df["lora_str"] = peft_df["lora"].astype(str)
    # Create consistent CoT labels across all plots
    peft_df["cot_str"] = peft_df["cot"].apply(lambda x: "With CoT" if x else "Without CoT")
    
    # Create color palette for PEFT types
    palette = {"True": "purple", "False": "green"}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        ax = sns.lineplot(
            data=peft_df,
            x="peft_epochs",
            y="macro_f1",
            hue="lora_str",  # Use string version for plotting
            style="cot_str",  # Use consistent CoT labeling
            markers=True,
            dashes=[(1, 0), (2, 2)],
            palette=palette,
            linewidth=2.5,
            markersize=10
        )
    
    # Set plot labels and title
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Macro F1 Score", fontsize=14)
    ax.set_title(f"PEFT Comparison: LoRA vs Prompt Tuning", fontsize=16)
    
    # Format y-axis and set x-axis to integers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Customize legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for l in labels:
        if l == "True":
            new_labels.append("LoRA")
        elif l == "False":
            new_labels.append("Prompt Tuning")
        elif l in ["With CoT", "Without CoT"]:
            new_labels.append(l)
        else:
            new_labels.append(l)
    
    ax.legend(handles, new_labels, title_fontsize=12, fontsize=11,
             title="PEFT Type & CoT Usage", loc='best', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    save_dir = ensure_save_dir(dataset_name)
    plot_path = os.path.join(save_dir, f"peft_comparison_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPEFT comparison plot saved to {plot_path}")
    
    plt.close()

def plot_prompt_comparison(results_df, dataset_name):
    """
    Generate plot comparing normal prompt vs relation in prompt for l_8i model
    
    Args:
        results_df: DataFrame with aggregated results
        dataset_name: Name of the dataset for the plot title and filename
    """
    # Filter for l_8i model only without LoRA
    prompt_df = results_df[
        (results_df["model_name"] == "l_8i") & 
        (results_df["peft_epochs"] > 0)
    ].copy()
    
    # Exclude q_7i model
    prompt_df = prompt_df[prompt_df["model_name"] != "q_7i"]
    
    if len(prompt_df) == 0:
        print("No valid experiments for prompt comparison plot")
        return
    
    # Add relation-in-prompt flag based on experiment date
    prompt_df["relation_in_prompt"] = prompt_df["experiment_dir"].apply(detect_relation_in_prompt)
    
    # Add LoRA flag to filter out LoRA experiments
    base_dir = f"predictions/{dataset_name}"
    prompt_df["lora"] = False
    
    for idx, row in prompt_df.iterrows():
        exp_dir = row["experiment_dir"]
        full_path = os.path.join(base_dir, exp_dir)
        config_path = os.path.join(full_path, "config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                prompt_df.at[idx, "lora"] = detect_lora_usage(config)
    
    # Filter out LoRA experiments
    prompt_df = prompt_df[~prompt_df["lora"]]
    
    # Print debug info about experiments being used
    print("\nExperiments used for PROMPT COMPARISON plot:")
    for idx, row in prompt_df.iterrows():
        print(f"  - Relation in prompt: {row['relation_in_prompt']} - CoT: {row['cot']} - Epoch: {row['peft_epochs']} - Dir: {row['experiment_dir']}")
    
    # Ensure F1 is numeric
    prompt_df["macro_f1"] = pd.to_numeric(prompt_df["macro_f1"], errors='coerce')
    prompt_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create the plot
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 8))
    
    # Convert boolean to string for the palette to work correctly
    prompt_df["relation_in_prompt_str"] = prompt_df["relation_in_prompt"].astype(str)
    prompt_df["cot_str"] = prompt_df["cot"].apply(lambda x: "With CoT" if x else "Without CoT")
    
    # Create color palette for prompt types
    palette = {"True": "orange", "False": "blue"}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        ax = sns.lineplot(
            data=prompt_df,
            x="peft_epochs",
            y="macro_f1",
            hue="relation_in_prompt_str",  # Use string version for plotting
            style="cot_str",  # Use the descriptive string version for style
            markers=True,
            dashes=[(1, 0), (2, 2)],
            palette=palette,
            linewidth=2.5,
            markersize=10
        )
    
    # Set plot labels and title
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Macro F1 Score", fontsize=14)
    ax.set_title(f"Prompt Comparison: Normal vs Relation-in-Prompt", fontsize=16)
    
    # Format y-axis and set x-axis to integers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Customize legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for l in labels:
        if l == "True":
            new_labels.append("Relation in Prompt")
        elif l == "False":
            new_labels.append("Standard Prompt")
        elif l in ["With CoT", "Without CoT"]:
            new_labels.append(l)
        else:
            new_labels.append(l)
    
    ax.legend(handles, new_labels, title_fontsize=12, fontsize=11,
             title="Prompt Type & CoT Usage", loc='best', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    save_dir = ensure_save_dir(dataset_name)
    plot_path = os.path.join(save_dir, f"prompt_comparison_{dataset_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPrompt comparison plot saved to {plot_path}")
    
    plt.close()

def aggregate_experiment_results(dataset_name, output_file=None, max_days=None, print_comparisons=False):
    """
    Aggregate experiment results for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        output_file: Path to output CSV file
        max_days: Maximum age in days for experiments to include, None for all
        print_comparisons: Whether to print the detailed relation comparison tables
        
    Returns:
        pd.DataFrame: DataFrame with aggregated results
    """
    # Get list of experiment directories
    base_dir = f"predictions/{dataset_name}"
    if not os.path.exists(base_dir):
        print(f"No experiments found for dataset: {dataset_name}")
        return None
        
    # Find all subdirectories
    exp_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    if not exp_dirs:
        print(f"No experiments found for dataset: {dataset_name}")
        return None
    
    print(f"Found {len(exp_dirs)} experiments for dataset: {dataset_name}")
    
    # Filter by recency if max_days is specified
    if max_days is not None:
        # Convert the full path to just the directory name for the filter function
        recent_exp_dirs = [d for d in exp_dirs if is_recent_experiment(os.path.join(base_dir, d), max_days)]
        print(f"Filtered to {len(recent_exp_dirs)} recent experiments (max age: {max_days} days)")
        exp_dirs = recent_exp_dirs
    
    # Collect results
    results = []
    
    for exp_dir in exp_dirs:
        full_path = os.path.join(base_dir, exp_dir)
        
        # Load config
        config_path = os.path.join(full_path, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Skipping {exp_dir} - missing config.yaml")
            continue
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load results.json
        results_path = os.path.join(full_path, "results.json")
        if not os.path.exists(results_path):
            print(f"Skipping {exp_dir} - missing results.json")
            continue
            
        with open(results_path, 'r') as f:
            metrics = json.load(f)
        
        # Get experiment details - handle different config formats
        raw_model_name = config["model"].get("base", config["model"].get("alias", "unknown"))
        model_name = raw_model_name  # no preprocessing needed
        few_shot = config["prompting"].get("few_shot", 0)
        
        # Handle both use_cot and cot fields (they mean the same thing)
        use_cot = config["prompting"].get("cot", config["prompting"].get("use_cot", False))
        cot_source = config["prompting"].get("cot_source", "none") if use_cot else "none"
        
        # Clean up CoT source name
        cot_source = clean_cot_source_name(cot_source)
        
        # Check for PEFT usage in different formats
        peft_used = False
        peft_epochs = 0
        cot_data_source = "none"
        cot_quality_filter = "none"
        
        # Check direct peft flag
        if config["model"].get("use_peft", False):
            peft_used = True
            peft_epochs = 1  # Default if not specified
        
        # First check if the epoch is directly available in the config
        if "epoch" in config["model"]:
            peft_used = True
            peft_epochs = config["model"]["epoch"]
            
        # For PEFT models, check the trained_model_dir for config.json
        trained_model_dir = config["model"].get("trained_model_dir", "")
        if trained_model_dir:
            peft_used = True
            
            # Look for config.json in the trained model directory
            model_config_path = os.path.join(trained_model_dir, "config.json")
            if os.path.exists(model_config_path):
                try:
                    with open(model_config_path, 'r') as f:
                        model_config = json.load(f)
                    
                    # Extract epochs from model config
                    peft_epochs = model_config.get("epochs", peft_epochs)  # Keep existing value if not present
                    
                    # Check for CoT information in model config
                    if model_config.get("use_cot", False) or model_config.get("cot", False):
                        use_cot = True
                        # Extract and clean CoT data source from the model config
                        if model_config.get("cot_data"):
                            cot_data_path = model_config.get("cot_data", "")
                            cot_basename = os.path.basename(cot_data_path)
                            cot_data_source = cot_basename.replace(".csv", "")
                            # Extract the primary part of the CoT data source
                            parts = cot_data_path.split('/')
                            if len(parts) >= 3:  # Most likely data/dataset/cot/...
                                cot_source = clean_cot_source_name(parts[3])  # This is typically the main CoT type
                            else:
                                cot_source = clean_cot_source_name(cot_data_source)
                        else:
                            cot_data_source = "unknown"
                            
                        cot_quality_filter = model_config.get("cot_quality_filter", "none")
                except:
                    print(f"Warning: Could not parse model config.json for {exp_dir}")
            else:
                # Only try to extract epochs from directory name if we haven't already found it
                if peft_epochs == 0:
                    # Try to extract epochs from directory name
                    epochs_match = re.search(r'_(\d+)epoch', trained_model_dir)
                    if epochs_match:
                        peft_epochs = int(epochs_match.group(1))
                    else:
                        # Try other common patterns in the directory name
                        epochs_match = re.search(r'epoch[_-]?(\d+)', trained_model_dir, re.IGNORECASE)
                        if epochs_match:
                            peft_epochs = int(epochs_match.group(1))
                        else:
                            # Default to 1 if we can't extract the number
                            peft_epochs = 1
        
        # Extract metrics from the "*** All Relations ***" section
        all_relations_key = "*** All Relations ***"
        if all_relations_key not in metrics:
            print(f"Skipping {exp_dir} - missing '{all_relations_key}' in results.json")
            continue
        
        all_relations = metrics[all_relations_key]
        
        # Extract specific metrics
        macro_p = all_relations.get("macro-p", 0)
        macro_r = all_relations.get("macro-r", 0)
        macro_f1 = all_relations.get("macro-f1", 0)
        micro_p = all_relations.get("micro-p", 0)
        micro_r = all_relations.get("micro-r", 0)
        micro_f1 = all_relations.get("micro-f1", 0)
        avg_preds = all_relations.get("avg. #preds", 0)
        empty_preds = all_relations.get("#empty preds", 0)
        
        # Add to results
        results.append({
            "experiment_dir": exp_dir,
            "model": raw_model_name,  # Keep original model identifier
            "model_name": model_name,  # Add clean base model name
            "few_shot": few_shot,
            "cot": use_cot,
            "cot_source": cot_source,
            "cot_data_source": cot_data_source if peft_used and use_cot else "none",
            "cot_quality_filter": cot_quality_filter if peft_used and use_cot else "none",
            "peft": peft_used,
            "peft_epochs": peft_epochs,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f1": macro_f1,
            "micro_p": micro_p,
            "micro_r": micro_r,
            "micro_f1": micro_f1,
            "avg_preds": avg_preds,
            "empty_preds": empty_preds,
            "experiment_name": config["experiment"].get("name", exp_dir)
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid results found")
        return None
    
    # Sort by your preferred order: PEFT, CoT, CoT Source, Epochs, Few-Shot
    results_df = results_df.sort_values(
        by=["peft", "cot", "cot_source", "peft_epochs", "few_shot"],
        ascending=[False, False, True, False, False]
    )
    
    # Save to file if specified
    if output_file:
        # If the output file doesn't include a directory, save to the saved directory
        if not os.path.dirname(output_file):
            save_dir = ensure_save_dir(dataset_name)
            output_file = os.path.join(save_dir, output_file)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    # Print summary table with enhanced information
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print("\nResults Summary (sorted by Relation Tuning, CoT, Source, Epochs, Few-Shot):")
    
    # Select and format the columns for display - remove experiment_name and move macro_f1 to the end
    display_cols = [
        "peft",
        "cot",
        "cot_source",
        "model_name",  # Add the new model_name column
    ]
    
    if any(results_df["peft"]):
        display_cols.append("peft_epochs")
    
    display_cols.append("few_shot")
    display_cols.append("macro_f1")  # Put score at the end
    
    # Create a formatted display DataFrame
    display_df = results_df[display_cols].copy()
    
    # For non-PEFT models with CoT, keep CoT source separate for clarity
    display_df["cot"] = display_df["cot"].apply(lambda x: "Yes" if x else "No")
    
    # Format PEFT information
    display_df["peft"] = display_df["peft"].apply(lambda x: "Yes" if x else "No")
    
    # Update the CoT source to "train" for trained models or when using few-shot
    display_df["cot_source"] = display_df.apply(
        lambda x: "train" if (not x["cot"] and (x["peft"] == "Yes" or x["few_shot"] > 0)) else x["cot_source"], 
        axis=1
    )
    
    # Escape special characters in string columns for LaTeX
    for col in display_df.columns:
        if display_df[col].dtype == 'object':  # String columns
            display_df[col] = display_df[col].apply(escape_latex_special_chars)
    
    # Rename columns for better readability
    display_df = display_df.rename(columns={
        "few_shot": "Few-Shot",
        "cot": "CoT",
        "cot_source": "Source",
        "peft": "Relation Tuning",
        "peft_epochs": "Epochs",
        "macro_f1": "Macro F1",
        "model_name": "Model"
    })
    
    # Format floating point numbers in a more readable way - use 3 decimal places for Macro F1
    display_df["Macro F1"] = display_df["Macro F1"].apply(lambda x: f"{x:.3f}")
    
    # Print standard output (with underscores not escaped for terminal display)
    print(display_df.to_string(index=False))
    
    # Generate LaTeX table with proper escaping
    # Use escape=False since we manually escaped the special characters
    latex_table = display_df.to_latex(index=False, escape=False)
    
    # Fix the column format - use a more robust approach to replace the tabular environment format
    column_format = ''.join(['c'] * len(display_df.columns))
    tabular_pattern = r'\\begin{tabular}{[^}]*}'
    replacement = f'\\begin{{tabular}}{{{column_format}}}'
    latex_table = re.sub(tabular_pattern, replacement, latex_table)
    
    # Fix table environment
    latex_table = latex_table.replace('\\begin{tabular}', '\\begin{table}[htbp]\n\\centering\n\\caption{Results Summary for ' + dataset_name + '}\n\\begin{tabular}')
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\n\\label{tab:results_' + dataset_name + '}\n\\end{table}')
    
    # Save LaTeX table to a file
    save_dir = ensure_save_dir(dataset_name)
    latex_file = os.path.join(save_dir, f"results_{dataset_name}_table.tex")
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {latex_file}")
    
    # Generate and save F1 by epoch plot after processing the results
    if len(results_df) > 0:
        # Generate the original overall plot
        plot_f1_by_epoch(results_df, dataset_name)
        
        # Generate the new comparison plots
        plot_model_comparison(results_df, dataset_name)
        plot_peft_comparison(results_df, dataset_name)
        plot_prompt_comparison(results_df, dataset_name)
    
    # After generating the summary table, create the relation comparison table only if requested
    if print_comparisons:
        print("\nLaTeX Table:")
        print(latex_table)
        
        # Find specific models for comparison - we want last epoch (10) of l_8i and q_7i with and without CoT
        cot_l8i_exp = None
        no_cot_l8i_exp = None
        cot_q7i_exp = None
        no_cot_q7i_exp = None
        
        # Create a copy of the dataframe with simpler filtering conditions
        filter_df = results_df.copy()
        
        # Find experiments with epoch 10 for all four model types
        for idx, row in filter_df.iterrows():
            model = row['model']
            cot = row['cot']
            epoch = row['peft_epochs']
            experiment_name = row['experiment_name']
            
            # Only consider epoch 10 experiments
            if epoch != 10:
                continue
                
            # Collect the experiment IDs
            if model == 'l_8i' and cot:
                cot_l8i_exp = experiment_name
            elif model == 'l_8i' and not cot:
                no_cot_l8i_exp = experiment_name
            elif model == 'q_7i' and cot:
                cot_q7i_exp = experiment_name
            elif model == 'q_7i' and not cot:
                no_cot_q7i_exp = experiment_name
        
        # Define target experiments using the found experiment IDs
        target_experiments = []
        
        if cot_l8i_exp:
            target_experiments.append((cot_l8i_exp, "l_8i with CoT (epoch 10)"))
        if no_cot_l8i_exp:
            target_experiments.append((no_cot_l8i_exp, "l_8i without CoT (epoch 10)"))
        if cot_q7i_exp:
            target_experiments.append((cot_q7i_exp, "q_7i with CoT (epoch 10)"))
        if no_cot_q7i_exp:
            target_experiments.append((no_cot_q7i_exp, "q_7i without CoT (epoch 10)"))
            
        # Add baseline experiments if needed
        target_experiments.append(("q_7i_0shot", "Prompting baseline (few-shot 5)"))
        target_experiments.append(("q_7i_2shot_syncot_temp1.0", "Prompting with CoT"))
        
        # Log which experiments we're using
        print("\nUsing the following experiments for comparison:")
        for exp_id, desc in target_experiments:
            print(f"  - {desc}: {exp_id}")
        
        # Create and print the relation-by-relation comparison (always using macro metrics)
        relation_comparison = create_relation_comparison_table(
            dataset_name, 
            target_experiments, 
            print_comparisons=True
        )
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    
    parser.add_argument("--dataset", type=str, default="dataset2024", 
                       help="Dataset name")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output CSV file")
    parser.add_argument("--max-days", type=int, default=None,
                       help="Maximum age in days for experiments to include")
    parser.add_argument("--print-comparisons", action="store_true")
    
    args = parser.parse_args()
    
    aggregate_experiment_results(args.dataset, args.output, args.max_days, args.print_comparisons)
