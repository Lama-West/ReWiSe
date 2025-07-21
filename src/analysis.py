import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
import re
from datetime import datetime


class ExperimentLoader:
    """Class for loading and processing experiment results."""
    
    def __init__(self, base_path: str = None):
        """
        Initialize the experiment loader.
        
        Args:
            base_path: Base path to experiments (default: auto-detect)
        """
        if base_path is None:
            # Try to auto-detect base path
            current_dir = Path.cwd()
            if (current_dir / "predictions").exists():
                self.base_path = current_dir
            elif (current_dir.parent / "predictions").exists():
                self.base_path = current_dir.parent
            else:
                self.base_path = current_dir
        else:
            self.base_path = Path(base_path)
    
    def load_inference_experiment(self, experiment_path: Union[str, Path]) -> Dict:
        """
        Load results from an inference experiment.
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Dictionary containing config, results_string, and results_disambiguated
        """
        exp_path = Path(experiment_path)
        
        result = {
            'experiment_path': str(exp_path),
            'experiment_name': exp_path.name,
            'config': None,
            'results_string': None,
            'results_disambiguated': None,
            'experiment_type': 'inference',
            'metadata': {}
        }
        
        # Load config
        config_path = exp_path / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                result['config'] = yaml.safe_load(f)
                result['metadata'] = self._extract_inference_metadata(result['config'])
        
        # Load string-based results
        results_string_path = exp_path / "results_string.json"
        if results_string_path.exists():
            with open(results_string_path, 'r') as f:
                result['results_string'] = json.load(f)
        
        # Load disambiguated results
        results_path = exp_path / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                result['results_disambiguated'] = json.load(f)
        
        return result
    
    def load_training_experiment(self, experiment_path: Union[str, Path]) -> Dict:
        """
        Load results from a training experiment (multiple epochs).
        
        Args:
            experiment_path: Path to training experiment directory
            
        Returns:
            Dictionary containing config and results for all epochs
        """
        exp_path = Path(experiment_path)
        
        result = {
            'experiment_path': str(exp_path),
            'experiment_name': exp_path.name,
            'config': None,
            'epochs': {},
            'experiment_type': 'training',
            'metadata': {}
        }
        
        # Load config
        config_path = exp_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                result['config'] = json.load(f)
                result['metadata'] = self._extract_training_metadata(result['config'])
        
        # Find all epoch directories
        epoch_dirs = [d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('epoch_')]
        
        if epoch_dirs:
            for epoch_dir in sorted(epoch_dirs):
                epoch_num = int(epoch_dir.name.split('_')[1])
                epoch_data = {
                    'epoch': epoch_num,
                    'results_string': None,
                    'results_disambiguated': None
                }
                
                # Load string-based results for this epoch
                results_string_path = epoch_dir / "results_string.json"
                if results_string_path.exists():
                    with open(results_string_path, 'r') as f:
                        epoch_data['results_string'] = json.load(f)
                
                # Load disambiguated results for this epoch
                results_path = epoch_dir / "results.json"
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        epoch_data['results_disambiguated'] = json.load(f)
                
                result['epochs'][epoch_num] = epoch_data
        else:
            # If no epoch directories found, check for results in the main directory
            # This handles training experiments without explicit epoch subdirectories
            epoch_data = {
                'epoch': 'final',
                'results_string': None,
                'results_disambiguated': None
            }
            
            # Load string-based results from main directory
            results_string_path = exp_path / "results_string.json"
            if results_string_path.exists():
                with open(results_string_path, 'r') as f:
                    epoch_data['results_string'] = json.load(f)
            
            # Load disambiguated results from main directory
            results_path = exp_path / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    epoch_data['results_disambiguated'] = json.load(f)
            
            if epoch_data['results_string'] or epoch_data['results_disambiguated']:
                result['epochs']['final'] = epoch_data
        
        return result
    
    def _extract_inference_metadata(self, config: Dict) -> Dict:
        """Extract metadata from inference experiment config."""
        metadata = {}
        
        if config:
            # Basic experiment info
            metadata['experiment_name'] = config.get('experiment', {}).get('name', '')
            metadata['report'] = config.get('experiment', {}).get('report', '')
            
            # Model info
            model_config = config.get('model', {})
            metadata['model_alias'] = model_config.get('alias', '')
            metadata['use_peft'] = model_config.get('use_peft', False)
            metadata['quantize'] = model_config.get('quantize', False)
            metadata['use_vllm'] = model_config.get('use_vllm', True)
            
            # Dataset info
            dataset_config = config.get('dataset', {})
            metadata['dataset'] = dataset_config.get('name', '')
            metadata['split'] = dataset_config.get('split', 'val')
            metadata['limit'] = dataset_config.get('limit', 0)
            
            # Prompting info
            prompting_config = config.get('prompting', {})
            metadata['few_shot'] = prompting_config.get('few_shot', 0)
            metadata['cot'] = prompting_config.get('cot', False)
            metadata['prompt_file'] = prompting_config.get('prompt_file', '')
            metadata['system_message'] = prompting_config.get('system_message', '')
            
            # Generation info
            generation_config = config.get('generation', {})
            metadata['temperature'] = generation_config.get('temperature', 1.0)
            metadata['max_tokens'] = generation_config.get('max_tokens', 100)
            metadata['n_consistency'] = generation_config.get('n_consistency', 1)
            
            # Job info
            job_config = config.get('job_name', {})
            metadata['launcher_job'] = job_config.get('launcher', '')
            metadata['experiment_job'] = job_config.get('experiment', '')
            
            # Parse timestamp from experiment name
            timestamp_match = re.match(r'(\d{8}_\d{6})', metadata['experiment_name'])
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                try:
                    metadata['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                except ValueError:
                    metadata['timestamp'] = None
            else:
                metadata['timestamp'] = None
        
        return metadata
    
    def _extract_training_metadata(self, config: Dict) -> Dict:
        """Extract metadata from training experiment config."""
        metadata = {}
        
        if config:
            # Basic info
            metadata['model'] = config.get('model', '')
            metadata['dataset'] = config.get('dataset', '')
            metadata['peft_method'] = config.get('peft_method', '')
            metadata['learning_rate'] = config.get('learning_rate', 0)
            metadata['batch_size'] = config.get('batch_size', 0)
            metadata['num_epochs'] = config.get('num_epochs', 0)
            metadata['max_seq_length'] = config.get('max_seq_length', 0)
            
            # Parse timestamp from directory name if available
            exp_name = Path(config.get('output_dir', '')).name
            timestamp_match = re.match(r'(\d{8}_\d{6})', exp_name)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                try:
                    metadata['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                except ValueError:
                    metadata['timestamp'] = None
            else:
                metadata['timestamp'] = None
        
        return metadata
    
    def load_experiment(self, experiment_path: Union[str, Path]) -> Dict:
        """
        Auto-detect and load either inference or training experiment.
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Dictionary containing experiment data
        """
        exp_path = Path(experiment_path)
        
        # Check if it's a training experiment (has config.json and epoch dirs)
        if (exp_path / "config.json").exists() and any(d.name.startswith('epoch_') for d in exp_path.iterdir() if d.is_dir()):
            return self.load_training_experiment(exp_path)
        else:
            return self.load_inference_experiment(exp_path)
    
    def find_experiments(self, dataset: str = None, experiment_type: str = None) -> List[Path]:
        """
        Find all experiments in the workspace.
        
        Args:
            dataset: Filter by dataset name (optional)
            experiment_type: 'inference' or 'training' (optional)
            
        Returns:
            List of experiment paths
        """
        experiments = []
        
        # Look for inference experiments in predictions/
        if experiment_type is None or experiment_type == 'inference':
            predictions_dir = self.base_path / "predictions"
            if predictions_dir.exists():
                for dataset_dir in predictions_dir.iterdir():
                    if dataset_dir.is_dir() and (dataset is None or dataset_dir.name == dataset):
                        for exp_dir in dataset_dir.iterdir():
                            if exp_dir.is_dir() and (exp_dir / "config.yaml").exists():
                                experiments.append(exp_dir)
        
        # Look for training experiments in trained_models/
        if experiment_type is None or experiment_type == 'training':
            trained_models_dir = self.base_path / "trained_models"
            if trained_models_dir.exists():
                for exp_dir in trained_models_dir.iterdir():
                    if exp_dir.is_dir() and (exp_dir / "config.json").exists():
                        experiments.append(exp_dir)
        
        return sorted(experiments)
    
    def load_all_experiments(self, dataset: str = None, experiment_type: str = None) -> List[Dict]:
        """
        Load all experiments in the workspace.
        
        Args:
            dataset: Filter by dataset name (optional)
            experiment_type: 'inference' or 'training' (optional)
            
        Returns:
            List of experiment dictionaries
        """
        experiment_paths = self.find_experiments(dataset, experiment_type)
        experiments = []
        
        for exp_path in experiment_paths:
            try:
                exp_data = self.load_experiment(exp_path)
                experiments.append(exp_data)
            except Exception as e:
                print(f"Warning: Failed to load experiment {exp_path}: {e}")
                continue
        
        return experiments


class ResultsAnalyzer:
    """Class for analyzing and aggregating experiment results."""
    
    def __init__(self, experiments: List[Dict]):
        """
        Initialize the results analyzer.
        
        Args:
            experiments: List of experiment dictionaries from ExperimentLoader
        """
        self.experiments = experiments
    
    def to_dataframe(self, result_type: str = 'string', metric: str = 'macro-f1') -> pd.DataFrame:
        """
        Convert experiment results to a pandas DataFrame for analysis.
        
        Args:
            result_type: 'string' or 'disambiguated'
            metric: Which metric to extract ('macro-f1', 'macro-p', 'macro-r', etc.)
            
        Returns:
            DataFrame with experiments as rows and relations as columns
        """
        rows = []
        
        for exp in self.experiments:
            if exp['experiment_type'] == 'inference':
                results_key = f'results_{result_type}'
                if results_key in exp and exp[results_key]:
                    row = exp['metadata'].copy()
                    
                    # Add relation-specific metrics
                    for relation, metrics in exp[results_key].items():
                        if relation != "*** All Relations ***":
                            row[f"{relation}_{metric}"] = metrics.get(metric, None)
                    
                    # Add overall metric
                    if "*** All Relations ***" in exp[results_key]:
                        row[f"overall_{metric}"] = exp[results_key]["*** All Relations ***"].get(metric, None)
                    
                    rows.append(row)
            
            elif exp['experiment_type'] == 'training':
                # For training experiments, include each epoch as a separate row
                for epoch_num, epoch_data in exp['epochs'].items():
                    results_key = f'results_{result_type}'
                    if results_key in epoch_data and epoch_data[results_key]:
                        row = exp['metadata'].copy()
                        row['epoch'] = epoch_num
                        
                        # Add relation-specific metrics
                        for relation, metrics in epoch_data[results_key].items():
                            if relation != "*** All Relations ***":
                                row[f"{relation}_{metric}"] = metrics.get(metric, None)
                        
                        # Add overall metric
                        if "*** All Relations ***" in epoch_data[results_key]:
                            row[f"overall_{metric}"] = epoch_data[results_key]["*** All Relations ***"].get(metric, None)
                        
                        rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary_stats(self, result_type: str = 'string') -> pd.DataFrame:
        """
        Get summary statistics across all experiments.
        
        Args:
            result_type: 'string' or 'disambiguated'
            
        Returns:
            DataFrame with summary statistics
        """
        df = self.to_dataframe(result_type=result_type, metric='macro-f1')
        
        # Get relation columns
        relation_cols = [col for col in df.columns if col.endswith('_macro-f1') and not col.startswith('overall')]
        
        if not relation_cols:
            return pd.DataFrame()
        
        # Calculate summary stats
        summary = df[relation_cols].describe()
        
        # Add experiment counts by configuration
        if 'model_alias' in df.columns:
            model_counts = df['model_alias'].value_counts()
            summary.loc['model_counts'] = [model_counts.to_dict() for _ in range(len(summary.columns))]
        
        return summary
    
    def compare_experiments(self, group_by: List[str], result_type: str = 'string', metric: str = 'macro-f1') -> pd.DataFrame:
        """
        Compare experiments grouped by specified criteria.
        
        Args:
            group_by: List of metadata fields to group by (e.g., ['model_alias', 'cot'])
            result_type: 'string' or 'disambiguated'
            metric: Which metric to compare
            
        Returns:
            DataFrame with grouped comparisons
        """
        df = self.to_dataframe(result_type=result_type, metric=metric)
        
        if df.empty:
            return pd.DataFrame()
        
        # Get overall metric column
        overall_col = f"overall_{metric}"
        if overall_col not in df.columns:
            return pd.DataFrame()
        
        # Group by specified fields and calculate statistics
        comparison = df.groupby(group_by)[overall_col].agg(['mean', 'std', 'count']).round(3)
        
        return comparison
    
    def print_latex_table(self, experiment_path: Union[str, Path], result_type: str = 'string', 
                          caption: str = None, label: str = None, precision: int = 3, 
                          only_macro: bool = False) -> str:
        """
        Generate a LaTeX table from a specific experiment's results.
        
        Args:
            experiment_path: Path to the experiment directory
            result_type: 'string' or 'disambiguated'
            caption: Table caption (optional)
            label: Table label for referencing (optional)
            precision: Number of decimal places to show
            only_macro: If True, only show macro-p, macro-r, macro-f1 columns
            
        Returns:
            String containing the LaTeX table code
        """
        # Find the experiment in our loaded experiments
        exp_path_str = str(Path(experiment_path).resolve())
        target_experiment = None
        
        for exp in self.experiments:
            if str(Path(exp['experiment_path']).resolve()) == exp_path_str:
                target_experiment = exp
                break
        
        if target_experiment is None:
            raise ValueError(f"Experiment not found: {experiment_path}")
        
        # Get the results data
        results_key = f'results_{result_type}'
        if target_experiment['experiment_type'] == 'inference':
            if results_key not in target_experiment or target_experiment[results_key] is None:
                raise ValueError(f"No {result_type} results found for experiment")
            results_data = target_experiment[results_key]
        else:
            # For training experiments, use the final epoch
            if 'final' in target_experiment['epochs']:
                epoch_data = target_experiment['epochs']['final']
            else:
                # Use the last epoch
                last_epoch = max(target_experiment['epochs'].keys())
                epoch_data = target_experiment['epochs'][last_epoch]
            
            if results_key not in epoch_data or epoch_data[results_key] is None:
                raise ValueError(f"No {result_type} results found for experiment")
            results_data = epoch_data[results_key]
        
        # Create LaTeX table
        latex_lines = []
        
        # Table header
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        
        # Column specification
        if only_macro:
            num_cols = 3  # macro-p, macro-r, macro-f1
            col_spec = "l" + "c" * num_cols
            headers = ["Relation", "macro-p", "macro-r", "macro-f1"]
        else:
            num_cols = 8  # macro-p, macro-r, macro-f1, micro-p, micro-r, micro-f1, avg. #preds, #empty preds
            col_spec = "l" + "c" * num_cols
            headers = ["Relation", "macro-p", "macro-r", "macro-f1", "micro-p", "micro-r", "micro-f1", "avg. \\#preds", "\\#empty preds"]
        
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\toprule")
        
        # Header row
        latex_lines.append(" & ".join(headers) + " \\\\")
        latex_lines.append("\\midrule")
        
        # Data rows
        for relation, metrics in results_data.items():
            if relation == "*** All Relations ***":
                continue  # Skip for now, will add at the end
            
            # Format relation name (escape underscores for LaTeX)
            relation_name = relation.replace("_", "\\_")
            
            # Extract metrics with proper formatting
            macro_p = f"{metrics.get('macro-p', 0):.{precision}f}"
            macro_r = f"{metrics.get('macro-r', 0):.{precision}f}"
            macro_f1 = f"{metrics.get('macro-f1', 0):.{precision}f}"
            
            if only_macro:
                row = [relation_name, macro_p, macro_r, macro_f1]
            else:
                micro_p = f"{metrics.get('micro-p', 0):.{precision}f}"
                micro_r = f"{metrics.get('micro-r', 0):.{precision}f}"
                micro_f1 = f"{metrics.get('micro-f1', 0):.{precision}f}"
                avg_preds = f"{metrics.get('avg. #preds', 0):.{precision}f}"
                empty_preds = f"{int(metrics.get('#empty preds', 0))}"
                row = [relation_name, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, avg_preds, empty_preds]
            
            latex_lines.append(" & ".join(row) + " \\\\")
        
        # Add overall results if available
        if "*** All Relations ***" in results_data:
            latex_lines.append("\\midrule")
            metrics = results_data["*** All Relations ***"]
            
            macro_p = f"{metrics.get('macro-p', 0):.{precision}f}"
            macro_r = f"{metrics.get('macro-r', 0):.{precision}f}"
            macro_f1 = f"{metrics.get('macro-f1', 0):.{precision}f}"
            
            if only_macro:
                row = ["\\textbf{All Relations}", macro_p, macro_r, macro_f1]
            else:
                micro_p = f"{metrics.get('micro-p', 0):.{precision}f}"
                micro_r = f"{metrics.get('micro-r', 0):.{precision}f}"
                micro_f1 = f"{metrics.get('micro-f1', 0):.{precision}f}"
                avg_preds = f"{metrics.get('avg. #preds', 0):.{precision}f}"
                empty_preds = f"{int(metrics.get('#empty preds', 0))}"
                row = ["\\textbf{All Relations}", macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, avg_preds, empty_preds]
            
            latex_lines.append(" & ".join(row) + " \\\\")
        
        # Table footer
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        
        # Caption and label
        if caption:
            latex_lines.append(f"\\caption{{{caption}}}")
        if label:
            latex_lines.append(f"\\label{{{label}}}")
        
        latex_lines.append("\\end{table}")
        
        # Join and return
        latex_table = "\n".join(latex_lines)
        
        # Print the table
        print(latex_table)
        
        return latex_table

def load_experiment(experiment_path: Union[str, Path]) -> Dict:
    """Load a single experiment (inference or training)."""
    loader = ExperimentLoader()
    return loader.load_experiment(experiment_path)

def load_all_experiments(dataset: str = None, experiment_type: str = None) -> List[Dict]:
    """Load all experiments in the workspace."""
    loader = ExperimentLoader()
    return loader.load_all_experiments(dataset, experiment_type)

def analyze_experiments(experiments: List[Dict]) -> ResultsAnalyzer:
    """Create a results analyzer for the given experiments."""
    return ResultsAnalyzer(experiments)

def quick_analysis(dataset: str = None, experiment_type: str = None, result_type: str = 'string', metric: str = 'macro-f1') -> pd.DataFrame:
    """Quick analysis of all experiments."""
    experiments = load_all_experiments(dataset, experiment_type)
    analyzer = analyze_experiments(experiments)
    return analyzer.to_dataframe(result_type=result_type, metric=metric)