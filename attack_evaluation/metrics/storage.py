"""
Storage utilities for attack results and precompiled distances.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch


def save_precompiled_distances(attack_data: Dict[str, Any], threat_model: str, 
                              output_dir: str, format: str = 'npz') -> str:
    """
    Save precompiled attack distances and metadata for future analysis.
    
    Args:
        attack_data: Results from run_attack()
        threat_model: Threat model used
        output_dir: Output directory
        format: Storage format ('npz', 'pickle', 'json')
        
    Returns:
        Path to saved file
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare precompiled data
    precompiled_data = {
        'metadata': {
            'threat_model': threat_model,
            'ASR': attack_data.get('ASR', 0.0),
            'accuracy': attack_data.get('accuracy', 0.0),
            'total_samples': len(attack_data.get('adv_success', [])),
            'successful_attacks': sum(attack_data.get('adv_success', [])),
            'targeted': attack_data.get('targeted', False),
        },
        'distances': attack_data.get('distances', {}),
        'optimal_distances': attack_data.get('best_optim_distances', {}),
        'success_indicators': {
            'adv_success': attack_data.get('adv_success', []),
            'ori_success': attack_data.get('ori_success', []),
        },
        'sample_metadata': {
            'hashes': attack_data.get('hashes', []),
            'box_failures': attack_data.get('box_failures', []),
            'batch_failures': attack_data.get('batch_failures', []),
        },
        'performance_data': {
            'times': attack_data.get('times', []),
            'num_forwards': attack_data.get('num_forwards', []),
            'num_backwards': attack_data.get('num_backwards', []),
        }
    }
    
    # Save in requested format
    if format == 'npz':
        return _save_as_npz(precompiled_data, save_dir)
    elif format == 'pickle':
        return _save_as_pickle(precompiled_data, save_dir)
    elif format == 'json':
        return _save_as_json(precompiled_data, save_dir)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npz', 'pickle', or 'json'")


def _save_as_npz(data: Dict[str, Any], save_dir: Path) -> str:
    """Save data in compressed NumPy format"""
    file_path = save_dir / 'precompiled_distances.npz'
    
    # Flatten nested dictionaries for npz format
    save_arrays = {}
    
    # Metadata (save as JSON string)
    save_arrays['metadata'] = np.array([json.dumps(data['metadata'])], dtype=object)
    
    # Distances
    for norm, distances in data['distances'].items():
        save_arrays[f'distances_{norm}'] = np.array(distances)
    
    # Optimal distances
    for norm, distances in data['optimal_distances'].items():
        save_arrays[f'optimal_{norm}'] = np.array(distances)
    
    # Success indicators
    for key, values in data['success_indicators'].items():
        save_arrays[f'success_{key}'] = np.array(values, dtype=bool)
    
    # Sample metadata
    save_arrays['sample_hashes'] = np.array(data['sample_metadata']['hashes'], dtype=object)
    save_arrays['box_failures'] = np.array(data['sample_metadata']['box_failures'], dtype=bool)
    save_arrays['batch_failures'] = np.array(data['sample_metadata']['batch_failures'], dtype=bool)
    
    # Performance data
    for key, values in data['performance_data'].items():
        save_arrays[f'perf_{key}'] = np.array(values)
    
    np.savez_compressed(file_path, **save_arrays)
    
    # Also save readable metadata
    metadata_file = save_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(data['metadata'], f, indent=2)
    
    return str(file_path)


def _save_as_pickle(data: Dict[str, Any], save_dir: Path) -> str:
    """Save data in pickle format"""
    file_path = save_dir / 'precompiled_distances.pkl'
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return str(file_path)


def _save_as_json(data: Dict[str, Any], save_dir: Path) -> str:
    """Save data in JSON format (limited by JSON serialization)"""
    file_path = save_dir / 'precompiled_distances.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = _convert_for_json(data)
    
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return str(file_path)


def _convert_for_json(data: Any) -> Any:
    """Recursively convert data for JSON serialization"""
    if isinstance(data, dict):
        return {key: _convert_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_convert_for_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data


def load_precompiled_distances(file_path: str) -> Dict[str, Any]:
    """
    Load precompiled distances from file.
    
    Args:
        file_path: Path to precompiled distances file
        
    Returns:
        Dictionary with loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == '.npz':
        return _load_from_npz(file_path)
    elif file_path.suffix == '.pkl':
        return _load_from_pickle(file_path)
    elif file_path.suffix == '.json':
        return _load_from_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _load_from_npz(file_path: Path) -> Dict[str, Any]:
    """Load data from NPZ file"""
    with np.load(file_path, allow_pickle=True) as data:
        # Reconstruct original structure
        result = {
            'distances': {},
            'optimal_distances': {},
            'success_indicators': {},
            'sample_metadata': {},
            'performance_data': {},
        }
        
        # Load metadata
        if 'metadata' in data:
            result['metadata'] = json.loads(data['metadata'].item())
        
        # Load distances
        for key in data.keys():
            if key.startswith('distances_'):
                norm = key.replace('distances_', '')
                result['distances'][norm] = data[key].tolist()
            elif key.startswith('optimal_'):
                norm = key.replace('optimal_', '')
                result['optimal_distances'][norm] = data[key].tolist()
            elif key.startswith('success_'):
                success_key = key.replace('success_', '')
                result['success_indicators'][success_key] = data[key].tolist()
            elif key.startswith('perf_'):
                perf_key = key.replace('perf_', '')
                result['performance_data'][perf_key] = data[key].tolist()
            elif key == 'sample_hashes':
                result['sample_metadata']['hashes'] = data[key].tolist()
            elif key == 'box_failures':
                result['sample_metadata']['box_failures'] = data[key].tolist()
            elif key == 'batch_failures':
                result['sample_metadata']['batch_failures'] = data[key].tolist()
        
        return result


def _load_from_pickle(file_path: Path) -> Dict[str, Any]:
    """Load data from pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _load_from_json(file_path: Path) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_attack_comparison(results_list: List[Dict[str, Any]], attack_names: List[str],
                          output_dir: str, comparison_name: str = 'attack_comparison') -> str:
    """
    Save comparison of multiple attack results.
    
    Args:
        results_list: List of attack result dictionaries
        attack_names: Names of the attacks
        output_dir: Output directory
        comparison_name: Name for the comparison file
        
    Returns:
        Path to saved comparison file
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare comparison data
    comparison_data = {
        'metadata': {
            'num_attacks': len(results_list),
            'attack_names': attack_names,
            'comparison_timestamp': str(np.datetime64('now')),
        },
        'individual_results': {},
        'summary_statistics': {},
    }
    
    # Store individual results (subset of data)
    for i, (results, name) in enumerate(zip(results_list, attack_names)):
        comparison_data['individual_results'][name] = {
            'ASR': results.get('ASR', 0.0),
            'accuracy': results.get('accuracy', 0.0),
            'distances': results.get('distances', {}),
            'timing': {
                'total_time': sum(results.get('times', [])),
                'mean_time': np.mean(results.get('times', [])) if results.get('times') else 0,
            },
            'queries': {
                'mean_forwards': np.mean(results.get('num_forwards', [])) if results.get('num_forwards') else 0,
                'mean_backwards': np.mean(results.get('num_backwards', [])) if results.get('num_backwards') else 0,
            }
        }
    
    # Compute summary statistics
    all_asrs = [results.get('ASR', 0.0) for results in results_list]
    comparison_data['summary_statistics'] = {
        'mean_ASR': float(np.mean(all_asrs)),
        'std_ASR': float(np.std(all_asrs)),
        'max_ASR': float(np.max(all_asrs)),
        'min_ASR': float(np.min(all_asrs)),
        'best_attack': attack_names[np.argmax(all_asrs)] if all_asrs else None,
        'worst_attack': attack_names[np.argmin(all_asrs)] if all_asrs else None,
    }
    
    # Save comparison
    file_path = save_dir / f'{comparison_name}.json'
    with open(file_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    return str(file_path)


def create_attack_report(attack_data: Dict[str, Any], attack_name: str, 
                        output_dir: str, include_plots: bool = True) -> str:
    """
    Create comprehensive report for a single attack.
    
    Args:
        attack_data: Results from run_attack()
        attack_name: Name of the attack
        output_dir: Output directory
        include_plots: Whether to generate plots
        
    Returns:
        Path to saved report
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report data
    report = {
        'attack_info': {
            'name': attack_name,
            'report_timestamp': str(np.datetime64('now')),
        },
        'summary_metrics': {
            'ASR': attack_data.get('ASR', 0.0),
            'accuracy': attack_data.get('accuracy', 0.0),
            'total_samples': len(attack_data.get('adv_success', [])),
            'successful_attacks': sum(attack_data.get('adv_success', [])),
            'targeted': attack_data.get('targeted', False),
        },
        'distance_analysis': {},
        'performance_analysis': {
            'total_runtime': sum(attack_data.get('times', [])),
            'mean_batch_time': np.mean(attack_data.get('times', [])) if attack_data.get('times') else 0,
            'mean_queries_forward': np.mean(attack_data.get('num_forwards', [])) if attack_data.get('num_forwards') else 0,
            'mean_queries_backward': np.mean(attack_data.get('num_backwards', [])) if attack_data.get('num_backwards') else 0,
        },
        'failure_analysis': {
            'box_constraint_failures': sum(attack_data.get('box_failures', [])),
            'batch_failures': sum(attack_data.get('batch_failures', [])),
        }
    }
    
    # Analyze distances for each norm
    distances = attack_data.get('distances', {})
    for norm, dist_values in distances.items():
        if dist_values:
            dist_array = np.array(dist_values)
            report['distance_analysis'][norm] = {
                'mean': float(np.mean(dist_array)),
                'median': float(np.median(dist_array)),
                'std': float(np.std(dist_array)),
                'min': float(np.min(dist_array)),
                'max': float(np.max(dist_array)),
                'p95': float(np.percentile(dist_array, 95)),
            }
    
    # Save report
    report_file = save_dir / f'{attack_name}_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate plots if requested
    if include_plots:
        try:
            import matplotlib.pyplot as plt
            _generate_attack_plots(attack_data, attack_name, save_dir)
        except ImportError:
            print("Warning: matplotlib not available, skipping plots")
    
    return str(report_file)


def _generate_attack_plots(attack_data: Dict[str, Any], attack_name: str, save_dir: Path):
    """Generate plots for attack analysis"""
    import matplotlib.pyplot as plt
    
    # Distance histograms
    distances = attack_data.get('distances', {})
    if distances:
        fig, axes = plt.subplots(1, len(distances), figsize=(5*len(distances), 4))
        if len(distances) == 1:
            axes = [axes]
        
        for i, (norm, dist_values) in enumerate(distances.items()):
            if dist_values:
                # Only plot non-zero distances (successful attacks)
                non_zero_distances = [d for d in dist_values if d > 0]
                if non_zero_distances:
                    axes[i].hist(non_zero_distances, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_xlabel(f'{norm.upper()} Distance')
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'{norm.upper()} Distance Distribution')
                    axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'{attack_name} - Distance Distributions')
        plt.tight_layout()
        plt.savefig(save_dir / f'{attack_name}_distances.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Robust accuracy curve
    if 'robust_accuracy_curve' in attack_data:
        curve = attack_data['robust_accuracy_curve']
        if curve and 'thresholds' in curve:
            plt.figure(figsize=(8, 6))
            plt.plot(curve['thresholds'], curve['robust_accuracies'], 'b-', linewidth=2)
            plt.xlabel('Perturbation Threshold')
            plt.ylabel('Robust Accuracy')
            plt.title(f'{attack_name} - Robust Accuracy Curve')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(save_dir / f'{attack_name}_robust_curve.png', dpi=150, bbox_inches='tight')
            plt.close()