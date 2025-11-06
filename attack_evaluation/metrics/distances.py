"""
Distance computation and statistics for attack results.
"""
import numpy as np
from typing import Dict, List, Any, Optional


def compute_distance_statistics(distances: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a list of distances.
    
    Args:
        distances: List of distance values
        
    Returns:
        Dictionary with distance statistics
    """
    if not distances:
        return {
            'mean_distance': 0.0,
            'median_distance': 0.0,
            'std_distance': 0.0,
            'min_distance': 0.0,
            'max_distance': 0.0,
            'p25_distance': 0.0,
            'p75_distance': 0.0,
            'p95_distance': 0.0,
        }
    
    dist_array = np.array(distances)
    
    stats = {
        'mean_distance': float(np.mean(dist_array)),
        'median_distance': float(np.median(dist_array)),
        'std_distance': float(np.std(dist_array)),
        'min_distance': float(np.min(dist_array)),
        'max_distance': float(np.max(dist_array)),
        'p25_distance': float(np.percentile(dist_array, 25)),
        'p75_distance': float(np.percentile(dist_array, 75)),
        'p95_distance': float(np.percentile(dist_array, 95)),
    }
    
    # Additional statistics for non-zero distances
    non_zero_distances = dist_array[dist_array > 0]
    if len(non_zero_distances) > 0:
        stats.update({
            'mean_nonzero_distance': float(np.mean(non_zero_distances)),
            'median_nonzero_distance': float(np.median(non_zero_distances)),
            'fraction_nonzero': float(len(non_zero_distances) / len(dist_array)),
        })
    else:
        stats.update({
            'mean_nonzero_distance': 0.0,
            'median_nonzero_distance': 0.0,
            'fraction_nonzero': 0.0,
        })
    
    return stats


def compute_pairwise_distances(distances1: List[float], distances2: List[float]) -> Dict[str, float]:
    """
    Compute pairwise comparison statistics between two distance arrays.
    
    Args:
        distances1: First set of distances
        distances2: Second set of distances
        
    Returns:
        Dictionary with comparison statistics
    """
    if len(distances1) != len(distances2):
        raise ValueError(f"Distance arrays must have same length: {len(distances1)} vs {len(distances2)}")
    
    if not distances1 or not distances2:
        return {}
    
    d1 = np.array(distances1)
    d2 = np.array(distances2)
    
    # Only compare where both attacks succeeded (distance > 0)
    valid_mask = (d1 > 0) & (d2 > 0)
    
    if not valid_mask.any():
        return {
            'correlation': 0.0,
            'mean_ratio': float('nan'),
            'median_ratio': float('nan'),
            'valid_comparisons': 0,
        }
    
    d1_valid = d1[valid_mask]
    d2_valid = d2[valid_mask]
    
    # Compute statistics
    ratios = d1_valid / d2_valid
    correlation = float(np.corrcoef(d1_valid, d2_valid)[0, 1])
    
    return {
        'correlation': correlation,
        'mean_ratio': float(np.mean(ratios)),
        'median_ratio': float(np.median(ratios)),
        'std_ratio': float(np.std(ratios)),
        'valid_comparisons': int(valid_mask.sum()),
        'total_comparisons': len(distances1),
    }


def compute_distance_distribution(distances: List[float], bins: int = 50) -> Dict[str, Any]:
    """
    Compute distance distribution histogram.
    
    Args:
        distances: List of distance values
        bins: Number of histogram bins
        
    Returns:
        Dictionary with histogram data
    """
    if not distances:
        return {
            'bin_edges': [],
            'counts': [],
            'bin_centers': [],
        }
    
    dist_array = np.array(distances)
    
    # Remove zero distances for histogram (failed attacks)
    non_zero_distances = dist_array[dist_array > 0]
    
    if len(non_zero_distances) == 0:
        return {
            'bin_edges': [],
            'counts': [],
            'bin_centers': [],
        }
    
    counts, bin_edges = np.histogram(non_zero_distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        'bin_edges': bin_edges.tolist(),
        'counts': counts.tolist(),
        'bin_centers': bin_centers.tolist(),
        'total_nonzero': len(non_zero_distances),
        'total_zero': len(dist_array) - len(non_zero_distances),
    }


def compute_optimality_score(test_distances: List[float], reference_distances: List[float]) -> float:
    """
    Compute optimality score as ratio of test to reference distances.
    
    Args:
        test_distances: Distances from test attack
        reference_distances: Distances from reference attack
        
    Returns:
        Optimality score (1.0 = optimal, >1.0 = sub-optimal)
    """
    if len(test_distances) != len(reference_distances):
        raise ValueError("Distance arrays must have same length")
    
    test_arr = np.array(test_distances)
    ref_arr = np.array(reference_distances)
    
    # Only compare successful attacks
    valid_mask = (test_arr > 0) & (ref_arr > 0)
    
    if not valid_mask.any():
        return float('nan')
    
    ratios = test_arr[valid_mask] / ref_arr[valid_mask]
    return float(np.mean(ratios))


def compute_attack_efficiency(distances: List[float], num_queries: List[int]) -> Dict[str, float]:
    """
    Compute attack efficiency metrics combining distance and query count.
    
    Args:
        distances: List of distance values
        num_queries: List of query counts per sample
        
    Returns:
        Dictionary with efficiency metrics
    """
    if len(distances) != len(num_queries):
        raise ValueError("Distance and query arrays must have same length")
    
    if not distances:
        return {}
    
    dist_array = np.array(distances)
    query_array = np.array(num_queries)
    
    # Successful attacks only
    success_mask = dist_array > 0
    
    if not success_mask.any():
        return {
            'mean_queries_success': 0.0,
            'efficiency_score': 0.0,
            'query_distance_correlation': 0.0,
        }
    
    successful_distances = dist_array[success_mask]
    successful_queries = query_array[success_mask]
    
    # Efficiency score: lower distance with fewer queries is better
    # Normalize both metrics and compute harmonic mean
    norm_distances = successful_distances / np.max(successful_distances)
    norm_queries = successful_queries / np.max(successful_queries)
    
    # Efficiency = 1 / (weighted sum of normalized distance and queries)
    efficiency_scores = 1 / (0.5 * norm_distances + 0.5 * norm_queries + 1e-8)
    
    correlation = 0.0
    if len(successful_distances) > 1:
        correlation = float(np.corrcoef(successful_distances, successful_queries)[0, 1])
    
    return {
        'mean_queries_success': float(np.mean(successful_queries)),
        'median_queries_success': float(np.median(successful_queries)),
        'efficiency_score': float(np.mean(efficiency_scores)),
        'query_distance_correlation': correlation,
    }