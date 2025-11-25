# Directory structure

import datetime
from pathlib import Path
from .logging import log_info

def create_output_dir(base_dir):
    """
    Create hierarchical output directory structure.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        dict: Dictionary containing all directory paths
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"spc_analysis_{timestamp}"
    
    # Create main directories
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    # Analysis subdirectories
    mad_dir = analysis_dir / 'mad_analysis'
    mad_dir.mkdir(exist_ok=True)
    
    dmso_dist_dir = analysis_dir / 'dmso_distances'
    dmso_dist_dir.mkdir(exist_ok=True)
    
    landmark_dist_dir = analysis_dir / 'landmark_distances'
    landmark_dist_dir.mkdir(exist_ok=True)
    
    # Visualization directories
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Add correlation plots directory
    correlation_dir = viz_dir / 'correlation'
    correlation_dir.mkdir(exist_ok=True)
    
    # DMSO distributions with improved structure
    dmso_dist_viz_dir = viz_dir / 'dmso_distributions'
    dmso_dist_viz_dir.mkdir(exist_ok=True)
    dmso_dist_viz_dir_combined = dmso_dist_viz_dir / 'combined'
    dmso_dist_viz_dir_combined.mkdir(exist_ok=True)
    dmso_dist_viz_dir_library = dmso_dist_viz_dir / 'by_library'
    dmso_dist_viz_dir_library.mkdir(exist_ok=True)
    
    # Histograms with improved structure
    histogram_dir = viz_dir / 'histograms'
    histogram_dir.mkdir(exist_ok=True)
    
    # General breakdowns
    histogram_dir_library = histogram_dir / 'by_library'
    histogram_dir_library.mkdir(exist_ok=True)
    histogram_dir_moa = histogram_dir / 'by_moa'
    histogram_dir_moa.mkdir(exist_ok=True)
    
    # Histogram folders by metric type
    hist_dispersion_dir = histogram_dir / 'dispersion_metrics'
    hist_dispersion_dir.mkdir(exist_ok=True)
    hist_dispersion_reference_dir = hist_dispersion_dir / 'reference'
    hist_dispersion_reference_dir.mkdir(exist_ok=True)
    hist_dispersion_test_dir = hist_dispersion_dir / 'test'
    hist_dispersion_test_dir.mkdir(exist_ok=True)
    hist_dispersion_comparison_dir = hist_dispersion_dir / 'comparisons'
    hist_dispersion_comparison_dir.mkdir(exist_ok=True)
    
    hist_dmso_dir = histogram_dir / 'dmso_distances'
    hist_dmso_dir.mkdir(exist_ok=True)
    hist_dmso_reference_dir = hist_dmso_dir / 'reference'
    hist_dmso_reference_dir.mkdir(exist_ok=True)
    hist_dmso_test_dir = hist_dmso_dir / 'test'
    hist_dmso_test_dir.mkdir(exist_ok=True)
    hist_dmso_comparison_dir = hist_dmso_dir / 'comparisons'
    hist_dmso_comparison_dir.mkdir(exist_ok=True)
    
    hist_landmark_dir = histogram_dir / 'landmark_distances'
    hist_landmark_dir.mkdir(exist_ok=True)
    hist_landmark_reference_dir = hist_landmark_dir / 'reference'
    hist_landmark_reference_dir.mkdir(exist_ok=True)
    hist_landmark_test_dir = hist_landmark_dir / 'test'
    hist_landmark_test_dir.mkdir(exist_ok=True)
    hist_landmark_comparison_dir = hist_landmark_dir / 'comparisons'
    hist_landmark_comparison_dir.mkdir(exist_ok=True)
    
    hist_scores_dir = histogram_dir / 'scores'
    hist_scores_dir.mkdir(exist_ok=True)
    hist_scores_reference_dir = hist_scores_dir / 'reference'
    hist_scores_reference_dir.mkdir(exist_ok=True)
    hist_scores_test_dir = hist_scores_dir / 'test'
    hist_scores_test_dir.mkdir(exist_ok=True)
    hist_scores_comparison_dir = hist_scores_dir / 'comparisons'
    hist_scores_comparison_dir.mkdir(exist_ok=True)
    
    # Landmark plots with improved structure
    landmark_viz_dir = viz_dir / 'landmark_plots'
    landmark_viz_dir.mkdir(exist_ok=True)
    
    # Combined plots folder
    landmark_combined_dir = landmark_viz_dir / 'combined'
    landmark_combined_dir.mkdir(exist_ok=True)
    landmark_combined_reference_dir = landmark_combined_dir / 'reference'
    landmark_combined_reference_dir.mkdir(exist_ok=True)
    landmark_combined_test_dir = landmark_combined_dir / 'test'
    landmark_combined_test_dir.mkdir(exist_ok=True)
    
    # Metric-specific folders
    landmark_by_metric_dir = landmark_viz_dir / 'by_metric'
    landmark_by_metric_dir.mkdir(exist_ok=True)
    
    # MAD cosine
    landmark_mad_dir = landmark_by_metric_dir / 'mad_cosine'
    landmark_mad_dir.mkdir(exist_ok=True)
    landmark_mad_interactive_dir = landmark_mad_dir / 'interactive'
    landmark_mad_interactive_dir.mkdir(exist_ok=True)
    landmark_mad_static_dir = landmark_mad_dir / 'static'
    landmark_mad_static_dir.mkdir(exist_ok=True)
    
    # VAR cosine
    landmark_var_dir = landmark_by_metric_dir / 'var_cosine'
    landmark_var_dir.mkdir(exist_ok=True)
    landmark_var_interactive_dir = landmark_var_dir / 'interactive'
    landmark_var_interactive_dir.mkdir(exist_ok=True)
    landmark_var_static_dir = landmark_var_dir / 'static'
    landmark_var_static_dir.mkdir(exist_ok=True)
    
    # STD cosine
    landmark_std_dir = landmark_by_metric_dir / 'std_cosine'
    landmark_std_dir.mkdir(exist_ok=True)
    landmark_std_interactive_dir = landmark_std_dir / 'interactive'
    landmark_std_interactive_dir.mkdir(exist_ok=True)
    landmark_std_static_dir = landmark_std_dir / 'static'
    landmark_std_static_dir.mkdir(exist_ok=True)
    
    # Keep legacy paths for backward compatibility
    landmark_viz_static_dir = landmark_viz_dir / 'static'
    landmark_viz_static_dir.mkdir(exist_ok=True)
    landmark_viz_interactive_dir = landmark_viz_dir / 'interactive'
    landmark_viz_interactive_dir.mkdir(exist_ok=True)
    
    landmark_selection_dir = viz_dir / 'landmark_selection'
    landmark_selection_dir.mkdir(exist_ok=True)
    
    # Add UMAP and TSNE visualization directories
    dimensionality_reduction_dir = viz_dir / 'dimensionality_reduction'
    dimensionality_reduction_dir.mkdir(exist_ok=True)
    umap_dir = dimensionality_reduction_dir / 'umap'
    umap_dir.mkdir(exist_ok=True)
    tsne_dir = dimensionality_reduction_dir / 'tsne'
    tsne_dir.mkdir(exist_ok=True)

    # After creating other visualization directories, add this section:
    # DMSO vs Metrics plots
    dmso_vs_metrics_dir = viz_dir / 'dmso_vs_metrics'
    dmso_vs_metrics_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_static_dir = dmso_vs_metrics_dir / 'static'
    dmso_vs_metrics_static_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_interactive_dir = dmso_vs_metrics_dir / 'interactive'
    dmso_vs_metrics_interactive_dir.mkdir(exist_ok=True)

    # Combined plots folder
    dmso_vs_metrics_combined_dir = dmso_vs_metrics_dir / 'combined'
    dmso_vs_metrics_combined_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_combined_reference_dir = dmso_vs_metrics_combined_dir / 'reference'
    dmso_vs_metrics_combined_reference_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_combined_test_dir = dmso_vs_metrics_combined_dir / 'test'
    dmso_vs_metrics_combined_test_dir.mkdir(exist_ok=True)

    # Metric-specific folders
    dmso_vs_metrics_by_metric_dir = dmso_vs_metrics_dir / 'by_metric'
    dmso_vs_metrics_by_metric_dir.mkdir(exist_ok=True)

    # MAD cosine
    dmso_vs_metrics_mad_dir = dmso_vs_metrics_by_metric_dir / 'mad_cosine'
    dmso_vs_metrics_mad_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_mad_interactive_dir = dmso_vs_metrics_mad_dir / 'interactive'
    dmso_vs_metrics_mad_interactive_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_mad_static_dir = dmso_vs_metrics_mad_dir / 'static'
    dmso_vs_metrics_mad_static_dir.mkdir(exist_ok=True)

    # VAR cosine
    dmso_vs_metrics_var_dir = dmso_vs_metrics_by_metric_dir / 'var_cosine'
    dmso_vs_metrics_var_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_var_interactive_dir = dmso_vs_metrics_var_dir / 'interactive'
    dmso_vs_metrics_var_interactive_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_var_static_dir = dmso_vs_metrics_var_dir / 'static'
    dmso_vs_metrics_var_static_dir.mkdir(exist_ok=True)

    # STD cosine
    dmso_vs_metrics_std_dir = dmso_vs_metrics_by_metric_dir / 'std_cosine'
    dmso_vs_metrics_std_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_std_interactive_dir = dmso_vs_metrics_std_dir / 'interactive'
    dmso_vs_metrics_std_interactive_dir.mkdir(exist_ok=True)
    dmso_vs_metrics_std_static_dir = dmso_vs_metrics_std_dir / 'static'
    dmso_vs_metrics_std_static_dir.mkdir(exist_ok=True)

    # Hierarchical clustering directories
    hierarchical_clustering_dir = viz_dir / 'hierarchical_clustering'
    hierarchical_clustering_dir.mkdir(exist_ok=True)
    hierarchical_cluster_map_dir = hierarchical_clustering_dir / 'hierarchical_cluster_map'
    hierarchical_cluster_map_dir.mkdir(exist_ok=True)
    hierarchical_cluster_map_rerun_dir = hierarchical_clustering_dir / 'hierarchical_cluster_map_rerun'
    hierarchical_cluster_map_rerun_dir.mkdir(exist_ok=True)
        
    # Create a directory paths dictionary
    dir_paths = {
        'root': output_dir,
        'data': data_dir,
        'analysis': {
            'root': analysis_dir,
            'mad': mad_dir,
            'dmso_distances': dmso_dist_dir,
            'landmark_distances': landmark_dist_dir
        },
        'visualizations': {
            'root': viz_dir,
            'correlation': correlation_dir,
            'dmso_distributions': {
                'root': dmso_dist_viz_dir,
                'combined': dmso_dist_viz_dir_combined,
                'by_library': dmso_dist_viz_dir_library
            },
            'histograms': {
                'root': histogram_dir,
                'by_library': histogram_dir_library,
                'by_moa': histogram_dir_moa,
                'dispersion_metrics': {
                    'root': hist_dispersion_dir,
                    'reference': hist_dispersion_reference_dir,
                    'test': hist_dispersion_test_dir,
                    'comparisons': hist_dispersion_comparison_dir
                },
                'dmso_distances': {
                    'root': hist_dmso_dir,
                    'reference': hist_dmso_reference_dir,
                    'test': hist_dmso_test_dir,
                    'comparisons': hist_dmso_comparison_dir
                },
                'landmark_distances': {
                    'root': hist_landmark_dir,
                    'reference': hist_landmark_reference_dir,
                    'test': hist_landmark_test_dir,
                    'comparisons': hist_landmark_comparison_dir
                },
                'scores': {
                    'root': hist_scores_dir,
                    'reference': hist_scores_reference_dir,
                    'test': hist_scores_test_dir,
                    'comparisons': hist_scores_comparison_dir
                }
            },
            'landmark_plots': {
                'root': landmark_viz_dir,
                'static': landmark_viz_static_dir,  # Legacy path
                'interactive': landmark_viz_interactive_dir,  # Legacy path
                'combined': {
                    'root': landmark_combined_dir,
                    'reference': landmark_combined_reference_dir,
                    'test': landmark_combined_test_dir
                },
                'by_metric': {
                    'root': landmark_by_metric_dir,
                    'mad_cosine': {
                        'root': landmark_mad_dir,
                        'interactive': landmark_mad_interactive_dir,
                        'static': landmark_mad_static_dir
                    },
                    'var_cosine': {
                        'root': landmark_var_dir,
                        'interactive': landmark_var_interactive_dir,
                        'static': landmark_var_static_dir
                    },
                    'std_cosine': {
                        'root': landmark_std_dir,
                        'interactive': landmark_std_interactive_dir,
                        'static': landmark_std_static_dir
                    }
                }
            },
            'landmark_selection': landmark_selection_dir,
            'dimensionality_reduction': {
                'root': dimensionality_reduction_dir,
                'umap': umap_dir,
                'tsne': tsne_dir
            },
            'dmso_vs_metrics': {
                'root': dmso_vs_metrics_dir,
                'static': dmso_vs_metrics_static_dir,
                'interactive': dmso_vs_metrics_interactive_dir,
                'combined': {
                    'root': dmso_vs_metrics_combined_dir,
                    'reference': dmso_vs_metrics_combined_reference_dir,
                    'test': dmso_vs_metrics_combined_test_dir
                },
                'by_metric': {
                    'root': dmso_vs_metrics_by_metric_dir,
                    'mad_cosine': {
                        'root': dmso_vs_metrics_mad_dir,
                        'interactive': dmso_vs_metrics_mad_interactive_dir,
                        'static': dmso_vs_metrics_mad_static_dir
                    },
                    'var_cosine': {
                        'root': dmso_vs_metrics_var_dir,
                        'interactive': dmso_vs_metrics_var_interactive_dir,
                        'static': dmso_vs_metrics_var_static_dir
                    },
                    'std_cosine': {
                        'root': dmso_vs_metrics_std_dir,
                        'interactive': dmso_vs_metrics_std_interactive_dir,
                        'static': dmso_vs_metrics_std_static_dir
                    }
                }
            },
            'hierarchical_clustering': {
                'root': hierarchical_clustering_dir,
                'cluster_map': hierarchical_cluster_map_dir,
                'cluster_map_rerun': hierarchical_cluster_map_rerun_dir
            }
        }
    }

    log_info(f"Created output directory structure: {output_dir}")

    return dir_paths