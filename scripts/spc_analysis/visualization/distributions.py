# Histogram plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine  # 
from ..utils.logging import log_info, log_section

def create_histogram(data, column, title, filename, dir_paths, colors=None, kde=True, bins=30, color=None, 
                    xlabel=None, threshold=None, threshold_label=None, is_reference=None):
    """Helper function to create consistent histograms"""
    if data is None or column not in data.columns:
        log_info(f"Skipping {title} - data not available")
        return
    
    # Filter out NaN values
    valid_data = data[column].dropna()
    if len(valid_data) == 0:
        log_info(f"Skipping {title} - no valid data")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    sns.histplot(valid_data, kde=kde, bins=bins, color=color)
    
    # Add threshold line if provided
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', 
                  label=threshold_label or f"Threshold: {threshold:.4f}")
        plt.legend()
    
    plt.title(title)
    plt.xlabel(xlabel or column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Determine which directory to use based on metric type and reference/test status
    ref_or_test = 'reference' if is_reference else 'test'
    if ref_or_test is None:
        ref_or_test = 'reference' if 'reference' in title.lower() else 'test'
        
    if 'mad_cosine' in column or 'var_cosine' in column or 'std_cosine' in column:
        # Dispersion metrics
        plot_path = dir_paths['visualizations']['histograms']['dispersion_metrics'][ref_or_test] / filename
    elif 'cosine_distance_from_dmso' in column:
        # DMSO distances
        plot_path = dir_paths['visualizations']['histograms']['dmso_distances'][ref_or_test] / filename
    elif 'closest_landmark_distance' in column:
        # Landmark distances
        plot_path = dir_paths['visualizations']['histograms']['landmark_distances'][ref_or_test] / filename
    elif 'score' in column or 'harmonic_mean' in column or 'ratio_score' in column:
        # Scores
        plot_path = dir_paths['visualizations']['histograms']['scores'][ref_or_test] / filename
    else:
        # Fallback to root histogram directory
        plot_path = dir_paths['visualizations']['histograms']['root'] / filename
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    log_info(f"Saved {title} to: {plot_path}")
    plt.close()

def generate_comprehensive_histograms(reference_df=None, test_df=None, reference_mad=None, 
                                     test_mad=None, reference_dmso_dist=None, test_dmso_dist=None,
                                     reference_landmark_dist=None, test_landmark_dist=None,
                                     reference_scores=None, test_scores=None, config=None, dir_paths=None):
    """
    Generate a comprehensive set of histograms for all key metrics.
    
    Args:
        reference_df: DataFrame with reference set data
        test_df: DataFrame with test set data
        reference_mad: DataFrame with reference dispersion metrics
        test_mad: DataFrame with test dispersion metrics
        reference_dmso_dist: DataFrame with reference DMSO distances
        test_dmso_dist: DataFrame with test DMSO distances
        reference_landmark_dist: DataFrame with reference landmark distances
        test_landmark_dist: DataFrame with test landmark distances
        reference_scores: DataFrame with reference scores
        test_scores: DataFrame with test scores
        config: Configuration dictionary
        dir_paths: Dictionary with directory paths
    """
    log_section("GENERATING COMPREHENSIVE HISTOGRAMS")
    
    # Create output directory for histograms
    if config is None or 'output_dir' not in config:
        log_info("ERROR: No output directory specified. Cannot save histograms.")
        return
    
    output_dir = dir_paths['visualizations']['histograms']['root']
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving histograms to: {output_dir}")
    
    # Set consistent styling for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'reference': '#1f77b4',  # blue
        'test': '#ff7f0e',       # orange
        'combined': '#2ca02c',   # green
        'overlay': ['#1f77b4', '#ff7f0e']  # blue, orange
    }
    
    # 1. Dispersion Metrics Histograms
    log_info("Generating dispersion metrics histograms...")
    
    # Identify which metrics are available in the data
    dispersion_metrics = []
    if reference_mad is not None:
        dispersion_metrics = [col for col in reference_mad.columns if col in ['mad_cosine', 'var_cosine', 'std_cosine']]
    elif test_mad is not None:
        dispersion_metrics = [col for col in test_mad.columns if col in ['mad_cosine', 'var_cosine', 'std_cosine']]
    
    log_info(f"Found dispersion metrics: {dispersion_metrics}")
    
    # Get thresholds from config
    thresholds = {
        'mad_cosine': config.get('mad_threshold', 0.05),
        'var_cosine': config.get('var_threshold', 0.005),
        'std_cosine': config.get('std_threshold', 0.07)
    }
    
    # Generate histograms for each dispersion metric
    for metric in dispersion_metrics:
        metric_title = metric.replace('_', ' ').title()
        threshold = thresholds.get(metric)
        
        if reference_mad is not None:
            create_histogram(
                reference_mad, metric, 
                f'{metric_title} Distribution - Reference Set',
                f'{metric}_reference.png', dir_paths, color=colors['reference'],
                threshold=threshold, 
                threshold_label=f"{metric_title} Threshold: {threshold:.4f}" if threshold else None,
                is_reference=True
            )
        
        if test_mad is not None:
            create_histogram(
                test_mad, metric, 
                f'{metric_title} Distribution - Test Set',
                f'{metric}_test.png', dir_paths, color=colors['test'],
                threshold=threshold, 
                threshold_label=f"{metric_title} Threshold: {threshold:.4f}" if threshold else None,
                is_reference=False
            )
        
        # Overlay histogram
        if reference_mad is not None and test_mad is not None:
            create_overlay_histogram(
                reference_mad, test_mad, metric, 
                f'{metric_title} Distribution - Reference vs Test',
                f'{metric}_comparison.png', dir_paths, colors
            )
    
    # 2. DMSO Distance Histograms
    log_info("Generating DMSO distance histograms...")
    
    # DMSO threshold from config
    dmso_threshold_percentile = config.get('dmso_threshold_percentile', '99')
    dmso_threshold = None
    if 'dmso_thresholds' in config and dmso_threshold_percentile in config['dmso_thresholds']:
        dmso_threshold = config['dmso_thresholds'][dmso_threshold_percentile]
    
    if reference_dmso_dist is not None:
        create_histogram(
            reference_dmso_dist, 'cosine_distance_from_dmso', 
            'Distance from DMSO Distribution - Reference Set',
            'dmso_distance_reference.png', dir_paths, color=colors['reference'],
            threshold=dmso_threshold, 
            threshold_label=f"DMSO {dmso_threshold_percentile}% Threshold: {dmso_threshold:.4f}" if dmso_threshold else None,
            is_reference=True
        )
    
    if test_dmso_dist is not None:
        create_histogram(
            test_dmso_dist, 'cosine_distance_from_dmso', 
            'Distance from DMSO Distribution - Test Set',
            'dmso_distance_test.png', dir_paths, color=colors['test'],
            threshold=dmso_threshold, 
            threshold_label=f"DMSO {dmso_threshold_percentile}% Threshold: {dmso_threshold:.4f}" if dmso_threshold else None,
            is_reference=False
        )
    
    # DMSO distance overlay histogram
    if reference_dmso_dist is not None and test_dmso_dist is not None:
        create_overlay_histogram(
            reference_dmso_dist, test_dmso_dist, 'cosine_distance_from_dmso', 
            'Distance from DMSO Distribution - Reference vs Test',
            'dmso_distance_comparison.png', dir_paths, colors
        )
    
    # 3. Landmark Distance Histograms
    log_info("Generating landmark distance histograms...")
    
    # Similarity threshold from config
    similarity_threshold = config.get('similarity_threshold', 0.2)
    
    if reference_landmark_dist is not None:
        create_histogram(
            reference_landmark_dist, 'closest_landmark_distance', 
            'Closest Landmark Distance Distribution - Reference Set',
            'landmark_distance_reference.png', dir_paths, color=colors['reference'],
            threshold=similarity_threshold, 
            threshold_label=f"Similarity Threshold: {similarity_threshold:.4f}",
            is_reference=True
        )
    
    if test_landmark_dist is not None:
        create_histogram(
            test_landmark_dist, 'closest_landmark_distance', 
            'Closest Landmark Distance Distribution - Test Set',
            'landmark_distance_test.png', dir_paths, color=colors['test'],
            threshold=similarity_threshold, 
            threshold_label=f"Similarity Threshold: {similarity_threshold:.4f}",
            is_reference=False
        )
    
    # Landmark distance overlay histogram
    if reference_landmark_dist is not None and test_landmark_dist is not None:
        create_overlay_histogram(
            reference_landmark_dist, test_landmark_dist, 'closest_landmark_distance', 
            'Closest Landmark Distance Distribution - Reference vs Test',
            'landmark_distance_comparison.png', dir_paths, colors
        )
    
    # 4. Score Distributions
    log_info("Generating score histograms...")
    
    # Check which scores we have
    score_columns = [
        ('ratio_score', 'Ratio Score'),
        ('harmonic_mean_2term', '2-term Harmonic Mean'),
        ('harmonic_mean_3term', '3-term Harmonic Mean')
    ]
    
    # Add new metric-specific score columns
    for metric in dispersion_metrics:
        score_columns.extend([
            (f'harmonic_mean_2term_{metric}', f'2-term Harmonic Mean ({metric})'),
            (f'harmonic_mean_3term_{metric}', f'3-term Harmonic Mean ({metric})')
        ])
    
    for col, label in score_columns:
        # Reference scores
        if reference_scores is not None and col in reference_scores.columns:
            create_histogram(
                reference_scores, col, 
                f'{label} Distribution - Reference Set',
                f'{col}_reference.png', dir_paths, color=colors['reference'],
                is_reference=True
            )
        
        # Test scores
        if test_scores is not None and col in test_scores.columns:
            create_histogram(
                test_scores, col, 
                f'{label} Distribution - Test Set',
                f'{col}_test.png', dir_paths, color=colors['test'],
                is_reference=False
            )
        
        # Overlay scores
        if (reference_scores is not None and col in reference_scores.columns and
            test_scores is not None and col in test_scores.columns):
            create_overlay_histogram(
                reference_scores, test_scores, col, 
                f'{label} Distribution - Reference vs Test',
                f'{col}_comparison.png', dir_paths, colors
            )

    # 5. Library Breakdowns
    log_info("Generating library breakdown histograms...")
    
    # Check if we have library information
    datasets_with_library = []
    for dataset, name in [
        (reference_scores, 'reference_scores'),
        (test_scores, 'test_scores'),
        (reference_mad, 'reference_mad'),
        (test_mad, 'test_mad'),
        (reference_dmso_dist, 'reference_dmso_dist'),
        (test_dmso_dist, 'test_dmso_dist'),
        (reference_landmark_dist, 'reference_landmark_dist'),
        (test_landmark_dist, 'test_landmark_dist')
    ]:
        if dataset is not None and 'library' in dataset.columns:
            datasets_with_library.append((dataset, name))
    
    if datasets_with_library:
        # We have at least one dataset with library info
        # Create subdirectory for library-specific plots
        lib_dir = dir_paths['visualizations']['histograms']['by_library']
        lib_dir.mkdir(exist_ok=True)
        
        for dataset, name in datasets_with_library:
            libraries = dataset['library'].unique()
            log_info(f"Found {len(libraries)} libraries in {name}")
            
            # Only proceed if we have multiple libraries
            if len(libraries) <= 1:
                continue
            
            # For each metric in the dataset, create library breakdowns
            metrics = []
            # Include all dispersion metrics
            for metric in dispersion_metrics:
                if metric in dataset.columns:
                    metrics.append((metric, metric.replace('_', ' ').title()))
                    
            if 'cosine_distance_from_dmso' in dataset.columns:
                metrics.append(('cosine_distance_from_dmso', 'Distance from DMSO'))
            if 'closest_landmark_distance' in dataset.columns:
                metrics.append(('closest_landmark_distance', 'Closest Landmark Distance'))
            for col, _ in score_columns:
                if col in dataset.columns:
                    metrics.append((col, score_columns[score_columns.index((col, _))][1]))
            
            # Create box plots comparing libraries
            for col, label in metrics:
                # Skip if not enough data
                if dataset[col].count() < 10:
                    continue
                
                plt.figure(figsize=(12, 7))
                sns.boxplot(x='library', y=col, data=dataset)
                plt.title(f'{label} by Library - {name.replace("_", " ").title()}')
                plt.xlabel('Library')
                plt.ylabel(label)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save box plot
                box_path = lib_dir / f'{col}_by_library_{name}.png'
                plt.savefig(box_path, dpi=300, bbox_inches='tight')
                log_info(f"Saved {label} by library box plot to: {box_path}")
                plt.close()
                
                # Create separate histograms for each library
                for lib in libraries:
                    lib_data = dataset[dataset['library'] == lib]
                    if len(lib_data) < 5:  # Skip if too few samples
                        continue
                    
                    create_histogram(
                        lib_data, col, 
                        f'{label} - {lib} Library ({name.replace("_", " ").title()})',
                        f'{col}_{lib.replace(" ", "_")}_{name}.png',
                        dir_paths,
                        color=np.random.rand(3,),  # Random color for variety
                        bins=min(25, max(10, len(lib_data) // 5))  # Adaptive bin count
                    )
    
    # 6. MOA Breakdowns (if available)
    log_info("Generating MOA breakdown histograms...")
    
    # Check if we have MOA information
    datasets_with_moa = []
    for dataset, name in [
        (reference_scores, 'reference_scores'),
        (test_scores, 'test_scores'),
        (reference_mad, 'reference_mad'),
        (test_mad, 'test_mad'),
        (reference_dmso_dist, 'reference_dmso_dist'),
        (test_dmso_dist, 'test_dmso_dist'),
        (reference_landmark_dist, 'reference_landmark_dist'),
        (test_landmark_dist, 'test_landmark_dist')
    ]:
        if dataset is not None and 'moa' in dataset.columns:
            datasets_with_moa.append((dataset, name))
    
    if datasets_with_moa:
        # We have at least one dataset with MOA info
        # Create subdirectory for MOA-specific plots
        moa_dir = dir_paths['visualizations']['histograms']['by_moa']
        moa_dir.mkdir(exist_ok=True)
        
        for dataset, name in datasets_with_moa:
            # Clean MOA values and get counts
            dataset = dataset.copy()
            dataset['moa'] = dataset['moa'].fillna('Unknown').astype(str)
            moa_counts = dataset['moa'].value_counts()
            
            # Limit to top 10 MOAs by count
            top_moas = moa_counts.nlargest(10).index.tolist()
            log_info(f"Found {len(moa_counts)} MOAs in {name}, using top 10")
            
            # Only proceed if we have multiple MOAs
            if len(top_moas) <= 1:
                continue
            
            # For each metric in the dataset, create MOA breakdowns
            metrics = []
            # Include all dispersion metrics
            for metric in dispersion_metrics:
                if metric in dataset.columns:
                    metrics.append((metric, metric.replace('_', ' ').title()))
                    
            if 'cosine_distance_from_dmso' in dataset.columns:
                metrics.append(('cosine_distance_from_dmso', 'Distance from DMSO'))
            if 'closest_landmark_distance' in dataset.columns:
                metrics.append(('closest_landmark_distance', 'Closest Landmark Distance'))
            
            # Create box plots comparing MOAs
            for col, label in metrics:
                # Skip if not enough data
                if dataset[col].count() < 10:
                    continue
                
                # Filter to top MOAs and valid values
                moa_data = dataset[dataset['moa'].isin(top_moas)].dropna(subset=[col])
                
                plt.figure(figsize=(14, 8))
                sns.boxplot(x='moa', y=col, data=moa_data)
                plt.title(f'{label} by MOA - {name.replace("_", " ").title()}')
                plt.xlabel('MOA')
                plt.ylabel(label)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save box plot
                box_path = moa_dir / f'{col}_by_moa_{name}.png'
                plt.savefig(box_path, dpi=300, bbox_inches='tight')
                log_info(f"Saved {label} by MOA box plot to: {box_path}")
                plt.close()

    log_info("Completed generating comprehensive histograms")


def generate_dmso_cosine_distribution_plots(reference_df=None, test_df=None, embedding_cols=None, config=None, dir_paths=None):
    """
    Generate DMSO cosine distance distribution plots for reference, test, and library-specific datasets.
    
    Args:
        reference_df: DataFrame with reference set data
        test_df: DataFrame with test set data
        embedding_cols: List of embedding column names
        config: Configuration dictionary
        dir_paths: Dictionary with directory paths
    """
    log_section("GENERATING DMSO COSINE DISTRIBUTION PLOTS")
    
    # Create output directory for DMSO distribution plots
    if config is None or 'output_dir' not in config:
        log_info("ERROR: No output directory specified. Cannot save DMSO distribution plots.")
        return
    
    output_dir = dir_paths['visualizations']['dmso_distributions']['root']
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving DMSO distribution plots to: {output_dir}")
    
    # Ensure library subdirectory exists
    lib_dir = dir_paths['visualizations']['dmso_distributions']['by_library']
    lib_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving library-specific DMSO distribution plots to: {lib_dir}")
        
    def plot_dmso_distribution(data, label, dir_paths, is_library=False):
        """
        Create DMSO cosine distribution plot for a given dataset
        
        Args:
            data: DataFrame containing data
            label: Label for plot (e.g., 'Reference', 'Test', or library name)
            dir_paths: Directory to save plots
            is_library: Boolean indicating if this is a library-specific plot
        """
        # Extract DMSO samples
        dmso_df = data[data['treatment'].str.startswith('DMSO')]
        
        if len(dmso_df) == 0:
            log_info(f"No DMSO samples found for {label}")
            return None, None
        
        # Calculate DMSO centroid
        dmso_centroid = dmso_df[embedding_cols].median().values
        
        # Calculate distances between DMSO samples and DMSO centroid
        dmso_distances = []
        for _, row in dmso_df.iterrows():
            sample_embedding = row[embedding_cols].values
            dist = cosine(dmso_centroid, sample_embedding)
            dmso_distances.append(dist)
        
        # Calculate thresholds
        thresholds = {
            '80': np.percentile(dmso_distances, 80),
            '90': np.percentile(dmso_distances, 90),
            '95': np.percentile(dmso_distances, 95),
            '99': np.percentile(dmso_distances, 99),
            '99.9': np.percentile(dmso_distances, 99.9),
            '99.99': np.percentile(dmso_distances, 99.99)
        }

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.hist(dmso_distances, bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'DMSO Cosine Distance Distribution - {label}')
        plt.xlabel('Cosine Distance from DMSO Centroid')
        plt.ylabel('Frequency')

        # Add threshold lines with different styles and colors
        colors = ['darkgreen', 'blue', 'orange', 'red', 'purple', 'black']
        line_styles = [':', '--', '-.', '-', '--', '-.']

        for (percentile, threshold), color, line_style in zip(thresholds.items(), colors, line_styles):
            plt.axvline(threshold, color=color, linestyle=line_style, 
                        label=f'{percentile}th Percentile: {threshold:.4f}')

        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot - use 'by_library' path if this is a library plot
        plot_filename = f'dmso_cosine_distribution_{label.replace(" ", "_")}.png'
        
        if is_library:
            plot_path = dir_paths['visualizations']['dmso_distributions']['by_library'] / plot_filename
        else:
            plot_path = dir_paths['visualizations']['dmso_distributions']['root'] / plot_filename
            
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        log_info(f"Saved DMSO distribution plot for {label} to: {plot_path}")
        plt.close()
        
        return dmso_distances, thresholds

    # Plot for overall reference and test datasets
    if reference_df is not None:
        ref_dmso_distances, _ = plot_dmso_distribution(reference_df, 'Reference', dir_paths)
    else:
        ref_dmso_distances = None
        
    if test_df is not None and not test_df.equals(reference_df):
        test_dmso_distances, _ = plot_dmso_distribution(test_df, 'Test', dir_paths)
    else:
        test_dmso_distances = None

    # Combine DMSO samples from reference and test datasets
    if ref_dmso_distances is not None and test_dmso_distances is not None:
        combined_dmso_distances = ref_dmso_distances + test_dmso_distances
        
        # Calculate combined thresholds and store them in config for plotting consistency
        combined_thresholds = {
            '80': np.percentile(combined_dmso_distances, 80),
            '90': np.percentile(combined_dmso_distances, 90),
            '95': np.percentile(combined_dmso_distances, 95),
            '99': np.percentile(combined_dmso_distances, 99),
            '99.9': np.percentile(combined_dmso_distances, 99.9),
            '99.99': np.percentile(combined_dmso_distances, 99.99)
        }
        
        # Store combined thresholds in config for use in combined plots
        config['dmso_combined_thresholds'] = combined_thresholds
        
        log_info("Combined DMSO thresholds calculated:")
        for pct, val in combined_thresholds.items():
            log_info(f"  Combined {pct}%: {val:.4f}")
        
        # Use combined thresholds for the combined plot
        thresholds = combined_thresholds

        # Create combined plot
        plt.figure(figsize=(10, 6))
        plt.hist([ref_dmso_distances, test_dmso_distances], bins=30, 
                 edgecolor='black', alpha=0.7, label=['Reference', 'Test'])
        plt.title('DMSO Cosine Distance Distribution - Combined (Using Combined Thresholds)')
        plt.xlabel('Cosine Distance from DMSO Centroid')
        plt.ylabel('Frequency')

        # Add threshold lines with different styles and colors
        colors = ['darkgreen', 'blue', 'darkorange', 'red', 'purple', 'black']
        line_styles = [':', '--', '-.', '-', '--', '-.']

        for (percentile, threshold), color, line_style in zip(thresholds.items(), colors, line_styles):
            plt.axvline(threshold, color=color, linestyle=line_style, 
                        label=f'{percentile}th Percentile: {threshold:.4f}')

        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save combined plot
        plot_filename = 'dmso_cosine_distribution_combined.png'
        plot_path = dir_paths['visualizations']['dmso_distributions']['combined'] / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        log_info(f"Saved combined DMSO distribution plot to: {plot_path}")
        plt.close()

    # Plot for libraries
    if config is not None and 'plate_definitions' in config:
        # Get unique libraries
        if reference_df is not None and 'library' in reference_df.columns:
            libraries = reference_df['library'].unique()
            
            for lib in libraries:
                lib_data = reference_df[reference_df['library'] == lib]
                
                if len(lib_data) > 5:  # Only plot if enough samples
                    plot_dmso_distribution(lib_data, f'Library - {lib}', dir_paths, is_library=True)
        
        if test_df is not None and not test_df.equals(reference_df) and 'library' in test_df.columns:
            libraries = test_df['library'].unique()
            
            for lib in libraries:
                lib_data = test_df[test_df['library'] == lib]
                
                if len(lib_data) > 5:  # Only plot if enough samples
                    plot_dmso_distribution(lib_data, f'Library - {lib}', dir_paths, is_library=True)


def create_overlay_histogram(data1, data2, column, title, filename, dir_paths, colors, labels=None, kde=True, bins=30):
    """Helper function to create overlay histograms comparing two datasets"""
    if data1 is None or data2 is None or column not in data1.columns or column not in data2.columns:
        log_info(f"Skipping {title} - data not available")
        return
    
    # Filter out NaN values
    valid_data1 = data1[column].dropna()
    valid_data2 = data2[column].dropna()
    
    if len(valid_data1) == 0 and len(valid_data2) == 0:
        log_info(f"Skipping {title} - no valid data")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create overlapping histograms
    if len(valid_data1) > 0:
        sns.histplot(valid_data1, kde=kde, bins=bins, color=colors['overlay'][0], 
                   alpha=0.6, label=labels[0] if labels else 'Reference')
    
    if len(valid_data2) > 0:
        sns.histplot(valid_data2, kde=kde, bins=bins, color=colors['overlay'][1], 
                   alpha=0.6, label=labels[1] if labels else 'Test')
    
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Determine which directory to use based on metric type
    if 'mad_cosine' in column or 'var_cosine' in column or 'std_cosine' in column:
        # Dispersion metrics
        plot_path = dir_paths['visualizations']['histograms']['dispersion_metrics']['comparisons'] / filename
    elif 'cosine_distance_from_dmso' in column:
        # DMSO distances
        plot_path = dir_paths['visualizations']['histograms']['dmso_distances']['comparisons'] / filename
    elif 'closest_landmark_distance' in column:
        # Landmark distances
        plot_path = dir_paths['visualizations']['histograms']['landmark_distances']['comparisons'] / filename
    elif 'score' in column:
        # Scores
        plot_path = dir_paths['visualizations']['histograms']['scores']['comparisons'] / filename
    else:
        # Fallback to root histogram directory
        plot_path = dir_paths['visualizations']['histograms']['root'] / filename
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    log_info(f"Saved {title} to: {plot_path}")
    plt.close()