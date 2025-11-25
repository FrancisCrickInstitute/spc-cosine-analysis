"""
Landmark Distance Threshold Analysis

This module analyzes the distribution of landmarks within various cosine distance 
thresholds to help determine optimal threshold values for landmark selection.

Analyzes thresholds: 0.10, 0.15, 0.20, 0.25, 0.30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from ..utils.logging import log_info, log_section


def calculate_landmark_counts_at_thresholds(df, landmark_distances, embedding_cols, thresholds):
    """
    For each treatment, count how many landmarks fall within each distance threshold
    
    Args:
        df: DataFrame with embeddings
        landmark_distances: DataFrame with landmark distance information
        embedding_cols: List of embedding column names
        thresholds: List of distance thresholds to analyze
        
    Returns:
        DataFrame: Counts of landmarks within each threshold for each treatment
    """
    log_info("Calculating landmark counts at multiple thresholds...")
    
    results = []
    
    # Get unique treatments
    treatments = df['treatment'].unique()
    
    # Create landmark embeddings lookup
    landmark_embeddings = {}
    if 'is_self_landmark' in df.columns:
        landmark_df = df[df['is_self_landmark'] == True]
        for _, row in landmark_df.iterrows():
            treatment = row['treatment']
            embedding = row[embedding_cols].values
            landmark_embeddings[treatment] = embedding
    
    log_info(f"Found {len(landmark_embeddings)} landmarks with embeddings")
    
    # Calculate distances for each treatment
    for treatment in treatments:
        treatment_rows = df[df['treatment'] == treatment]
        if len(treatment_rows) == 0:
            continue
        
        # Get treatment embedding (median of replicates)
        treatment_embedding = treatment_rows[embedding_cols].median().values
        
        # Get metadata
        first_row = treatment_rows.iloc[0]
        library = first_row.get('library', 'Unknown')
        is_landmark = first_row.get('is_self_landmark', False)
        is_test = library in ['CCA_V1', 'GSK_clickable_V1', 'GSK_fragments_V3', 'HTC_V1']
        
        # Calculate distances to all landmarks
        landmark_distances_list = []
        for lm_treatment, lm_embedding in landmark_embeddings.items():
            if lm_treatment != treatment:  # Exclude self
                dist = cosine(treatment_embedding, lm_embedding)
                landmark_distances_list.append(dist)
        
        if len(landmark_distances_list) == 0:
            continue
        
        # Count landmarks within each threshold
        threshold_counts = {}
        for threshold in thresholds:
            count = sum(1 for d in landmark_distances_list if d <= threshold)
            threshold_counts[f'landmarks_within_{threshold:.2f}'] = count
        
        # Add closest landmark distance
        closest_distance = min(landmark_distances_list) if landmark_distances_list else np.nan
        
        # Create result record
        result = {
            'treatment': treatment,
            'library': library,
            'is_test': is_test,
            'is_landmark': is_landmark,
            'closest_landmark_distance': closest_distance,
            **threshold_counts,
            'total_landmarks_available': len(landmark_embeddings) - 1  # Exclude self
        }
        
        # Add metadata
        for col in ['PP_ID', 'PP_ID_uM', 'moa', 'compound_name', 'compound_uM']:
            if col in first_row:
                result[col] = first_row[col]
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    log_info(f"Calculated threshold counts for {len(results_df)} treatments")
    
    return results_df


def plot_landmark_counts_by_threshold(threshold_counts_df, output_dir, config=None):
    """
    Create visualizations of landmark counts at different thresholds
    
    Args:
        threshold_counts_df: DataFrame with landmark counts at each threshold
        output_dir: Path to save plots
        config: Configuration dictionary
    """
    log_section("CREATING LANDMARK THRESHOLD ANALYSIS PLOTS")
    
    # Create output directories
    dist_dir = output_dir / 'distributions'
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    comp_dir = output_dir / 'comparisons'
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    by_lib_dir = output_dir / 'by_library'
    by_lib_dir.mkdir(parents=True, exist_ok=True)
    
    # Get threshold columns
    threshold_cols = [col for col in threshold_counts_df.columns if col.startswith('landmarks_within_')]
    thresholds = sorted([float(col.split('_')[-1]) for col in threshold_cols])
    
    log_info(f"Creating plots for {len(thresholds)} thresholds: {thresholds}")
    
    # Filter to test compounds only for main analysis
    test_df = threshold_counts_df[threshold_counts_df['is_test'] == True].copy()
    log_info(f"Analyzing {len(test_df)} test compounds")
    
    # =========================================================================
    # PLOT 1: Stacked bar chart - landmark counts by threshold
    # =========================================================================
    log_info("Creating stacked bar chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for stacking
    # Bin the counts: 0, 1, 2, 3, 4-10, 11-20, 21+
    bins = [0, 1, 2, 3, 4, 11, 21, np.inf]
    bin_labels = ['0', '1', '2', '3', '4-10', '11-20', '21+']
    
    plot_data = []
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = test_df[col_name]
        
        # Bin the counts
        binned = pd.cut(counts, bins=bins, labels=bin_labels, right=False)
        bin_counts = binned.value_counts().sort_index()
        
        plot_data.append({
            'threshold': f'{threshold:.2f}',
            **{label: bin_counts.get(label, 0) for label in bin_labels}
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create stacked bar chart
    x = np.arange(len(plot_df))
    width = 0.6
    
    bottom = np.zeros(len(plot_df))
    colors = sns.color_palette("viridis", len(bin_labels))
    
    for idx, label in enumerate(bin_labels):
        values = plot_df[label].values
        ax.bar(x, values, width, label=f'{label} landmarks', bottom=bottom, color=colors[idx])
        bottom += values
    
    ax.set_xlabel('Distance Threshold', fontsize=12)
    ax.set_ylabel('Number of Test Compounds', fontsize=12)
    ax.set_title('Distribution of Landmark Counts at Different Distance Thresholds\n(Test Compounds Only)', 
                fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['threshold'])
    ax.legend(title='Landmarks within threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = dist_dir / 'landmark_counts_by_threshold_stacked.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Saved: {output_path}")
    
    # =========================================================================
    # PLOT 2: Cumulative distribution curves
    # =========================================================================
    log_info("Creating cumulative distribution plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = test_df[col_name].values
        
        # Calculate cumulative distribution
        unique_counts = np.arange(0, counts.max() + 1)
        cumulative = [np.sum(counts >= count) / len(counts) * 100 for count in unique_counts]
        
        ax.plot(unique_counts, cumulative, marker='o', label=f'{threshold:.2f}', linewidth=2)
    
    ax.set_xlabel('Number of Landmarks Within Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Test Compounds (%)', fontsize=12)
    ax.set_title('Cumulative Distribution: % of Compounds with ≥N Landmarks\n(Test Compounds Only)',
                fontsize=14, pad=20)
    ax.legend(title='Distance Threshold', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(0, 105)
    
    # Add reference lines
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(ax.get_xlim()[1] * 0.95, 52, '50%', fontsize=9, color='red', ha='right')
    ax.text(ax.get_xlim()[1] * 0.95, 77, '75%', fontsize=9, color='orange', ha='right')
    
    plt.tight_layout()
    output_path = dist_dir / 'cumulative_landmark_coverage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Saved: {output_path}")
    
    # =========================================================================
    # PLOT 3: Heatmap - thresholds vs landmark counts
    # =========================================================================
    log_info("Creating threshold heatmap...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create matrix: rows = thresholds, columns = landmark count bins
    max_count = max([test_df[f'landmarks_within_{t:.2f}'].max() for t in thresholds])
    count_bins = list(range(0, min(int(max_count) + 2, 51)))  # Cap at 50 for readability
    
    heatmap_data = []
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = test_df[col_name]
        
        row_data = []
        for count in count_bins:
            n_compounds = np.sum(counts == count)
            row_data.append(n_compounds)
        
        heatmap_data.append(row_data)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                              index=[f'{t:.2f}' for t in thresholds],
                              columns=[str(c) for c in count_bins])
    
    # Plot heatmap
    sns.heatmap(heatmap_df, cmap='YlOrRd', annot=False, fmt='d', 
                cbar_kws={'label': 'Number of Compounds'}, ax=ax)
    
    ax.set_xlabel('Number of Landmarks Within Threshold', fontsize=12)
    ax.set_ylabel('Distance Threshold', fontsize=12)
    ax.set_title('Heatmap: Test Compounds by Threshold and Landmark Count', fontsize=14, pad=20)
    
    # Only show every 5th x-tick for readability
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::5])
    ax.set_xticklabels([count_bins[int(i)] if int(i) < len(count_bins) else '' 
                        for i in xticks[::5]], rotation=0)
    
    plt.tight_layout()
    output_path = dist_dir / 'threshold_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Saved: {output_path}")
    
    # =========================================================================
    # PLOT 4: Violin plots - distribution of closest distances
    # =========================================================================
    log_info("Creating violin plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create categories based on number of landmarks within 0.20
    if 'landmarks_within_0.20' in test_df.columns:
        test_df['landmark_category'] = pd.cut(
            test_df['landmarks_within_0.20'],
            bins=[-1, 0, 1, 2, 5, np.inf],
            labels=['0', '1', '2', '3-5', '6+']
        )
        
        # Prepare data for violin plot
        violin_data = []
        categories = ['0', '1', '2', '3-5', '6+']
        
        for cat in categories:
            cat_data = test_df[test_df['landmark_category'] == cat]['closest_landmark_distance']
            if len(cat_data) > 0:
                for val in cat_data:
                    if pd.notna(val):
                        violin_data.append({'category': cat, 'distance': val})
        
        if violin_data:
            violin_df = pd.DataFrame(violin_data)
            
            # Create violin plot
            sns.violinplot(data=violin_df, x='category', y='distance', ax=ax, palette='Set2')
            
            # Add horizontal line at 0.2
            ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Current threshold (0.2)')
            
            ax.set_xlabel('Number of Landmarks Within 0.20 Threshold', fontsize=12)
            ax.set_ylabel('Closest Landmark Distance', fontsize=12)
            ax.set_title('Distribution of Closest Landmark Distances\nGrouped by Landmark Count (0.20 threshold)',
                        fontsize=14, pad=20)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            output_path = dist_dir / 'closest_distance_by_landmark_count.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            log_info(f"Saved: {output_path}")
    
    # =========================================================================
    # PLOT 5: Threshold comparison - 0.10 vs 0.20 vs 0.30
    # =========================================================================
    log_info("Creating threshold comparison plot...")
    
    if all(f'landmarks_within_{t:.2f}' in test_df.columns for t in [0.10, 0.20, 0.30]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, threshold in enumerate([0.10, 0.20, 0.30]):
            col_name = f'landmarks_within_{threshold:.2f}'
            counts = test_df[col_name]
            
            # Create histogram
            axes[idx].hist(counts, bins=range(0, int(counts.max()) + 2), 
                          edgecolor='black', alpha=0.7, color=f'C{idx}')
            
            # Add statistics
            mean_val = counts.mean()
            median_val = counts.median()
            
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            axes[idx].axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            
            axes[idx].set_xlabel('Number of Landmarks', fontsize=11)
            axes[idx].set_ylabel('Number of Test Compounds', fontsize=11)
            axes[idx].set_title(f'Threshold: {threshold:.2f}\n({len(counts)} compounds)', fontsize=12)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add text box with statistics
            stats_text = f'0 landmarks: {(counts == 0).sum()} ({(counts == 0).sum()/len(counts)*100:.1f}%)\n'
            stats_text += f'1+ landmarks: {(counts >= 1).sum()} ({(counts >= 1).sum()/len(counts)*100:.1f}%)\n'
            stats_text += f'3+ landmarks: {(counts >= 3).sum()} ({(counts >= 3).sum()/len(counts)*100:.1f}%)'
            
            axes[idx].text(0.98, 0.98, stats_text, transform=axes[idx].transAxes,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=9)
        
        plt.suptitle('Comparison of Landmark Counts at Different Thresholds', fontsize=14, y=1.02)
        plt.tight_layout()
        output_path = comp_dir / 'threshold_comparison_histograms.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_info(f"Saved: {output_path}")
    
    # =========================================================================
    # PLOT 6: By library analysis
    # =========================================================================
    log_info("Creating by-library analysis...")
    
    libraries = test_df['library'].unique()
    log_info(f"Found {len(libraries)} test libraries")
    
    for library in libraries:
        if pd.isna(library):
            continue
        
        lib_df = test_df[test_df['library'] == library]
        
        if len(lib_df) < 10:  # Skip libraries with few compounds
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot boxplots for each threshold
        box_data = []
        for threshold in thresholds:
            col_name = f'landmarks_within_{threshold:.2f}'
            for count in lib_df[col_name]:
                box_data.append({
                    'threshold': f'{threshold:.2f}',
                    'count': count
                })
        
        box_df = pd.DataFrame(box_data)
        
        sns.boxplot(data=box_df, x='threshold', y='count', ax=ax, palette='Set3')
        
        ax.set_xlabel('Distance Threshold', fontsize=12)
        ax.set_ylabel('Number of Landmarks', fontsize=12)
        ax.set_title(f'Landmark Counts by Threshold - {library}\n({len(lib_df)} compounds)', 
                    fontsize=14, pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean markers
        for i, threshold in enumerate(thresholds):
            col_name = f'landmarks_within_{threshold:.2f}'
            mean_val = lib_df[col_name].mean()
            ax.plot(i, mean_val, marker='D', markersize=10, color='red', zorder=10)
        
        plt.tight_layout()
        safe_lib_name = library.replace('/', '_').replace(' ', '_')
        output_path = by_lib_dir / f'{safe_lib_name}_threshold_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_info(f"Saved: {output_path}")
    
    # =========================================================================
    # Save summary statistics
    # =========================================================================
    log_info("Saving summary statistics...")
    
    summary_stats = []
    for threshold in thresholds:
        col_name = f'landmarks_within_{threshold:.2f}'
        counts = test_df[col_name]
        
        stats = {
            'threshold': threshold,
            'mean': counts.mean(),
            'median': counts.median(),
            'std': counts.std(),
            'min': counts.min(),
            'max': counts.max(),
            'q25': counts.quantile(0.25),
            'q75': counts.quantile(0.75),
            'compounds_with_0_landmarks': (counts == 0).sum(),
            'compounds_with_1plus_landmarks': (counts >= 1).sum(),
            'compounds_with_2plus_landmarks': (counts >= 2).sum(),
            'compounds_with_3plus_landmarks': (counts >= 3).sum(),
            'pct_with_0_landmarks': (counts == 0).sum() / len(counts) * 100,
            'pct_with_1plus_landmarks': (counts >= 1).sum() / len(counts) * 100,
            'pct_with_2plus_landmarks': (counts >= 2).sum() / len(counts) * 100,
            'pct_with_3plus_landmarks': (counts >= 3).sum() / len(counts) * 100
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    output_path = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(output_path, index=False)
    log_info(f"Saved: {output_path}")
    
    # Print summary to log
    log_info("\n" + "="*80)
    log_info("SUMMARY STATISTICS")
    log_info("="*80)
    for _, row in summary_df.iterrows():
        log_info(f"\nThreshold: {row['threshold']:.2f}")
        log_info(f"  Mean landmarks: {row['mean']:.2f}")
        log_info(f"  Median landmarks: {row['median']:.1f}")
        log_info(f"  Compounds with 0 landmarks: {row['compounds_with_0_landmarks']} ({row['pct_with_0_landmarks']:.1f}%)")
        log_info(f"  Compounds with 1+ landmarks: {row['compounds_with_1plus_landmarks']} ({row['pct_with_1plus_landmarks']:.1f}%)")
        log_info(f"  Compounds with 3+ landmarks: {row['compounds_with_3plus_landmarks']} ({row['pct_with_3plus_landmarks']:.1f}%)")
    
    log_info("\n✓ All threshold analysis plots created successfully")


def run_landmark_threshold_analysis(merged_df, embedding_cols, landmark_distances, config, dir_paths):
    """
    Main function to run landmark threshold analysis
    
    Args:
        merged_df: DataFrame with embeddings and metadata
        embedding_cols: List of embedding column names
        landmark_distances: DataFrame with landmark distance information
        config: Configuration dictionary
        dir_paths: Directory paths dictionary
    """
    log_section("LANDMARK DISTANCE THRESHOLD ANALYSIS")
    
    # Check if enabled in config
    if not config.get('run_landmark_threshold_analysis', False):
        log_info("Landmark threshold analysis disabled in config")
        return
    
    # Define thresholds to analyze
    thresholds = config.get('landmark_thresholds_to_analyze', [0.10, 0.15, 0.20, 0.25, 0.30])
    log_info(f"Analyzing {len(thresholds)} thresholds: {thresholds}")
    
    # Create output directory
    output_dir = dir_paths['visualizations']['root'] / 'landmark_threshold_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Output directory: {output_dir}")
    
    try:
        # Calculate landmark counts at each threshold
        threshold_counts = calculate_landmark_counts_at_thresholds(
            merged_df, landmark_distances, embedding_cols, thresholds
        )
        
        # Save the threshold counts data
        output_path = output_dir / 'threshold_counts_per_treatment.csv'
        threshold_counts.to_csv(output_path, index=False)
        log_info(f"Saved threshold counts to: {output_path}")
        
        # Create visualizations
        plot_landmark_counts_by_threshold(threshold_counts, output_dir, config)
        
        log_section("✓ LANDMARK THRESHOLD ANALYSIS COMPLETED!")
        log_info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        log_info(f"\n ERROR in landmark threshold analysis: {e}")
        import traceback
        log_info(traceback.format_exc())