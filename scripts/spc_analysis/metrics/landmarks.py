# Landmark identification

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from ..utils.logging import log_info, log_section
from pathlib import Path
from ..utils.logging import log_section, log_info

def identify_landmarks(mad_df, distance_df, config, dir_paths):
    """
    Identify landmarks based on dispersion metrics and distance from DMSO.
    These landmarks must come from the reference dataset ONLY.
    
    Parameters:
    -----------
    mad_df : pd.DataFrame
        DataFrame containing dispersion metrics for REFERENCE treatments only
    distance_df : pd.DataFrame
        DataFrame containing distance from DMSO values for REFERENCE treatments only
    config : dict
        Configuration dictionary with parameters:
        - metric_type: Metric to use for landmark selection ('mad_cosine', 'var_cosine', 'std_cosine')
        - metric_threshold: Threshold for metric values (default thresholds based on metric type)
        - dmso_threshold_percentile: Percentile for DMSO threshold (default: '99')
        - output_dir: Directory to save outputs (optional)
        - plate_definitions: Plate metadata (optional)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing identified landmarks from the reference set
    """
    try:
        log_section("IDENTIFYING LANDMARKS FROM REFERENCE SET")
        log_info("STARTING LANDMARK IDENTIFICATION - REFERENCE SET ONLY")
        
        # Add explicit check that we're only using reference data
        if 'is_reference' in mad_df.columns:
            # Filter to ensure we only use reference data if the column exists
            reference_only = mad_df[mad_df['is_reference'] == True]
            if len(reference_only) < len(mad_df):
                log_info(f"Filtered {len(mad_df) - len(reference_only)} non-reference rows from mad_df")
                mad_df = reference_only
                
        if 'is_reference' in distance_df.columns:
            reference_only = distance_df[distance_df['is_reference'] == True]
            if len(reference_only) < len(distance_df):
                log_info(f"Filtered {len(distance_df) - len(reference_only)} non-reference rows from distance_df")
                distance_df = reference_only
        
        # Add detailed debugging for the input dataframes
        log_info(f"mad_df columns: {mad_df.columns.tolist()}")
        log_info(f"distance_df columns: {distance_df.columns.tolist()}")

        # Add the metric_name_mapping here
        metric_name_mapping = {
            'mad_cosine': 'mad_cosine',
            'std_cosine': 'stddev_cosine',  # Maps std_cosine to stddev_cosine in filenames
            'var_cosine': 'variance_cosine'  # Maps var_cosine to variance_cosine in filenames
        }

        # Check if specific metric columns exist
        for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
            log_info(f"Metric '{metric}' exists in mad_df: {metric in mad_df.columns}")
        
        log_info(f"Available metrics in mad_df: {[col for col in mad_df.columns if col in ['mad_cosine', 'var_cosine', 'std_cosine']]}")
        
        # Skip if either dataframe is None
        if mad_df is None or distance_df is None:
            log_info("Cannot identify landmarks - missing required data")
            return None
        
        # Merge metrics and distance dataframes
        log_info("Merging metrics and distance data...")
        merged = pd.merge(mad_df, distance_df, on='treatment', how='inner', suffixes=('', '_y'))

        # ADD THIS CRITICAL FILTERING SECTION:
        # ============ START OF FIX ============
        log_info("Filtering out invalid treatments before landmark selection...")
        initial_count = len(merged)
        
        # Remove invalid treatments
        # 1. Remove rows where treatment is NaN or empty
        merged = merged[merged['treatment'].notna()]
        merged = merged[merged['treatment'] != '']
        
        # 2. Remove treatments that are just "nan@" followed by concentration
        invalid_pattern = r'^nan@|^NaN@|^none@|^None@'
        merged = merged[~merged['treatment'].str.match(invalid_pattern, case=False, na=False)]
        
        # 3. Remove treatments that contain only concentration (e.g., just "@10.0")
        merged = merged[~merged['treatment'].str.match(r'^@[\d.]+$', na=False)]
        
        # 4. Also check compound_name if available - landmarks should have valid compound names
        if 'compound_name' in merged.columns:
            merged = merged[merged['compound_name'].notna()]
            merged = merged[~merged['compound_name'].str.lower().isin(['nan', 'none', 'unknown', ''])]
        
        final_count = len(merged)
        log_info(f"Removed {initial_count - final_count} invalid treatments")
        log_info(f"Remaining valid treatments for landmark selection: {final_count}")
        
        # Log some examples of what remains
        if len(merged) > 0:
            log_info("Sample valid treatments for landmark selection:")
            for i, treatment in enumerate(merged['treatment'].head(5)):
                compound = merged.iloc[i]['compound_name'] if 'compound_name' in merged.columns else 'N/A'
                log_info(f"  {i+1}. Treatment: '{treatment}', Compound: '{compound}'")

        # Add more debugging after the merge
        log_info("DIRECT METRIC CHECK")
        log_info(f"'mad_cosine' in merged columns: {'mad_cosine' in merged.columns}")
        log_info(f"'var_cosine' in merged columns: {'var_cosine' in merged.columns}")
        log_info(f"'std_cosine' in merged columns: {'std_cosine' in merged.columns}")
        
        # Remove duplicate columns
        duplicate_cols = [col for col in merged.columns if col.endswith('_y')]
        merged = merged.drop(columns=duplicate_cols)
        
        log_info(f"Merged data has {len(merged)} rows")

        # Get the primary metric to use for landmark selection from config
        primary_metric = config.get('metric_type', 'mad_cosine')
        
        # Set default metric if the specified one doesn't exist in the data
        if primary_metric not in merged.columns and 'mad_cosine' in merged.columns:
            log_info(f"Warning: Metric '{primary_metric}' not found in data. Falling back to 'mad_cosine'")
            primary_metric = 'mad_cosine'
        elif primary_metric not in merged.columns:
            log_info(f"Error: Metric '{primary_metric}' not found in data and no fallback available")
            return None
            
        log_info(f"Using {primary_metric} for primary landmark selection")
        
        # Define thresholds for landmark selection based on metric type
        thresholds = {
            'mad_cosine': config.get('mad_threshold', 0.05),
            'var_cosine': config.get('var_threshold', 0.005),  # Lower default for variance
            'std_cosine': config.get('std_threshold', 0.07)    # Default for std dev
        }
        
        log_info(f"Metric thresholds: {thresholds}")

        # Define DMSO distance threshold
        dmso_threshold_percentile = config.get('dmso_threshold_percentile', '99')
        dmso_col = f"exceeds_{dmso_threshold_percentile.replace('.', '_')}"

        # Select landmarks using the primary metric
        if dmso_col not in merged.columns:
            log_info(f"Warning: DMSO threshold column '{dmso_col}' not found. Using distance directly.")
            dmso_dist_threshold = config.get('dmso_distance_threshold', 0.2)
            landmarks = merged[(merged[primary_metric] <= thresholds[primary_metric]) & 
                            (merged['cosine_distance_from_dmso'] >= dmso_dist_threshold)]
        else:
            landmarks = merged[(merged[primary_metric] <= thresholds[primary_metric]) & 
                            (merged[dmso_col] == True)]

        log_info(f"Identified {len(landmarks)} potential landmarks using {primary_metric}")

        # Sort landmarks by the selected metric
        landmarks = landmarks.sort_values(primary_metric, ascending=True)
        
        # Add explicit column to mark these as reference landmarks
        landmarks['is_reference'] = True
        
        # Save landmarks based on primary metric first
        if 'output_dir' in config:
            output_path = dir_paths['analysis']['root'] / f'landmarks_{primary_metric}.csv'
            landmarks.to_csv(output_path, index=False)
            log_info(f"Saved primary landmarks to: {output_path}")

        # Generate landmark selection plots for all available metrics regardless of which is used for selection
        if 'output_dir' in config and len(merged) > 0:
            try:
                selection_plot_dir = dir_paths['visualizations']['landmark_selection']
                selection_plot_dir.mkdir(exist_ok=True)
                log_info(f"Created directory for landmark selection plots: {selection_plot_dir}")
                
                # Find all available metrics in the data
                log_info("==== AVAILABLE METRICS FOR PLOTTING ====")
                log_info(f"Checking for metrics in merged.columns: {merged.columns.tolist()}")
                for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
                    log_info(f"Metric '{metric}' exists: {metric in merged.columns}")
                
                # Add more explicit logging:
                log_info("==== AVAILABLE METRICS FOR PLOTTING ====")
                log_info(f"Checking for metrics in merged.columns: {merged.columns.tolist()}")
                for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
                    log_info(f"Metric '{metric}' exists: {metric in merged.columns}")
                # With this:
                log_info("FORCING ALL METRICS")
                all_metrics_direct = []
                for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
                    exists = metric in merged.columns
                    log_info(f"Direct check: '{metric}' exists in merged: {exists}")
                    if exists:
                        all_metrics_direct.append(metric)
                    else:
                        log_info(f"WARNING: '{metric}' not found in merged dataframe columns!")

                # If we found metrics, use them
                if all_metrics_direct:
                    all_metrics = all_metrics_direct
                    log_info(f"Using directly checked metrics: {all_metrics}")
                else:
                    # Fallback to just mad_cosine to avoid errors
                    all_metrics = ['mad_cosine'] if 'mad_cosine' in merged.columns else []
                    log_info(f"Falling back to just: {all_metrics}")
                    log_info("==== STARTING METRIC PLOT GENERATION ====")

                # Also save landmarks for each available metric
                for metric in all_metrics:
                    if metric != primary_metric:  # Skip the primary one we already saved
                        # Create landmark selection for this metric
                        curr_threshold = thresholds.get(metric, 0.05)
                        if dmso_col not in merged.columns:
                            curr_landmarks = merged[(merged[metric] <= curr_threshold) & 
                                                (merged['cosine_distance_from_dmso'] >= dmso_dist_threshold)]
                        else:
                            curr_landmarks = merged[(merged[metric] <= curr_threshold) & 
                                                (merged[dmso_col] == True)]
                        
                        # Save these landmarks
                        metric_output_path = dir_paths['analysis']['root'] / f'landmarks_{metric}.csv'
                        curr_landmarks.to_csv(metric_output_path, index=False)
                        log_info(f"Saved {metric} landmarks to: {metric_output_path}")

                # Add a sample of data for each metric
                for metric in all_metrics:
                    log_info(f"Sample values for {metric}:")
                    sample_data = merged[metric].head(5).tolist()
                    log_info(f"  {sample_data}")
                
                # For each metric, generate selection plots
                for curr_metric in all_metrics:
                    log_info(f"Creating landmark selection plots for {curr_metric}")
                    log_info(f"Number of values for {curr_metric}: {merged[curr_metric].count()}")
                    log_info(f"Range of values: min={merged[curr_metric].min()}, max={merged[curr_metric].max()}")
                    
                    # Get threshold for this metric
                    curr_threshold = thresholds.get(curr_metric, 0.05)
                    
                    # Create landmark selection based on this metric (for highlighting in the plot)
                    if dmso_col not in merged.columns:
                        curr_landmarks = merged[(merged[curr_metric] <= curr_threshold) & 
                                            (merged['cosine_distance_from_dmso'] >= dmso_dist_threshold)]
                    else:
                        curr_landmarks = merged[(merged[curr_metric] <= curr_threshold) & 
                                            (merged[dmso_col] == True)]
                    
                    log_info(f"Identified {len(curr_landmarks)} potential landmarks using {curr_metric}")
                    
                    # Define consistent hover columns with formatting
                    hover_columns = {
                        'treatment': ('Treatment', None),
                        'library': ('Library', None),
                        'moa': ('MOA', None),
                        'cosine_distance_from_dmso': ('Distance from DMSO', ':.3f'),
                        curr_metric: (curr_metric.replace('_', ' ').title(), ':.3f'),
                        'plate': ('Plate', None),
                        'well': ('Well', None)
                    }
        
                    # Add other metrics to hover data if they exist
                    for other_metric in all_metrics:
                        if other_metric != curr_metric:
                            hover_columns[other_metric] = (other_metric.replace('_', ' ').title(), ':.3f')
                    
                    # Prepare hover data - ensure all columns exist
                    hover_data = {}
                    for col, (display_name, fmt) in hover_columns.items():
                        if col in merged.columns:
                            hover_data[col] = merged[col]
                        else:
                            merged[col] = 'N/A'
                            hover_data[col] = merged[col]
                    
                    # Create hover template
                    hover_template_parts = []
                    for i, (col, (display_name, fmt)) in enumerate(hover_columns.items()):
                        if fmt:
                            hover_template_parts.append(f"<b>{display_name}</b>: %{{customdata[{i}]:{fmt}}}")
                        else:
                            hover_template_parts.append(f"<b>{display_name}</b>: %{{customdata[{i}]}}")
                    
                    hover_template = "<br>".join(hover_template_parts) + "<extra></extra>"
                    
                    # Truncate long text fields for display
                    merged['treatment_short'] = merged['treatment'].apply(
                        lambda x: (str(x)[:25] + '...') if len(str(x)) > 25 else str(x)
                    )
                    merged['moa_short'] = merged['moa'].astype(str).apply(
                        lambda x: (x[:25] + '...') if len(x) > 25 else x
                    )
                    
                    # Determine coloring strategy
                    color_col = None
                    if 'plate_definitions' in config:
                        merged['library'] = merged['plate'].apply(
                            lambda x: config['plate_definitions'].get(str(x), {}).get('library', 'Unknown')
                        )
                        color_col = 'library'
                    elif 'moa' in merged.columns:
                        color_col = 'moa_short'
                    
                    # Calculate DMSO threshold if not provided
                    dmso_threshold_percentile = config.get('dmso_threshold_percentile', '99')
                    dmso_threshold = None

                    # Try to get threshold from config first (preferred method)
                    if 'dmso_thresholds' in config and dmso_threshold_percentile in config['dmso_thresholds']:
                        dmso_threshold = config['dmso_thresholds'][dmso_threshold_percentile]
                        log_info(f"PLOT: Using stored {dmso_threshold_percentile}% DMSO threshold: {dmso_threshold:.4f}")
                    else:
                        log_info(f"ERROR: No DMSO {dmso_threshold_percentile}% threshold found in config. Skipping threshold line.")
                        dmso_threshold = None
                    
                    # Create base figure function
                    def create_landmark_figure(color_by=None, title_suffix=""):
                        fig = px.scatter(
                            merged,
                            x='cosine_distance_from_dmso',
                            y=curr_metric,
                            color=color_by,
                            custom_data=list(hover_data.values()),
                            title=f'Landmark Selection: {curr_metric.replace("_", " ").title()} vs DMSO Distance {title_suffix}',
                            hover_name='treatment_short'
                        )
                        
                        # Apply consistent hover template
                        fig.update_traces(hovertemplate=hover_template)
                        
                        # Add threshold lines
                        if dmso_threshold is not None:
                            fig.add_shape(
                                type='line',
                                x0=dmso_threshold,
                                x1=dmso_threshold,
                                y0=0,
                                y1=merged[curr_metric].max(),
                                line=dict(color='orange', width=2, dash='dash'),
                                name=f'DMSO {dmso_threshold_percentile}%ile: {dmso_threshold:.4f}'
                            )
                            log_info(f"PLOT: Added DMSO threshold line at x={dmso_threshold:.4f}")
                        else:
                            log_info("PLOT: Skipped DMSO threshold line - no threshold value available")
                        
                        fig.add_shape(
                            type='line',
                            x0=0,
                            x1=merged['cosine_distance_from_dmso'].max(),
                            y0=curr_threshold,
                            y1=curr_threshold,
                            line=dict(color='red', width=2, dash='dash'),
                            name=f'{curr_metric} Threshold'
                        )
                        
                        # Highlight landmarks if any
                        if len(curr_landmarks) > 0:
                            landmark_data = [curr_landmarks[col].values for col in hover_data.keys()]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=curr_landmarks['cosine_distance_from_dmso'],
                                    y=curr_landmarks[curr_metric],
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color='red',
                                        line=dict(width=1.5, color='black')
                                    ),
                                    name='Landmarks',
                                    customdata=np.column_stack(landmark_data),
                                    hovertemplate=hover_template,
                                    hoverlabel=dict(
                                        bgcolor='white',
                                        font_size=12,
                                        font_family="Arial"
                                    )
                                )
                            )
                        
                        # Improve layout
                        fig.update_layout(
                            xaxis_title='Cosine Distance from DMSO',
                            yaxis_title=curr_metric.replace('_', ' ').title(),
                            legend_title=color_by.replace('_', ' ').title() if color_by else '',
                            width=1440,
                            height=800,
                            legend=dict(
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02,
                                itemsizing="constant",
                                font=dict(size=10)
                            ),
                            margin=dict(r=200, t=50),
                            hovermode='closest'
                        )
                        
                        return fig
                    
                    # Create and save library plot if available
                    if color_col == 'library':
                        fig_library = create_landmark_figure(color_by='library', title_suffix="(Colored by Library)")
                        mapped_metric_name = metric_name_mapping.get(curr_metric, curr_metric)
                        lib_html_path = selection_plot_dir / f'landmark_selection_{mapped_metric_name}_by_library.html'
                        fig_library.write_html(lib_html_path)  # This was missing
                        log_info(f"Saved interactive landmark selection by library plot to: {lib_html_path}")
                    
                    # Create and save MOA plot if MOA data exists
                    if 'moa' in merged.columns:
                        # Clean MOA data first
                        merged['moa'] = merged['moa'].fillna('Unknown').astype(str)
                        
                        # Create shortened version for display
                        merged['moa_short'] = merged['moa'].apply(
                            lambda x: (x[:25] + '...') if len(x) > 25 else x
                        )
                        
                        # Generate the plot
                        fig_moa = create_landmark_figure(
                            color_by='moa_short', 
                            title_suffix="(Colored by MOA)"
                        )
                        
                        # Customize MOA plot further if needed
                        fig_moa.update_layout(
                            legend_title_text='MOA',
                            legend=dict(
                                itemsizing='constant',
                                itemwidth=30  # Makes legend items more compact
                            )
                        )
                        
                        mapped_metric_name = metric_name_mapping.get(curr_metric, curr_metric)
                        moa_html_path = selection_plot_dir / f'landmark_selection_{mapped_metric_name}_by_moa.html'
                        fig_moa.write_html(moa_html_path)
                        log_info(f"Saved interactive landmark selection by MOA plot to: {moa_html_path}")
                    else:
                        log_info(f"Skipping MOA-colored plot for {curr_metric} - 'moa' column not found")
                    
                    # Create generic plot if no specific coloring
                    if color_col is None:
                        fig_generic = create_landmark_figure(title_suffix="")
                        mapped_metric_name = metric_name_mapping.get(curr_metric, curr_metric)
                        generic_html_path = selection_plot_dir / f'landmark_selection_{mapped_metric_name}.html'
                        fig_generic.write_html(generic_html_path)
                        log_info(f"Saved interactive landmark selection plot to: {generic_html_path}")
                
            except Exception as e:
                log_info(f"Error creating interactive plots: {str(e)}")
                log_info("Make sure plotly is installed: pip install plotly")

        return landmarks

    except Exception as e:
        log_info(f"ERROR IN LANDMARK IDENTIFICATION: {str(e)}")
        import traceback
        log_info(traceback.format_exc())
        return None

def generate_landmark_vs_dmso_distance_plots(scores_df, config, dir_paths, is_reference=False):
    """
    Generate Landmark Distance vs DMSO Distance plots with multiple coloring metrics.
    Creates separate plots for each scoring method (3-term harmonic mean, 2-term harmonic mean, ratio score)
    for each dispersion metric (mad_cosine, var_cosine, std_cosine).
    
    Args:
        scores_df: DataFrame with scores and distances
        config: Configuration dictionary
        dir_paths: Dictionary with directory paths
        is_reference: Boolean indicating if this is reference data
    """
    log_section(f"GENERATING LANDMARK VS DMSO DISTANCE PLOTS {'(REFERENCE)' if is_reference else '(TEST)'}")
    
    # Add these diagnostic logs
    log_info(f"Total number of treatments in scores: {len(scores_df['treatment'].unique())}")
    log_info(f"Sample of first 10 treatments: {scores_df['treatment'].unique()[:10]}")
    log_info(f"Unique libraries: {scores_df['library'].unique() if 'library' in scores_df.columns else 'No library column'}")

    # Log all column names
    log_info("Columns in scores DataFrame:")
    for col in scores_df.columns:
        log_info(f"  {col}: {scores_df[col].count()} non-null values")
    
    # Add a try-except block to get more error information
    try:
        # For each metric type
        metrics = ['mad_cosine', 'var_cosine', 'std_cosine']
        for metric in metrics:
            # Find score columns for this metric
            score_columns = [
                f'harmonic_mean_3term_{metric}', 
                f'harmonic_mean_2term_{metric}', 
                'ratio_score'
            ]
            
            # Log detailed information about each score column
            for col in score_columns:
                if col in scores_df.columns:
                    log_info(f"Column {col}:")
                    log_info(f"  Total values: {len(scores_df[col])}")
                    log_info(f"  Non-null values: {scores_df[col].count()}")
                    log_info(f"  Range: min={scores_df[col].min()}, max={scores_df[col].max()}")
                    log_info(f"  Sample values: {scores_df[col].head().tolist()}")
                else:
                    log_info(f"Column {col} not found in dataframe")
    
    except Exception as e:
        log_info(f"Error during diagnostic logging: {str(e)}")
        import traceback
        log_info(traceback.format_exc())
    
    # Validate input
    if scores_df is None:
        log_info("ERROR: scores_df is None. Cannot generate landmark distance plots.")
        return
        
    # Use dir_paths for output directory
    landmark_plot_dir = dir_paths['visualizations']['landmark_plots']['root']
    log_info(f"Using landmark plots directory: {landmark_plot_dir}")
    
    # Get the reference/test subfolder for the combined plots
    ref_or_test = 'reference' if is_reference else 'test'
    combined_dir = dir_paths['visualizations']['landmark_plots']['combined'][ref_or_test]
    
    # Ensure all directories exist
    for path_name in ['static', 'interactive']:
        legacy_dir = dir_paths['visualizations']['landmark_plots'][path_name]
        legacy_dir.mkdir(exist_ok=True, parents=True)
    
    combined_dir.mkdir(exist_ok=True, parents=True)
    
    # Detailed data logging
    log_info("\n--- DETAILED DATA ANALYSIS ---")
    log_info(f"Total number of rows: {len(scores_df)}")
    
    # Define all available metrics
    metrics = []
    for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
        # Check for needed columns
        if any(col.endswith(metric) for col in scores_df.columns):
            metrics.append(metric)
    
    log_info(f"Found metrics for plotting: {metrics}")
    
    # For each metric, define which score columns to use
    for metric in metrics:
        log_info(f"\nProcessing plots for metric: {metric}")
        
        # Get metric-specific directories
        metric_dir = dir_paths['visualizations']['landmark_plots']['by_metric'][metric]
        static_dir = metric_dir['static'] 
        interactive_dir = metric_dir['interactive']
        
        # Ensure directories exist
        static_dir.mkdir(exist_ok=True, parents=True)
        interactive_dir.mkdir(exist_ok=True, parents=True)
        
        # Define scoring metrics for this dispersion metric
        color_metrics = []
        
        # Check which score columns exist for this metric
        if f'harmonic_mean_3term_{metric}' in scores_df.columns:
            color_metrics.append((f'harmonic_mean_3term_{metric}', f'Harmonic Mean Score (3-term, {metric})'))
        elif metric == 'mad_cosine' and 'harmonic_mean_3term' in scores_df.columns:
            # For backward compatibility with original harmonic_mean_3term (which uses mad_cosine)
            color_metrics.append(('harmonic_mean_3term', 'Harmonic Mean Score (3-term)'))
        
        if f'harmonic_mean_2term_{metric}' in scores_df.columns:
            color_metrics.append((f'harmonic_mean_2term_{metric}', f'Harmonic Mean Score (2-term, {metric})'))
        elif metric == 'mad_cosine' and 'harmonic_mean_2term' in scores_df.columns:
            # For backward compatibility with original harmonic_mean_2term
            color_metrics.append(('harmonic_mean_2term', 'Harmonic Mean Score (2-term)'))
        
        # Ratio score is the same for all metrics
        if 'ratio_score' in scores_df.columns:
            color_metrics.append(('ratio_score', 'Ratio Score'))
        
        if not color_metrics:
            log_info(f"No score columns found for {metric}, skipping...")
            continue
            
        log_info(f"Found {len(color_metrics)} score types for {metric}: {[m[0] for m in color_metrics]}")
    
        # Required columns
        required_columns = ['cosine_distance_from_dmso', 'closest_landmark_distance']
            
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in scores_df.columns]
        if missing_columns:
            log_info(f"ERROR: Missing required columns: {missing_columns}")
            log_info("Available columns:")
            for col in scores_df.columns:
                log_info(f"  {col}")
            return
    
        # Log closest landmark columns if available
        landmark_columns = [col for col in scores_df.columns if 'landmark' in col.lower()]
        log_info(f"Available landmark columns: {landmark_columns}")
    
        # Log column statistics
        for col in required_columns + [metric_col[0] for metric_col in color_metrics]:
            if col in scores_df.columns:
                log_info(f"\nColumn Analysis for {col}:")
                # Basic statistics
                try:
                    log_info(f"  Total values: {len(scores_df[col])}")
                    log_info(f"  Non-null values: {scores_df[col].count()}")
                    log_info(f"  Null values: {scores_df[col].isnull().sum()}")
                    
                    if pd.api.types.is_numeric_dtype(scores_df[col]):
                        log_info(f"  Data type: {scores_df[col].dtype}")
                        log_info(f"  Min value: {scores_df[col].min()}")
                        log_info(f"  Max value: {scores_df[col].max()}")
                        log_info(f"  Mean: {scores_df[col].mean()}")
                        log_info(f"  Median: {scores_df[col].median()}")
                except Exception as e:
                    log_info(f"  Error analyzing column {col}: {str(e)}")
        
        # For each color metric, create static and interactive plots
        for color_col, color_label in color_metrics:
            if color_col in scores_df.columns:
                log_info(f"\nGenerating plots with {color_label} coloring")
                
                # Static Plot Creation
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Filter out NaN values for this metric
                    valid_data = scores_df.dropna(subset=[color_col])
                    
                    if len(valid_data) > 0:
                        scatter = plt.scatter(
                            valid_data['cosine_distance_from_dmso'],
                            valid_data['closest_landmark_distance'],
                            c=valid_data[color_col],
                            cmap='viridis',
                            alpha=0.7
                        )
                        plt.colorbar(scatter, label=color_label)
                        log_info(f"  Created scatter plot with {len(valid_data)} data points")
                    else:
                        plt.scatter([], [], color='blue')
                        log_info("  WARNING: No valid data points for this metric")
                    
                    plt.xlabel('Cosine Distance from DMSO')
                    plt.ylabel('Distance to Closest Landmark')
                    plt.title(f'{"Reference" if is_reference else "Test"}: Landmark vs DMSO Distance\nColored by {color_label}')
                    
                    # Add DMSO distance threshold if available
                    if 'dmso_thresholds' in config:
                        threshold_key = config.get('dmso_threshold_percentile', '99')
                        if threshold_key in config['dmso_thresholds']:
                            threshold = config['dmso_thresholds'][threshold_key]
                            plt.axvline(threshold, color='red', linestyle='--', 
                                        label=f"DMSO {threshold_key}% threshold: {threshold:.4f}")
                    
                    # Add similarity threshold if available
                    if 'similarity_threshold' in config:
                        threshold = config['similarity_threshold']
                        plt.axhline(threshold, color='orange', linestyle='--',
                                    label=f"Similarity threshold: {threshold:.4f}")
                    
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # File name for static plot
                    static_plot_name = f'landmark_vs_dmso_{color_col}_{ref_or_test}.png'
                    
                    # Save static plot to metric-specific directory
                    static_plot_path = static_dir / static_plot_name
                    plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
                    log_info(f"  Saved static plot to: {static_plot_path}")
                    
                    # Also save to legacy path for backward compatibility
                    legacy_static_path = dir_paths['visualizations']['landmark_plots']['static'] / static_plot_name
                    plt.savefig(legacy_static_path, dpi=300, bbox_inches='tight')
                    log_info(f"  Also saved to legacy path: {legacy_static_path}")
                    
                    plt.close()
                except Exception as e:
                    log_info(f"  ERROR creating static plot for {color_label}: {str(e)}")
                
                # Interactive Plot Creation
                try:
                    # Create interactive plot 
                    valid_data = scores_df.dropna(subset=[color_col])
                    
                    if len(valid_data) > 0:
                        # Determine hover columns based on which metrics are available
                        base_hover_columns = [
                            'treatment', 'compound_name', 'compound_uM', 'moa', 'library',
                            'cosine_distance_from_dmso', 'closest_landmark_distance',
                            'closest_landmark', 'second_closest_landmark', 'third_closest_landmark',
                            'ratio_score'
                        ]
                        
                        # Add metric-specific columns
                        for m in metrics:
                            if m in valid_data.columns:
                                base_hover_columns.append(m)
                            
                            # Add harmonic mean scores for this metric
                            for score_type in [f'harmonic_mean_2term_{m}', f'harmonic_mean_3term_{m}']:
                                if score_type in valid_data.columns:
                                    base_hover_columns.append(score_type)
                        
                        # Add original harmonic means for backward compatibility
                        for orig_score in ['harmonic_mean_2term', 'harmonic_mean_3term']:
                            if orig_score in valid_data.columns:
                                base_hover_columns.append(orig_score)
                        
                        # Filter to only columns that actually exist
                        hover_columns = [col for col in base_hover_columns if col in valid_data.columns]

                        # Build hover_data dict with custom formatting
                        hover_data = {}
                        for col in hover_columns:
                            if col in valid_data.columns:
                                # Format numeric columns with precision
                                if col in metrics or 'cosine_distance' in col or 'landmark_distance' in col or 'harmonic_mean' in col or 'ratio_score' == col or 'compound_uM' == col:
                                    hover_data[col] = ':.4f'
                                else:
                                    hover_data[col] = True
                            
                        # Create labels dictionary for plot
                        labels = {
                            'cosine_distance_from_dmso': 'Cosine Distance from DMSO',
                            'closest_landmark_distance': 'Distance to Closest Landmark',
                            'closest_landmark': 'Closest Landmark',
                            'second_closest_landmark': 'Second Closest Landmark',
                            'third_closest_landmark': 'Third Closest Landmark',
                            color_col: color_label,
                            'treatment': 'Treatment',
                            'compound_name': 'Compound Name',
                            'compound_uM': 'Concentration (µM)',
                            'moa': 'Mechanism of Action',
                            'library': 'Library',
                            'ratio_score': 'Ratio Score'
                        }
                        
                        # Add labels for all metrics
                        for m in metrics:
                            labels[m] = f'{m.replace("_", " ").title()}'
                            labels[f'harmonic_mean_2term_{m}'] = f'2-term Harmonic Mean ({m})'
                            labels[f'harmonic_mean_3term_{m}'] = f'3-term Harmonic Mean ({m})'
                        
                        # Add original harmonic means labels
                        labels['harmonic_mean_2term'] = '2-term Harmonic Mean'
                        labels['harmonic_mean_3term'] = '3-term Harmonic Mean'
                        
                        fig = px.scatter(
                            valid_data,
                            x='cosine_distance_from_dmso',
                            y='closest_landmark_distance',
                            color=color_col,
                            color_continuous_scale='viridis',
                            hover_data=hover_data,
                            labels=labels,
                            title=f'{"Reference" if is_reference else "Test"}: Landmark vs DMSO Distance<br>Colored by {color_label}'
                        )
                        
                        # Add threshold lines if available
                        if 'dmso_thresholds' in config:
                            threshold_key = config.get('dmso_threshold_percentile', '99')
                            if threshold_key in config['dmso_thresholds']:
                                threshold = config['dmso_thresholds'][threshold_key]
                                fig.add_vline(
                                    x=threshold, 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text=f"DMSO {threshold_key}% threshold"
                                )
                        
                        if 'similarity_threshold' in config:
                            threshold = config['similarity_threshold']
                            fig.add_hline(
                                y=threshold, 
                                line_dash="dash", 
                                line_color="orange",
                                annotation_text='Similarity threshold'
                            )
                        
                        # File name for interactive plot
                        html_plot_name = f'landmark_vs_dmso_{color_col}_{ref_or_test}_interactive.html'
                        
                        # Save interactive plot to metric-specific directory
                        html_plot_path = interactive_dir / html_plot_name
                        fig.write_html(html_plot_path)
                        log_info(f"  Saved interactive plot to: {html_plot_path}")
                        
                        # Also save to legacy path for backward compatibility
                        legacy_html_path = dir_paths['visualizations']['landmark_plots']['interactive'] / html_plot_name
                        fig.write_html(legacy_html_path)
                        log_info(f"  Also saved to legacy path: {legacy_html_path}")
                    else:
                        log_info(f"  WARNING: No valid data points for interactive plot with {color_label}")
                
                except ImportError:
                    log_info(f"  ERROR: Plotly not installed. Cannot create interactive plot for {color_label}.")
                except Exception as e:
                    log_info(f"  ERROR creating interactive plot for {color_label}: {str(e)}")
            else:
                log_info(f"Skipping {color_label} plot - column not found in data")
        
        # Generate a combined interactive plot with dropdown for different coloring for this metric
        try:
            # Check if we have any valid metrics for this metric type
            available_metrics = [cmetric for cmetric, _ in color_metrics if cmetric in scores_df.columns]
            
            if available_metrics:
                log_info(f"\nGenerating combined interactive plot with metric selector for {metric}")
                
                fig = go.Figure()
                
                # Add a trace for each metric
                for cmetric, label in color_metrics:
                    if cmetric in scores_df.columns:
                        valid_data = scores_df.dropna(subset=[cmetric])
                        
                        if len(valid_data) > 0:
                            hover_text = []
                            for _, row in valid_data.iterrows():
                                hover_info = []
                                
                                # Add compound information
                                if 'compound_name' in valid_data.columns:
                                    hover_info.append(f"Compound: {row['compound_name'] if not pd.isna(row['compound_name']) else 'Unknown'}")
                                if 'treatment' in valid_data.columns:
                                    hover_info.append(f"Treatment: {row['treatment'] if not pd.isna(row['treatment']) else 'Unknown'}")
                                if 'compound_uM' in valid_data.columns:
                                    hover_info.append(f"Concentration: {row['compound_uM']:.4f} µM" if not pd.isna(row['compound_uM']) else "Concentration: Unknown")
                                if 'moa' in valid_data.columns:
                                    hover_info.append(f"MOA: {row['moa'] if not pd.isna(row['moa']) else 'Unknown'}")
                                if 'library' in valid_data.columns:
                                    hover_info.append(f"Library: {row['library'] if not pd.isna(row['library']) else 'Unknown'}")
                                
                                # Add landmark information
                                if 'closest_landmark' in valid_data.columns:
                                    hover_info.append(f"Closest landmark: {row['closest_landmark'] if not pd.isna(row['closest_landmark']) else 'Unknown'}")
                                if 'second_closest_landmark' in valid_data.columns:
                                    hover_info.append(f"Second closest: {row['second_closest_landmark'] if not pd.isna(row['second_closest_landmark']) else 'Unknown'}")
                                if 'third_closest_landmark' in valid_data.columns:
                                    hover_info.append(f"Third closest: {row['third_closest_landmark'] if not pd.isna(row['third_closest_landmark']) else 'Unknown'}")
                                
                                # Add metrics
                                for disp_metric in metrics:
                                    if disp_metric in valid_data.columns:
                                        hover_info.append(f"{disp_metric.replace('_', ' ').title()}: {row[disp_metric]:.4f}" if not pd.isna(row[disp_metric]) else f"{disp_metric.replace('_', ' ').title()}: Unknown")
                                
                                hover_info.append(f"DMSO Distance: {row['cosine_distance_from_dmso']:.4f}" if not pd.isna(row['cosine_distance_from_dmso']) else "DMSO Distance: Unknown")
                                hover_info.append(f"Landmark Distance: {row['closest_landmark_distance']:.4f}" if not pd.isna(row['closest_landmark_distance']) else "Landmark Distance: Unknown")
                                
                                # Add scores
                                for score_col, score_label in color_metrics:
                                    if score_col in valid_data.columns:
                                        hover_info.append(f"{score_label}: {row[score_col]:.4f}" if not pd.isna(row[score_col]) else f"{score_label}: Unknown")
                                
                                # Current metric being displayed
                                hover_info.append(f"<b>{label}: {row[cmetric]:.4f}</b>" if not pd.isna(row[cmetric]) else f"<b>{label}: Unknown</b>")
                                
                                hover_text.append("<br>".join(hover_info))

                            fig.add_trace(
                                go.Scatter(
                                    x=valid_data['cosine_distance_from_dmso'],
                                    y=valid_data['closest_landmark_distance'],
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=valid_data[cmetric],
                                        colorscale='Viridis',
                                        colorbar=dict(title=label),
                                        showscale=True
                                    ),
                                    text=hover_text,
                                    hoverinfo='text',
                                    name=label,
                                    visible=(cmetric == available_metrics[0])  # Only first one visible initially
                                )
                            )
                                        
                # Add buttons for switching between metrics
                buttons = []
                for i, (cmetric, label) in enumerate([(m, l) for m, l in color_metrics if m in available_metrics]):
                    visibility = [j == i for j in range(len(available_metrics))]
                    buttons.append(
                        dict(
                            label=label,
                            method="update",
                            args=[{"visible": visibility},
                                  {"title": f"{'Reference' if is_reference else 'Test'}: Landmark vs DMSO Distance<br>Colored by {label}"}]
                        )
                    )
                
                # Add dropdown menu
                fig.update_layout(
                    updatemenus=[
                        dict(
                            active=0,
                            buttons=buttons,
                            direction="down",
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=1.0,
                            xanchor="right",
                            y=1.2,
                            yanchor="top"
                        ),
                    ],

                    title=f"{'Reference' if is_reference else 'Test'}: Landmark vs DMSO Distance ({metric})<br>Select coloring metric",
                    xaxis_title="Cosine Distance from DMSO",
                    yaxis_title="Distance to Closest Landmark",
                    height=700,
                    width=1000,
                    margin=dict(t=60, b=40, l=40, r=120)  # Add this line for margins
                )
                
                # Add threshold lines
                if 'dmso_thresholds' in config:
                    threshold_key = config.get('dmso_threshold_percentile', '99')
                    if threshold_key in config['dmso_thresholds']:
                        threshold = config['dmso_thresholds'][threshold_key]
                        fig.add_vline(
                            x=threshold, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"DMSO {threshold_key}% threshold"
                        )
                
                if 'similarity_threshold' in config:
                    threshold = config['similarity_threshold']
                    fig.add_hline(
                        y=threshold, 
                        line_dash="dash", 
                        line_color="orange",
                        annotation_text='Similarity threshold'
                    )
                
                # File name for combined interactive plot
                html_plot_name = f'landmark_vs_dmso_{metric}_combined_{ref_or_test}_interactive.html'
                
                # Save combined interactive plot to reference/test specific combined directory
                html_plot_path = combined_dir / html_plot_name
                fig.write_html(html_plot_path)
                log_info(f"Saved combined interactive plot to: {html_plot_path}")
                
                # Also save to root directory for backward compatibility
                legacy_combined_path = dir_paths['visualizations']['landmark_plots']['root'] / html_plot_name
                fig.write_html(legacy_combined_path)
                log_info(f"Also saved to legacy path: {legacy_combined_path}")
            else:
                log_info(f"Skipping combined plot for {metric} - no valid metrics found")
        
        except ImportError:
            log_info(f"ERROR: Plotly not installed. Cannot create combined interactive plot for {metric}.")
        except Exception as e:
            log_info(f"ERROR creating combined interactive plot for {metric}: {str(e)}")
    
    log_info("Completed generating landmark vs DMSO distance plots")
