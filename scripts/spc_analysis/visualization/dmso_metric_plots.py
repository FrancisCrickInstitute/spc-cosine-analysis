# DMSO vs dispersion metric plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from ..utils.logging import log_info, log_section


# DMSO vs dispersion metric plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from ..utils.logging import log_info, log_section

def generate_dmso_vs_dispersion_plots(reference_scores=None, test_scores=None, config=None, dir_paths=None):
    """
    Generate visualizations comparing DMSO distance against dispersion metrics (MAD, variance, std deviation)
    with similar structure to the landmark distance plots.
    
    Args:
        reference_scores: DataFrame with reference set scores and metrics
        test_scores: DataFrame with test set scores and metrics
        config: Configuration dictionary
        dir_paths: Dictionary with directory paths
    """
    log_section("GENERATING DMSO DISTANCE VS DISPERSION METRICS PLOTS")
    
    # Create directory structure for these plots
    dmso_metric_dir = dir_paths['visualizations']['dmso_vs_metrics']['root']
    
    # Process reference scores
    if reference_scores is not None:
        log_info("\nGenerating DMSO vs. Metric plots for reference set")
        _generate_dmso_vs_metric_plots(reference_scores, config, dir_paths, is_reference=True)
    
    # Process test scores
    if test_scores is not None:
        log_info("\nGenerating DMSO vs. Metric plots for test set")
        _generate_dmso_vs_metric_plots(test_scores, config, dir_paths, is_reference=False)
    
    log_info("Completed generating DMSO vs. Dispersion Metric plots")


def _generate_dmso_vs_metric_plots(scores_df, config, dir_paths, is_reference=False):
    """
    Helper function to generate plots for a specific dataset (reference or test)
    
    Args:
        scores_df: DataFrame with scores and metrics
        config: Configuration dictionary
        dir_paths: Dictionary with directory paths
        is_reference: Boolean indicating if this is reference data
    """
    if scores_df is None:
        log_info("ERROR: scores_df is None. Cannot generate DMSO vs metric plots.")
        return
        
    # Set label for reference/test
    ref_or_test = 'reference' if is_reference else 'test'
    
    # Get the directory paths
    dmso_metric_dir = dir_paths['visualizations']['dmso_vs_metrics']
    static_dir = dmso_metric_dir['static']
    interactive_dir = dmso_metric_dir['interactive']
    combined_dir = dmso_metric_dir['combined'][ref_or_test]
    
    # Check if necessary columns exist
    log_info("Checking for required columns in scores dataframe")
    for col in ['cosine_distance_from_dmso']:
        if col not in scores_df.columns:
            log_info(f"ERROR: Missing required column: {col}")
            return
    
    # Check which dispersion metrics are available
    available_metrics = []
    for metric in ['normalized_mad_cosine', 'normalized_var_cosine', 'normalized_std_cosine', 
                  'mad_cosine', 'var_cosine', 'std_cosine']:
        if metric in scores_df.columns:
            available_metrics.append(metric)
    
    if not available_metrics:
        log_info("ERROR: No dispersion metrics found in scores dataframe")
        return
    
    log_info(f"Found {len(available_metrics)} metrics for plotting: {available_metrics}")
    
    # Define mapping from raw metrics to normalized metrics
    raw_to_normalized = {
        'mad_cosine': 'normalized_mad_cosine',
        'var_cosine': 'normalized_var_cosine',
        'std_cosine': 'normalized_std_cosine'
    }
    
    # Define mapping from metrics to display names
    metric_display_names = {
        'mad_cosine': 'MAD Cosine',
        'var_cosine': 'Variance Cosine',
        'std_cosine': 'Standard Deviation Cosine',
        'normalized_mad_cosine': 'Normalized MAD Cosine',
        'normalized_var_cosine': 'Normalized Variance Cosine',
        'normalized_std_cosine': 'Normalized Standard Deviation Cosine'
    }
    
    # Define core metrics to process
    core_metrics = ['mad_cosine', 'var_cosine', 'std_cosine']
    
    # Process each core metric
    for metric in core_metrics:
        # Check if we have the raw or normalized version (prioritize normalized)
        normalized_metric = raw_to_normalized.get(metric)
        if normalized_metric in scores_df.columns:
            y_metric = normalized_metric
            log_info(f"Using normalized metric {y_metric}")
        elif metric in scores_df.columns:
            y_metric = metric
            log_info(f"Using raw metric {y_metric}")
        else:
            log_info(f"Skipping {metric} - not found in dataframe")
            continue
        
        display_name = metric_display_names.get(y_metric)
        metric_short_name = metric.split('_')[0]  # 'mad', 'var', or 'std'
        
        # Get metric-specific directories
        metric_dir = dmso_metric_dir['by_metric'][metric]
        metric_static_dir = metric_dir['static']
        metric_interactive_dir = metric_dir['interactive']
        
        # Define color variables to use for this metric
        color_options = []
        
        # Add harmonic mean scores if available
        if f'harmonic_mean_3term_{metric}' in scores_df.columns:
            color_options.append((f'harmonic_mean_3term_{metric}', f'3-term Harmonic Mean ({metric_short_name.upper()})'))
        elif 'harmonic_mean_3term' in scores_df.columns and metric == 'mad_cosine':
            color_options.append(('harmonic_mean_3term', '3-term Harmonic Mean'))
            
        if f'harmonic_mean_2term_{metric}' in scores_df.columns:
            color_options.append((f'harmonic_mean_2term_{metric}', f'2-term Harmonic Mean ({metric_short_name.upper()})'))
        elif 'harmonic_mean_2term' in scores_df.columns and metric == 'mad_cosine':
            color_options.append(('harmonic_mean_2term', '2-term Harmonic Mean'))
            
        # Add library/MOA if available
        if 'library' in scores_df.columns:
            color_options.append(('library', 'Library'))
        if 'moa' in scores_df.columns:
            color_options.append(('moa', 'Mechanism of Action'))
        
        log_info(f"Using color options: {[opt[1] for opt in color_options]}")
        
        # Create static plot
        try:
            plt.figure(figsize=(12, 8))
            plt.scatter(
                scores_df['cosine_distance_from_dmso'],
                scores_df[y_metric],
                alpha=0.7
            )
            plt.xlabel('Cosine Distance from DMSO')
            plt.ylabel(display_name)
            plt.title(f'{ref_or_test.title()}: DMSO Distance vs {display_name}')
            plt.grid(True, alpha=0.3)
            
            # Add DMSO threshold if available
            if 'dmso_thresholds' in config:
                threshold_key = config.get('dmso_threshold_percentile', '99')
                if threshold_key in config['dmso_thresholds']:
                    threshold = config['dmso_thresholds'][threshold_key]
                    plt.axvline(threshold, color='red', linestyle='--', 
                              label=f"DMSO {threshold_key}% threshold: {threshold:.4f}")
                    plt.legend()
            
            # Save plot
            static_filename = f'dmso_vs_{metric_short_name}_{ref_or_test}.png'
            static_path = metric_static_dir / static_filename
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            log_info(f"Saved static plot to: {static_path}")
            
            # Also save to legacy static directory
            legacy_static_path = static_dir / static_filename
            plt.savefig(legacy_static_path, dpi=300, bbox_inches='tight')
            
            plt.close()
        except Exception as e:
            log_info(f"Error creating static plot for {display_name}: {str(e)}")
        
        # Create individual interactive plots for each color option
        for color_col, color_label in color_options:
            try:
                # Skip if color column doesn't exist
                if color_col not in scores_df.columns:
                    log_info(f"Skipping {color_label} - column not found in dataframe")
                    continue
                
                # Determine hover data columns - include as much information as possible
                hover_data = {}
                
                # Basic information
                for col in ['treatment', 'compound_name', 'compound_uM', 'moa', 'library', 'plate', 'well']:
                    if col in scores_df.columns:
                        hover_data[col] = True
                
                # Metric information
                for metric_col in ['mad_cosine', 'var_cosine', 'std_cosine', 
                                  'cosine_distance_from_dmso', 'closest_landmark_distance']:
                    if metric_col in scores_df.columns:
                        hover_data[metric_col] = ':.4f'
                
                # Landmark information
                for landmark_col in ['closest_landmark', 'second_closest_landmark', 'third_closest_landmark']:
                    if landmark_col in scores_df.columns:
                        hover_data[landmark_col] = True
                
                # Score information
                for score_col in [
                    'harmonic_mean_2term', 'harmonic_mean_3term',
                    f'harmonic_mean_2term_{metric}', f'harmonic_mean_3term_{metric}'
                ]:
                    if score_col in scores_df.columns:
                        hover_data[score_col] = ':.4f'
                
                # The color column should be formatted if it's numeric
                if pd.api.types.is_numeric_dtype(scores_df[color_col]):
                    hover_data[color_col] = ':.4f'
                
                # Determine if this is continuous or categorical
                is_continuous = pd.api.types.is_numeric_dtype(scores_df[color_col])
                if is_continuous and 'moa' not in color_col and 'library' not in color_col:
                    # For continuous coloring
                    fig = px.scatter(
                        scores_df,
                        x='cosine_distance_from_dmso',
                        y=y_metric,
                        color=color_col,
                        color_continuous_scale='viridis',
                        title=f'{ref_or_test.title()}: DMSO Distance vs {display_name}<br>Colored by {color_label}',
                        hover_data=hover_data,
                        labels={
                            'cosine_distance_from_dmso': 'DMSO Distance',
                            y_metric: display_name,
                            color_col: color_label
                        }
                    )
                else:
                    # For categorical coloring
                    fig = px.scatter(
                        scores_df,
                        x='cosine_distance_from_dmso',
                        y=y_metric,
                        color=color_col,
                        title=f'{ref_or_test.title()}: DMSO Distance vs {display_name}<br>Colored by {color_label}',
                        hover_data=hover_data,
                        labels={
                            'cosine_distance_from_dmso': 'DMSO Distance',
                            y_metric: display_name,
                            color_col: color_label
                        }
                    )
                
                # Add DMSO threshold line if available
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
                
                # Improve layout
                fig.update_layout(
                    xaxis_title='Cosine Distance from DMSO',
                    yaxis_title=display_name,
                    height=700,
                    width=1000
                )
                
                # Save interactive plot
                color_short = color_col.split('_')[-1] if '_' in color_col else color_col
                interactive_filename = f'dmso_vs_{metric_short_name}_by_{color_short}_{ref_or_test}_interactive.html'
                interactive_path = metric_interactive_dir / interactive_filename
                fig.write_html(interactive_path)
                log_info(f"Saved interactive plot to: {interactive_path}")
                
                # Also save to legacy interactive directory
                legacy_interactive_path = interactive_dir / interactive_filename
                fig.write_html(legacy_interactive_path)
                
            except Exception as e:
                log_info(f"Error creating interactive plot for {display_name} colored by {color_label}: {str(e)}")
        
        # Create combined interactive plot with dropdown
        try:
            # Check which color options are valid (columns exist in dataframe)
            valid_color_options = [(col, label) for col, label in color_options if col in scores_df.columns]
            
            if valid_color_options:
                log_info(f"Creating combined interactive plot for {metric_short_name} with {len(valid_color_options)} color options")
                fig = go.Figure()
                
                # Prepare hover data for each data point
                hover_template = (
                    "<b>Treatment:</b> %{customdata[0]}<br>" +
                    "<b>Library:</b> %{customdata[1]}<br>" +
                    "<b>Closest landmark:</b> %{customdata[2]}<br>" +
                    "<b>Second closest:</b> %{customdata[3]}<br>" +
                    "<b>Third closest:</b> %{customdata[4]}<br>" +
                    f"<b>{metric_display_names.get('mad_cosine')}:</b> %{{customdata[5]:.4f}}<br>" +
                    f"<b>{metric_display_names.get('var_cosine')}:</b> %{{customdata[6]:.4f}}<br>" +
                    f"<b>{metric_display_names.get('std_cosine')}:</b> %{{customdata[7]:.4f}}<br>" +
                    "<b>DMSO Distance:</b> %{x:.4f}<br>" +
                    "<b>Landmark Distance:</b> %{customdata[8]:.4f}<br>"
                )
                
                # Add dynamic harmonic mean scores to hover
                if 'harmonic_mean_3term_mad_cosine' in scores_df.columns:
                    hover_template += "<b>Harmonic Mean Score (3-term, MAD):</b> %{customdata[9]:.4f}<br>"
                elif 'harmonic_mean_3term' in scores_df.columns:
                    hover_template += "<b>Harmonic Mean Score (3-term):</b> %{customdata[9]:.4f}<br>"
                    
                if 'harmonic_mean_2term_mad_cosine' in scores_df.columns:
                    hover_template += "<b>Harmonic Mean Score (2-term, MAD):</b> %{customdata[10]:.4f}<br>"
                elif 'harmonic_mean_2term' in scores_df.columns:
                    hover_template += "<b>Harmonic Mean Score (2-term):</b> %{customdata[10]:.4f}<br>"
                
                # End the hover template
                hover_template += "<extra></extra>"
                
                # Prepare customdata array
                custom_data_cols = [
                    'treatment', 
                    'library' if 'library' in scores_df.columns else None,
                    'closest_landmark' if 'closest_landmark' in scores_df.columns else None,
                    'second_closest_landmark' if 'second_closest_landmark' in scores_df.columns else None,
                    'third_closest_landmark' if 'third_closest_landmark' in scores_df.columns else None,
                    'mad_cosine' if 'mad_cosine' in scores_df.columns else None,
                    'var_cosine' if 'var_cosine' in scores_df.columns else None,
                    'std_cosine' if 'std_cosine' in scores_df.columns else None,
                    'closest_landmark_distance' if 'closest_landmark_distance' in scores_df.columns else None
                ]
                
                # Add harmonic mean scores
                if 'harmonic_mean_3term_mad_cosine' in scores_df.columns:
                    custom_data_cols.append('harmonic_mean_3term_mad_cosine')
                elif 'harmonic_mean_3term' in scores_df.columns:
                    custom_data_cols.append('harmonic_mean_3term')
                else:
                    custom_data_cols.append(None)
                    
                if 'harmonic_mean_2term_mad_cosine' in scores_df.columns:
                    custom_data_cols.append('harmonic_mean_2term_mad_cosine')
                elif 'harmonic_mean_2term' in scores_df.columns:
                    custom_data_cols.append('harmonic_mean_2term')
                else:
                    custom_data_cols.append(None)
                
                # Filter out None values
                custom_data_cols = [col for col in custom_data_cols if col is not None]
                
                # Create custom data array
                if custom_data_cols:
                    custom_data = np.array([scores_df[col].values if col in scores_df.columns 
                                         else np.array(['N/A'] * len(scores_df)) for col in custom_data_cols]).T
                else:
                    custom_data = None
                
                # Add a trace for each color option
                for i, (color_col, color_label) in enumerate(valid_color_options):
                    is_continuous = pd.api.types.is_numeric_dtype(scores_df[color_col])
                    
                    # For continuous variables (like harmonic means)
                    if is_continuous and 'moa' not in color_col and 'library' not in color_col:
                        fig.add_trace(
                            go.Scatter(
                                x=scores_df['cosine_distance_from_dmso'],
                                y=scores_df[y_metric],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=scores_df[color_col],
                                    colorscale='Viridis',
                                    colorbar=dict(title=color_label),
                                    showscale=True
                                ),
                                name=color_label,
                                customdata=custom_data,
                                hovertemplate=hover_template,
                                visible=(i == 0)  # Only first one visible initially
                            )
                        )
                    # For categorical variables (like library, moa)
                    else:
                        categories = scores_df[color_col].dropna().unique()
                        visible_status = i == 0  # First category option is visible initially
                        
                        for cat in categories:
                            cat_mask = scores_df[color_col] == cat
                            if not any(cat_mask):
                                continue
                                
                            fig.add_trace(
                                go.Scatter(
                                    x=scores_df.loc[cat_mask, 'cosine_distance_from_dmso'],
                                    y=scores_df.loc[cat_mask, y_metric],
                                    mode='markers',
                                    marker=dict(size=8),
                                    name=str(cat),
                                    legendgroup=color_col,
                                    customdata=custom_data[cat_mask] if custom_data is not None else None,
                                    hovertemplate=hover_template,
                                    visible=visible_status
                                )
                            )
                
                # Create buttons for dropdown
                buttons = []
                trace_index = 0
                
                for i, (color_col, color_label) in enumerate(valid_color_options):
                    is_continuous = pd.api.types.is_numeric_dtype(scores_df[color_col])
                    
                    if is_continuous and 'moa' not in color_col and 'library' not in color_col:
                        # For continuous variables - one trace
                        visibility = [j == i for j in range(len(valid_color_options))]
                        visibility_full = [False] * len(fig.data)
                        visibility_full[trace_index] = True
                        
                        buttons.append(
                            dict(
                                label=color_label,
                                method="update",
                                args=[
                                    {"visible": visibility_full},
                                    {"title": f"{ref_or_test.title()}: DMSO Distance vs {display_name}<br>Colored by {color_label}"}
                                ]
                            )
                        )
                        trace_index += 1
                    else:
                        # For categorical variables - multiple traces
                        categories = scores_df[color_col].dropna().unique()
                        n_cats = len(categories)
                        
                        # Create a visibility list for this categorical option
                        visibility_full = [False] * len(fig.data)
                        for j in range(n_cats):
                            if trace_index + j < len(visibility_full):
                                visibility_full[trace_index + j] = True
                        
                        buttons.append(
                            dict(
                                label=color_label,
                                method="update",
                                args=[
                                    {"visible": visibility_full},
                                    {"title": f"{ref_or_test.title()}: DMSO Distance vs {display_name}<br>Colored by {color_label}"}
                                ]
                            )
                        )
                        trace_index += n_cats
                
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
                    title=f"{ref_or_test.title()}: DMSO Distance vs {display_name}<br>Select coloring variable",
                    xaxis_title="Cosine Distance from DMSO",
                    yaxis_title=display_name,
                    height=700,
                    width=1000,
                    margin=dict(t=60, b=40, l=40, r=120)  # Add margins
                )
                
                # Add DMSO threshold line if available
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
                
                # Save combined interactive plot
                combined_filename = f'dmso_vs_{metric_short_name}_combined_{ref_or_test}_interactive.html'
                combined_path = combined_dir / combined_filename
                fig.write_html(combined_path)
                log_info(f"Saved combined interactive plot to: {combined_path}")
                
            else:
                log_info(f"No valid color options for combined plot of {display_name}")
                
        except Exception as e:
            log_info(f"Error creating combined interactive plot for {display_name}: {str(e)}")
            import traceback
            log_info(traceback.format_exc())