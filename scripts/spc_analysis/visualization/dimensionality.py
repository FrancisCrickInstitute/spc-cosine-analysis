# Dimensionality reduction: UMAP & tSNE

import os
import gc
import traceback
import numpy as np
import pandas as pd
import umap
import sklearn.manifold
import plotly.express as px
from ..utils.logging import log_info, log_section, get_memory_usage_str
from .embedding import generate_interactive_embedding_plot, create_combined_interactive_plot

def generate_dimensionality_reduction_plots(merged_df, landmarks=None, embedding_cols=None, config=None, dir_paths=None, viz_data=None):
    """
    Generate UMAP and t-SNE plots with coloring by multiple parameters.
    Also adds coordinates to the visualization dataframe.
    
    Args:
        merged_df: DataFrame with embeddings and metadata
        landmarks: DataFrame with landmark information (optional)
        embedding_cols: List of embedding column names
        config: Configuration dictionary
        dir_paths: Dictionary with directory paths
        viz_data: Optional pre-prepared visualization data with all metrics
    """
    log_section("GENERATING DIMENSIONALITY REDUCTION PLOTS")
    # Validate inputs
    if merged_df is None or merged_df.empty:
        log_info("Error: No data provided for dimensionality reduction")
        return merged_df
    
    if embedding_cols is None or len(embedding_cols) == 0:
        # Try to detect embedding columns if not provided
        embedding_cols = [col for col in merged_df.columns if col.startswith('Z') and col[1:].isdigit()]
        if len(embedding_cols) == 0:
            log_info("Error: No embedding columns found for dimensionality reduction")
            return merged_df
    
    log_info(f"Using {len(embedding_cols)} embedding dimensions for dimensionality reduction")
    
    # Use viz_data if provided, otherwise use merged_df
    sample_data = viz_data if viz_data is not None else merged_df
    log_info(f"Using {'pre-prepared visualization data' if viz_data is not None else 'merged dataframe'} for plotting")
    
    # CRITICAL FIX: Use the same dataframe for embeddings and coordinate assignment
    # If viz_data is provided, use it for both embeddings and assignment
    if viz_data is not None:
        # Check if viz_data has embedding columns
        if all(col in viz_data.columns for col in embedding_cols):
            embeddings_df = viz_data
            log_info("Using viz_data for both embeddings and coordinate assignment")
        else:
            log_info("WARNING: viz_data missing embedding columns, using merged_df for embeddings")
            embeddings_df = merged_df
            # We'll need to align the results later
    else:
        embeddings_df = merged_df
    
    # Log the shapes for debugging
    log_info(f"Embeddings source shape: {embeddings_df.shape}")
    log_info(f"Sample data shape: {sample_data.shape}")
    
    # Extract embeddings for dimensionality reduction
    embeddings = embeddings_df[embedding_cols].values
    log_info(f"Extracted embeddings with shape: {embeddings.shape}")
    log_info(f"Memory usage after embedding extraction: {get_memory_usage_str()}")

    # Define all color columns to generate plots for
    # Format: (column_name, is_continuous, display_name, color_palette)
    # Define all color columns to generate plots for (UPDATED - no duplicates)
    color_columns = [
        # Original columns
        ('library', False, 'Library', px.colors.qualitative.Bold),
        ('landmark_label', False, 'Landmark Status (Legacy)', px.colors.qualitative.Set1),
        ('manual_annotation', False, 'Manual Annotation', px.colors.qualitative.Set2), 
        ('moa', False, 'MOA', px.colors.qualitative.Dark24),
        ('moa_first', False, 'MOA (First)', px.colors.qualitative.Dark24),
        ('moa_truncated_10', False, 'MOA (Truncated)', px.colors.qualitative.Dark24),
        ('moa_compound_uM', False, 'MOA @ Concentration', px.colors.qualitative.Dark24),
        ('treatment', False, 'Treatment', px.colors.qualitative.Light24),
        
        # Landmark label columns
        ('landmark_label_mad', False, 'Landmark Status (MAD)', px.colors.qualitative.Set1),
        ('landmark_label_std', False, 'Landmark Status (StdDev)', px.colors.qualitative.Set1),
        ('landmark_label_var', False, 'Landmark Status (Variance)', px.colors.qualitative.Set1),
        
        # Target description (PRIMARY - no duplicates)
        ('annotated_target_description', False, 'Target Description', px.colors.qualitative.Dark24),
        ('annotated_target_description_truncated_10', False, 'Target Description (Truncated)', px.colors.qualitative.Dark24),
        
        # Other discrete columns
        ('plate', False, 'Plate', px.colors.qualitative.Bold),
        ('well_row', False, 'Well Row', px.colors.qualitative.Bold),
        ('well_column', False, 'Well Column', px.colors.qualitative.Bold),
        ('compound_uM', False, 'Compound Concentration (µM)', px.colors.qualitative.Vivid),
        
        # Unified continuous columns (including new metrics)
        ('cosine_distance_from_dmso', True, 'Cosine Distance from DMSO', 'Plasma'),
        ('mad_cosine', True, 'MAD Cosine', 'Inferno'),
        ('var_cosine', True, 'Variance Cosine', 'Cividis'),
        ('std_cosine', True, 'Std Dev Cosine', 'Magma'),
        ('median_distance', True, 'Median Distance', 'Viridis'),
        ('well_count', False, 'Well Count', px.colors.qualitative.Prism),
    ]

    # Create hover data list (UPDATED - no duplicates, add landmark metadata)
    hover_data_cols = [
        'treatment', 'compound_name', 'compound_uM', 
        'moa', 'moa_first', 'moa_truncated_10', 'moa_compound_uM',
        'annotated_target_description', 'annotated_target_description_truncated_10',  # Primary target columns
        'plate', 'well', 'well_row', 'well_column', 'library', 
        'cell_count', 'cell_pct',
        'manual_annotation',
        
        # Unified metrics
        'cosine_distance_from_dmso', 'mad_cosine', 'var_cosine', 'std_cosine',
        'median_distance', 'well_count',
        
        # Reference/test metrics
        'reference_cosine_distance_from_dmso', 'reference_mad_cosine', 
        'reference_var_cosine', 'reference_std_cosine',
        'reference_median_distance', 'reference_well_count',
        'test_cosine_distance_from_dmso', 'test_mad_cosine',
        'test_var_cosine', 'test_std_cosine',
        'test_median_distance', 'test_well_count',
        
        # Landmark labels
        'landmark_label', 'landmark_label_mad', 'landmark_label_std', 'landmark_label_var',
        
        # Landmark distances and metadata (basic)
        'closest_landmark', 'closest_landmark_distance', 'closest_landmark_moa',
        'second_closest_landmark', 'second_closest_landmark_distance', 'second_closest_landmark_moa',
        'third_closest_landmark', 'third_closest_landmark_distance', 'third_closest_landmark_moa',
        
        # PP and gene data
        'PP_ID', 'PP_ID_uM', 'gene_description'
    ]

    # Filter color columns to only those that exist in the data
    available_color_columns = []
    for col_info in color_columns:
        col_name = col_info[0]
        if col_name in sample_data.columns:
            # Check if column has data
            non_nan_count = sample_data[col_name].notna().sum()
            if non_nan_count > 0:
                available_color_columns.append(col_info)
            else:
                log_info(f"Skipping color column '{col_name}' - no non-NaN values")
        else:
            log_info(f"Skipping color column '{col_name}' - not found in data")
    
    # Force garbage collection before UMAP
    gc.collect()
    log_info(f"Forced garbage collection before UMAP")
    log_info(f"Memory usage before UMAP: {get_memory_usage_str()}")

    # Generate UMAP projection...
    log_info("="*80)
    log_info("STARTING UMAP PROJECTION")
    log_info("="*80)
    log_info(f"Dataset: {embeddings.shape[0]} samples × {embeddings.shape[1]} dimensions")
    log_info(f"Parameters: n_neighbors=30, min_dist=0.1, metric=cosine")
    log_info("This may take 5-15 minutes for large datasets...")
    log_info("Progress: Initializing UMAP...")

    try:
        # ... UMAP setup code stays the same ...
        umap_params = {
            'n_neighbors': 30,
            'min_dist': 0.1,
            'n_components': 2,
            'metric': 'cosine',
            'random_state': 42
        }
        log_info(f"UMAP parameters: {umap_params}")
        
        umap_reducer = umap.UMAP(**umap_params)
        log_info("Progress: Running UMAP fit_transform (this is the slow step)...")
        umap_result = umap_reducer.fit_transform(embeddings)
        
        log_info("="*80)
        log_info("UMAP PROJECTION COMPLETED SUCCESSFULLY")
        log_info("="*80)
        log_info(f"Result shape: {umap_result.shape}")
        log_info(f"Memory usage after UMAP: {get_memory_usage_str()}")
        
        # CRITICAL FIX: Handle coordinate assignment correctly
        if embeddings_df is sample_data:
            # Same dataframe, direct assignment
            log_info("Adding UMAP coordinates to sample_data (same as embeddings source)")
            sample_data['UMAP1'] = umap_result[:, 0]
            sample_data['UMAP2'] = umap_result[:, 1]
        else:
            # Different dataframes, need to align
            log_info("Aligning UMAP coordinates between different dataframes")
            # Create a temporary dataframe with coordinates
            coords_df = embeddings_df[['treatment', 'plate', 'well']].copy()
            coords_df['UMAP1'] = umap_result[:, 0]
            coords_df['UMAP2'] = umap_result[:, 1]
            
            # Merge coordinates into sample_data
            merge_keys = ['treatment', 'plate', 'well']
            sample_data = sample_data.merge(coords_df[merge_keys + ['UMAP1', 'UMAP2']], 
                                          on=merge_keys, how='left')
            log_info(f"Merged UMAP coordinates. Sample data shape now: {sample_data.shape}")
        
        log_info(f"UMAP projection complete with shape: {umap_result.shape}")
        
        # UMAP Plots - Individual
        log_info("Generating individual UMAP plots")
        
        for col_name, is_continuous, display_name, color_palette in available_color_columns:
            log_info(f"Creating UMAP plot colored by {display_name}")
            try:
                # Create sanitized filename
                safe_name = col_name.replace('_', '')
                filename = f'umap_by_{safe_name}.html'
                
                generate_interactive_embedding_plot(
                    sample_data, 
                    x='UMAP1', 
                    y='UMAP2', 
                    color=col_name,
                    hover_data=hover_data_cols,
                    title=f'UMAP Projection Colored by {display_name}',
                    filename=filename,
                    dir_path=dir_paths['visualizations']['dimensionality_reduction']['umap'],
                    is_continuous=is_continuous,
                    color_palette=color_palette
                )
                log_info(f"Created UMAP plot colored by {display_name}")
            except Exception as e:
                log_info(f"Error creating UMAP plot for {display_name}: {str(e)}")
                log_info(traceback.format_exc())
        
        # UMAP Combined Interactive Plot with Dropdown
        if available_color_columns:
            log_info("Creating combined interactive UMAP plot with dropdown selector")
            try:
                create_combined_interactive_plot(
                    sample_data,
                    'UMAP1', 'UMAP2',
                    available_color_columns,
                    hover_data_cols,
                    'UMAP Projection',
                    'umap_combined_interactive.html',
                    dir_paths['visualizations']['dimensionality_reduction']['umap']
                )
                log_info("Created combined interactive UMAP plot")
            except Exception as e:
                log_info(f"Error creating combined UMAP plot: {str(e)}")
                log_info(f"Error traceback: {traceback.format_exc()}")
        else:
            log_info("Skipping combined UMAP plot - no valid color columns")
        
    except Exception as e:
        log_info(f"Error generating UMAP projection: {str(e)}")
        log_info(f"Error traceback: {traceback.format_exc()}")
        log_info(f"Memory at error: {get_memory_usage_str()}")
    
    # Force garbage collection before t-SNE
    gc.collect()
    log_info(f"Forced garbage collection before t-SNE")
    log_info(f"Memory usage before t-SNE: {get_memory_usage_str()}")
    
    # Generate t-SNE projection
    log_info("="*80)
    log_info("STARTING t-SNE PROJECTION")
    log_info("="*80)
    log_info(f"Dataset: {embeddings.shape[0]} samples × {embeddings.shape[1]} dimensions")
    log_info(f"Parameters: perplexity=30, metric=cosine")
    log_info("This may take 10-20 minutes for large datasets...")
    log_info("Progress: Initializing t-SNE...")
    
    # Generate t-SNE projection
    log_info("Generating t-SNE projection...")
    try:
        # ... t-SNE setup code stays the same ...
        tsne_params = {
            'n_components': 2,
            'perplexity': 30,
            'metric': 'cosine',
            'random_state': 42
        }
        log_info(f"t-SNE parameters: {tsne_params}")
        
        tsne = sklearn.manifold.TSNE(**tsne_params)
        log_info("Progress: Running t-SNE fit_transform (this is the slow step)...")
        tsne_result = tsne.fit_transform(embeddings)
        
        log_info("="*80)
        log_info("t-SNE PROJECTION COMPLETED SUCCESSFULLY")
        log_info("="*80)
        log_info(f"Result shape: {tsne_result.shape}")
        log_info(f"Memory usage after t-SNE: {get_memory_usage_str()}")
        
        # CRITICAL FIX: Handle coordinate assignment correctly
        if embeddings_df is sample_data:
            # Same dataframe, direct assignment
            log_info("Adding t-SNE coordinates to sample_data (same as embeddings source)")
            sample_data['TSNE1'] = tsne_result[:, 0]
            sample_data['TSNE2'] = tsne_result[:, 1]
        else:
            # Different dataframes, need to align
            log_info("Aligning t-SNE coordinates between different dataframes")
            # Create a temporary dataframe with coordinates
            coords_df = embeddings_df[['treatment', 'plate', 'well']].copy()
            coords_df['TSNE1'] = tsne_result[:, 0]
            coords_df['TSNE2'] = tsne_result[:, 1]
            
            # Merge coordinates into sample_data
            merge_keys = ['treatment', 'plate', 'well']
            # Check if TSNE columns already exist and drop them
            if 'TSNE1' in sample_data.columns:
                sample_data = sample_data.drop(columns=['TSNE1', 'TSNE2'])
            
            sample_data = sample_data.merge(coords_df[merge_keys + ['TSNE1', 'TSNE2']], 
                                          on=merge_keys, how='left')
            log_info(f"Merged t-SNE coordinates. Sample data shape now: {sample_data.shape}")
        
        log_info(f"t-SNE projection complete with shape: {tsne_result.shape}")
        
        # t-SNE Plots - Individual
        log_info("Generating individual t-SNE plots")
        for col_name, is_continuous, display_name, color_palette in available_color_columns:
            log_info(f"Creating t-SNE plot colored by {display_name}")
            try:
                # Create sanitized filename
                safe_name = col_name.replace('_', '')
                filename = f'tsne_by_{safe_name}.html'
                
                generate_interactive_embedding_plot(
                    sample_data, 
                    x='TSNE1', 
                    y='TSNE2', 
                    color=col_name,
                    hover_data=hover_data_cols,
                    title=f't-SNE Projection Colored by {display_name}',
                    filename=filename,
                    dir_path=dir_paths['visualizations']['dimensionality_reduction']['tsne'],
                    is_continuous=is_continuous,
                    color_palette=color_palette
                )
                log_info(f"Created t-SNE plot colored by {display_name}")
            except Exception as e:
                log_info(f"Error creating t-SNE plot for {display_name}: {str(e)}")
                log_info(traceback.format_exc())
            
        # t-SNE Combined Interactive Plot with Dropdown
        if available_color_columns:
            log_info("Creating combined interactive t-SNE plot with dropdown selector")
            try:
                create_combined_interactive_plot(
                    sample_data,
                    'TSNE1', 'TSNE2',
                    available_color_columns,
                    hover_data_cols,
                    't-SNE Projection',
                    'tsne_combined_interactive.html',
                    dir_paths['visualizations']['dimensionality_reduction']['tsne']
                )
                log_info("Created combined interactive t-SNE plot")
            except Exception as e:
                log_info(f"Error creating combined t-SNE plot: {str(e)}")
                log_info(f"Error traceback: {traceback.format_exc()}")
        else:
            log_info("Skipping combined t-SNE plot - no valid color columns")
        
    except Exception as e:
        log_info(f"Error generating t-SNE projection: {str(e)}")
        log_info(f"Error traceback: {traceback.format_exc()}")
        log_info(f"Memory at error: {get_memory_usage_str()}")
    
    # Save coordinates to a dedicated file
    if ('UMAP1' in sample_data.columns and 'UMAP2' in sample_data.columns) or ('TSNE1' in sample_data.columns and 'TSNE2' in sample_data.columns):
        log_info("Saving dimensionality reduction coordinates to dedicated file...")
        
        # Create a coordinates dataframe with necessary columns for identification
        coord_columns = ['treatment', 'plate', 'well']
        if 'compound_name' in sample_data.columns:
            coord_columns.append('compound_name')
        if 'compound_uM' in sample_data.columns:
            coord_columns.append('compound_uM')
        
        # Add UMAP coordinates if available
        if 'UMAP1' in sample_data.columns and 'UMAP2' in sample_data.columns:
            coord_columns.extend(['UMAP1', 'UMAP2'])
        
        # Add t-SNE coordinates if available
        if 'TSNE1' in sample_data.columns and 'TSNE2' in sample_data.columns:
            coord_columns.extend(['TSNE1', 'TSNE2'])
        
        # Filter to columns that actually exist
        coord_columns = [col for col in coord_columns if col in sample_data.columns]
        
        # Create the coordinates dataframe
        coords_df = sample_data[coord_columns].copy()
        
        # Save to CSV
        output_path = dir_paths['data'] / 'umap_embeddings.csv'
        coords_df.to_csv(output_path, index=False)
        log_info(f"Saved dimensionality reduction coordinates to: {output_path}")
        
        # ALWAYS update the visualization_data.csv file, regardless of whether viz_data was provided
        viz_data_path = dir_paths['data'] / 'visualization_data.csv'
        if viz_data_path.exists():
            log_info("Updating existing visualization data with dimensionality reduction coordinates...")
            try:
                # Read existing viz data
                viz_df = pd.read_csv(viz_data_path)
                
                # Determine merge keys
                merge_keys = ['treatment']
                if 'plate' in coords_df.columns and 'plate' in viz_df.columns:
                    merge_keys.append('plate')
                if 'well' in coords_df.columns and 'well' in viz_df.columns:
                    merge_keys.append('well')
                
                # Determine which coordinate columns to merge
                dim_red_cols = []
                if 'UMAP1' in coords_df.columns and 'UMAP2' in coords_df.columns:
                    dim_red_cols.extend(['UMAP1', 'UMAP2'])
                if 'TSNE1' in coords_df.columns and 'TSNE2' in coords_df.columns:
                    dim_red_cols.extend(['TSNE1', 'TSNE2'])
                
                # Only proceed if we have coordinate columns
                if dim_red_cols:
                    # Remove existing UMAP/TSNE columns from viz_df if they exist
                    for col in dim_red_cols:
                        if col in viz_df.columns:
                            viz_df = viz_df.drop(columns=[col])
                    
                    # Merge the coordinates into the visualization data
                    log_info(f"Merging coordinates using keys: {merge_keys}")
                    viz_df = viz_df.merge(coords_df[merge_keys + dim_red_cols], on=merge_keys, how='left')
                    
                    # Save updated visualization data
                    viz_df.to_csv(viz_data_path, index=False)
                    log_info(f"Successfully updated visualization data with coordinates")
                else:
                    log_info("No coordinate columns found to merge")
            except Exception as e:
                log_info(f"Error updating visualization data: {str(e)}")
                log_info(traceback.format_exc())
    
    log_info("Dimensionality reduction completed")
    log_info(f"Final memory usage: {get_memory_usage_str()}")
    
    # Return the updated sample_data
    return sample_data