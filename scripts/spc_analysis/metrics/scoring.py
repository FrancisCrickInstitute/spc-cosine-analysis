# Updated scoring.py with expanded landmark metadata preservation

import numpy as np
import pandas as pd
from ..utils.logging import log_section, log_info


def calculate_scores(dmso_distances, landmark_distances, mad_df=None, config=None, dir_paths=None, is_reference=True):
    """
    Calculate scores for compounds based on different methods.
    
    UPDATED: Now preserves expanded landmark metadata including truncated versions
    
    Args:
        dmso_distances: DataFrame with distances from DMSO
        landmark_distances: DataFrame with distances to landmarks
        mad_df: DataFrame with MAD values (optional)
        config: Configuration dictionary
        is_reference: Boolean indicating if this is reference or test data
        
    Returns:
        DataFrame: Scores for each compound
    """
    dataset_label = "REFERENCE" if is_reference else "TEST"
    log_section(f"CALCULATING COMPOUND SCORES FOR {dataset_label} SET")

    # Print detailed diagnostic information
    log_info("DIAGNOSTIC: Input DataFrame Details")
    log_info(f"DMSO Distances columns: {dmso_distances.columns.tolist()}")
    log_info(f"Landmark Distances columns: {landmark_distances.columns.tolist()}")
    if mad_df is not None:
        log_info(f"MAD DataFrame columns: {mad_df.columns.tolist()}")

    # Skip if either dataframe is None
    if dmso_distances is None or landmark_distances is None:
        log_info("Cannot calculate scores - missing required data")
        return None
    
    # Merge DMSO distances and landmark distances
    log_info("Merging distance data...")
    
    # Ensure 'treatment' column is present in both dataframes
    if 'treatment' not in dmso_distances.columns or 'treatment' not in landmark_distances.columns:
        log_info("Cannot merge dataframes - 'treatment' column missing")
        return None
    
    # Use suffixes to handle duplicate columns
    merged = pd.merge(dmso_distances, landmark_distances, on='treatment', how='inner', suffixes=('', '_landmark_dup'))
    log_info(f"Merged data has {len(merged)} rows")
    
    # IMPROVED: Handle duplicate columns while preserving landmark-specific metadata
    duplicate_cols = [col for col in merged.columns if col.endswith('_landmark_dup')]
    log_info(f"Found {len(duplicate_cols)} duplicate columns to process")
    
    # First, check what landmark metadata we have BEFORE dropping
    landmark_meta_before = [col for col in merged.columns if 'landmark_' in col]
    truncated_before = [col for col in landmark_meta_before if 'truncated' in col]
    log_info(f"Before deduplication: {len(landmark_meta_before)} landmark columns, {len(truncated_before)} truncated")
    
    for dup_col in duplicate_cols:
        orig_col = dup_col.replace('_landmark_dup', '')
        
        # ONLY drop if it's a general metadata column (not landmark-specific)
        if orig_col in ['library', 'moa', 'compound_name', 'compound_uM', 'cell_type', 'plate', 'well',
                        'lib_plate_order', 'perturbation_name', 'chemical_name', 'supplier_ID', 
                        'control_type', 'control_name', 'is_control', 'annotated_target', 
                        'annotated_target_description', 'PP_ID', 'SMILES', 'chemical_description',
                        'sample_count', 'is_reference', 'manual_annotation']:
            # Drop the duplicate for general metadata columns
            merged = merged.drop(columns=[dup_col])
            log_info(f"Dropped duplicate metadata: {dup_col}")
        else:
            # KEEP both - this is landmark-specific data (like truncated descriptions)
            log_info(f"PRESERVED landmark-specific: {orig_col} (keeping both versions)")
    
    # After deduplication, verify landmark metadata is preserved
    landmark_meta_after = [col for col in merged.columns if 'landmark_' in col]
    truncated_after = [col for col in landmark_meta_after if 'truncated' in col]
    log_info(f"After deduplication: {len(landmark_meta_after)} landmark columns, {len(truncated_after)} truncated")
    
    # Log sample of preserved truncated columns
    if truncated_after:
        log_info(f"Sample preserved truncated landmark columns:")
        for col in truncated_after[:5]:
            non_null = merged[col].notna().sum()
            log_info(f"  {col}: {non_null} non-null values")
    
    log_info(f"Final merged dataframe has {len(merged.columns)} columns")
    
    # CHANGE 3: Check for landmark columns with truncated versions
    landmark_cols = [col for col in merged.columns if 'landmark_' in col]
    truncated_landmark_cols = [col for col in landmark_cols if 'truncated' in col]
    log_info(f"Found {len(landmark_cols)} landmark columns, {len(truncated_landmark_cols)} are truncated versions")
    
    # Log sample of landmark columns found
    if truncated_landmark_cols:
        log_info("Sample truncated landmark columns:")
        for col in truncated_landmark_cols[:5]:
            log_info(f"  {col}")
    
    # Ensure we have a library column
    if 'library' not in merged.columns:
        if 'library_dmso' in merged.columns:
            merged['library'] = merged['library_dmso']
            merged = merged.drop(columns=['library_dmso'])
        elif 'library_landmark' in merged.columns:
            merged['library'] = merged['library_landmark']
            merged = merged.drop(columns=['library_landmark'])
    
    # Add metrics from mad_df if available
    if mad_df is not None and 'treatment' in mad_df.columns:
        log_info("Adding dispersion metrics to scores...")
        
        # Get all available metrics
        metric_columns = []
        for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
            if metric in mad_df.columns:
                metric_columns.append(metric)
        
        if metric_columns:
            # Also preserve metadata columns from mad_df if they're not already in merged
            mad_cols_to_merge = ['treatment'] + metric_columns
            
            # Add any metadata columns from mad_df that aren't already in merged
            for col in mad_df.columns:
                if col not in merged.columns and col not in mad_cols_to_merge:
                    if col in ['median_distance', 'well_count']:
                        mad_cols_to_merge.append(col)
            
            merged = pd.merge(merged, mad_df[mad_cols_to_merge], on='treatment', how='left')
            # log_info(f"Added metrics: {metric_columns}") # Remove excessive logging
    
    # Calculate ratio-based score
    # Normalize input scores for harmonic mean
    log_info("Normalizing inputs for harmonic mean scoring...")

    # DIAGNOSTIC: Print out the exact columns before normalization
    log_info("DIAGNOSTIC: Columns before normalization")
    log_info(f"Columns in merged DataFrame: {merged.columns.tolist()}")

    # Check if required columns exist
    required_columns = ['cosine_distance_from_dmso', 'closest_landmark_distance']
    for col in required_columns:
        if col not in merged.columns:
            log_info(f"WARNING: Required column '{col}' is missing!")

    # Normalize distance from DMSO (higher is better)
    if 'cosine_distance_from_dmso' in merged.columns:
        max_dmso_dist = merged['cosine_distance_from_dmso'].max()
        if max_dmso_dist > 0:
            merged['normalized_dmso_distance'] = merged['cosine_distance_from_dmso'] / max_dmso_dist
        else:
            merged['normalized_dmso_distance'] = 0
    
    # Normalize landmark similarity (1 - distance, higher is better)
    if 'closest_landmark_distance' in merged.columns:
        merged['normalized_landmark_similarity'] = 1 - merged['closest_landmark_distance']
        # Clip to ensure non-negative values
        merged['normalized_landmark_similarity'] = merged['normalized_landmark_similarity'].clip(0, 1)
    
    # Normalize each metric (invert so lower is better, i.e., 1 - normalized metric)
    for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
        if metric in merged.columns:
            max_val = merged[metric].max()
            if max_val > 0:
                merged[f'normalized_{metric}'] = 1 - (merged[metric] / max_val)
            else:
                merged[f'normalized_{metric}'] = 1
            log_info(f"Created normalized column for {metric}")
    
    # Calculate 2-term harmonic means for each metric
    log_info("Calculating 2-term weighted harmonic means...")

    # DIAGNOSTIC CODE
    log_info("DIAGNOSTIC: Columns available for harmonic mean calculation:")
    log_info(f"Columns in DataFrame: {merged.columns.tolist()}")

    # Log specific column existence
    required_columns = [
        'cosine_distance_from_dmso', 
        'closest_landmark_distance', 
        'mad_cosine', 'var_cosine', 'std_cosine'
    ]
    for col in required_columns:
        log_info(f"Column '{col}' exists: {col in merged.columns}")
        if col in merged.columns:
            log_info(f"  Non-null count for {col}: {merged[col].count()}")
    
    # Calculate 2-term harmonic mean with each metric
    if 'normalized_dmso_distance' in merged.columns and 'normalized_landmark_similarity' in merged.columns:
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        
        # Original 2-term harmonic mean with no specific metric
        merged['harmonic_mean_2term'] = 2 / (
            (1 / (merged['normalized_dmso_distance'] + epsilon)) +
            (1 / (merged['normalized_landmark_similarity'] + epsilon))
        )
        
        # Calculate 2-term harmonic means with each metric if available
        for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
            if f'normalized_{metric}' in merged.columns:
                merged[f'harmonic_mean_2term_{metric}'] = 2 / (
                    (1 / (merged['normalized_dmso_distance'] + epsilon)) +
                    (1 / (merged[f'normalized_{metric}'] + epsilon))
                )
                log_info(f"Calculated 2-term harmonic mean with {metric}")
    
    # Calculate 3-term harmonic means for each metric
    log_info("Calculating 3-term weighted harmonic means...")

    # Calculate harmonic mean only if necessary columns exist
    for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
        if all(col in merged.columns for col in ['normalized_dmso_distance', 'normalized_landmark_similarity', f'normalized_{metric}']):
            # Avoid division by zero by adding small epsilon
            epsilon = 1e-10
            
            merged[f'harmonic_mean_3term_{metric}'] = 3 / (
                (1 / (merged['normalized_dmso_distance'] + epsilon)) +
                (1 / (merged['normalized_landmark_similarity'] + epsilon)) +
                (1 / (merged[f'normalized_{metric}'] + epsilon))
            )
            log_info(f"Calculated 3-term harmonic mean with {metric}")
    
    # Original 3-term harmonic mean if all necessary columns exist
    if all(col in merged.columns for col in ['normalized_dmso_distance', 'normalized_landmark_similarity', 'normalized_mad_cosine']):
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        merged['harmonic_mean_3term'] = 3 / (
            (1 / (merged['normalized_dmso_distance'] + epsilon)) +
            (1 / (merged['normalized_landmark_similarity'] + epsilon)) +
            (1 / (merged['normalized_mad_cosine'] + epsilon))
        )
        log_info("Calculated original 3-term harmonic mean with mad_cosine")

    # Ensure numeric types for scoring columns
    score_cols = ['ratio_score', 'harmonic_mean_2term', 'harmonic_mean_3term']
    for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
        score_cols.extend([f'harmonic_mean_2term_{metric}', f'harmonic_mean_3term_{metric}'])
    
    for col in score_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
    
    # CHANGE 3: Log final columns summary with emphasis on landmark metadata
    log_info("FINAL COMPOUND SCORES COLUMNS:")
    
    # Enhanced logging for landmark columns
    landmark_metadata_cols = [col for col in merged.columns if 'landmark_' in col]
    truncated_cols = [col for col in landmark_metadata_cols if 'truncated' in col]
    
    log_info(f"Total landmark metadata columns: {len(landmark_metadata_cols)}")
    log_info(f"Truncated landmark columns: {len(truncated_cols)}")
    
    # Show sample of truncated columns
    if truncated_cols:
        log_info("Sample truncated landmark columns in final output:")
        for col in truncated_cols[:8]:  # Show first 8
            non_null = merged[col].notna().sum()
            log_info(f"  {col}: {non_null} non-null values")
    
    # Standard metadata columns
    metadata_cols = ['PP_ID', 'SMILES', 'chemical_description', 'annotated_target', 'supplier_ID']
    for col in metadata_cols:
        if col in merged.columns:
            non_null = merged[col].notna().sum()
            log_info(f"  {col}: {non_null} non-null values")

    # Save scores with updated filename
    if 'output_dir' in config:
        output_name = 'compound_reference_scores.csv' if is_reference else 'compound_test_scores.csv'
        output_path = dir_paths['analysis']['root'] / output_name
        merged.to_csv(output_path, index=False)
        log_info(f"Saved {dataset_label} compound scores to: {output_path}")
        
        # CHANGE 3: Log specific information about truncated columns in saved file
        truncated_in_output = [col for col in merged.columns if 'truncated' in col and 'landmark_' in col]
        log_info(f"Saved {len(truncated_in_output)} truncated landmark columns to scores file")
    
    return merged