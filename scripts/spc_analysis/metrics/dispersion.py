# Metrics for distance

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
from ..utils.logging import log_info, log_section

def calculate_replicate_metrics(df, embedding_cols, is_reference, config=None, dir_paths=None):
    """
    Enhanced version that calculates multiple dispersion metrics while maintaining original functionality
    
    Args:
        df: DataFrame with embeddings
        embedding_cols: List of embedding column names
        is_reference: Boolean indicating if this is reference data
        config: Configuration dictionary (optional)
        dir_paths: Directory paths (optional)
        
    Returns:
        DataFrame: Metrics for each treatment including:
            - mad_cosine (original)
            - var_cosine (new)
            - std_cosine (new)
            - median_distance (original)
            - well_count (original)
            - sample_count (original)
            - is_reference (original)
            - All original metadata columns
    """
    log_section(f"COMPUTING DISPERSION METRICS FOR {'REFERENCE' if is_reference else 'TEST'} SET")
    log_info("METRICS FUNCTION VERSION: NEW WITH ALL METRICS")  # Add this line
    
    treatments = df['treatment'].unique()
    log_info(f"Found {len(treatments)} unique treatments")
    
    metrics_results = []
    
    for treatment in tqdm(treatments, desc="Computing metrics"):
        # Skip invalid treatments (keep original handling)
        if pd.isna(treatment) or treatment == '' or treatment == 'NaN':
            continue
            
        # Original plate-well handling
        treatment_df = df[df['treatment'] == treatment].copy()
        treatment_df['plate_well'] = treatment_df['plate'].astype(str) + '_' + treatment_df['well'].astype(str)
        plate_wells = treatment_df['plate_well'].unique()

        if len(plate_wells) <= 1:
            log_info(f"Skipping treatment {treatment} - only has {len(plate_wells)} plate-wells")
            continue
            
        # Original centroid calculation
        well_centroids = {}
        for plate_well in plate_wells:
            plate, well = plate_well.split('_', 1)
            well_data = treatment_df[(treatment_df['plate'].astype(str) == plate) & (treatment_df['well'] == well)]
            if len(well_data) > 0:
                well_centroids[plate_well] = well_data[embedding_cols].mean().values

        # Pairwise distance calculation (original)
        distances = []
        for i, plate_well1 in enumerate(plate_wells):
            if plate_well1 not in well_centroids:
                continue
            for plate_well2 in plate_wells[i+1:]:
                if plate_well2 in well_centroids:
                    distances.append(cosine(well_centroids[plate_well1], well_centroids[plate_well2]))

        if not distances:
            log_info(f"No valid distances for treatment {treatment}")
            continue

        # Original logging - removing as too long.
        #log_info(f"Treatment: {treatment}")
        #log_info(f"  Number of plate-wells: {len(plate_wells)}")
        #log_info(f"  Total samples: {len(treatment_df)}")
        #log_info(f"  Number of pairwise distances calculated: {len(distances)}")
        
        # Calculate all metrics
        median_dist = np.median(distances)
        abs_deviations = np.abs(distances - median_dist)
        
        # Original metrics + new ones
        result = {
            'treatment': treatment,
            'mad_cosine': np.median(abs_deviations),  # Original
            'var_cosine': np.var(distances),          # New
            'std_cosine': np.std(distances),          # New
            'median_distance': median_dist,           # Original
            'well_count': len(plate_wells),           # Original
            'sample_count': len(treatment_df),        # Original
            'is_reference': is_reference              # Original
        }
        
        # Preserve all metadata columns (expanded list)
        first_row = treatment_df.iloc[0]
        metadata_cols = ['moa', 'compound_name', 'compound_uM', 'cell_type', 'library',
                        # Add new PP metadata columns
                        'lib_plate_order', 'perturbation_name', 'chemical_name', 
                        'supplier_ID', 'control_type', 'control_name', 'is_control',
                        'annotated_target', 'annotated_target_description', 
                        'PP_ID', 'SMILES', 'chemical_description',
                        'compound_type',
                        'manual_annotation']

        for col in metadata_cols:
            if col in first_row:
                result[col] = first_row[col]
        
        metrics_results.append(result)
    
    # Original empty dataframe handling
    if len(metrics_results) == 0:
        log_info("Warning: No results calculated. Check if treatments exist in the dataset.")
        return pd.DataFrame(columns=['treatment', 'mad_cosine', 'var_cosine', 'std_cosine',
                                   'median_distance', 'well_count', 'sample_count', 'is_reference'])
    
    # Convert to DataFrame and sort by MAD (original behavior)
    metrics_df = pd.DataFrame(metrics_results).sort_values('mad_cosine', ascending=True)

    # Add this new logging section right here, before saving the metrics
    log_info("=== METRICS CALCULATION COMPLETE ===")
    log_info(f"Generated metrics dataframe with {len(metrics_df)} rows and columns: {metrics_df.columns.tolist()}")
    log_info(f"Metrics summary:")
    for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
        if metric in metrics_df.columns:
            log_info(f"  {metric}: min={metrics_df[metric].min():.6f}, max={metrics_df[metric].max():.6f}, mean={metrics_df[metric].mean():.6f}")
        else:
            log_info(f"  {metric}: NOT FOUND IN METRICS DATAFRAME")
    
    # Original saving behavior
    if config and 'output_dir' in config and dir_paths:
        output_name = 'reference_metrics.csv' if is_reference else 'test_metrics.csv'
        output_path = dir_paths['analysis']['mad'] / output_name
        metrics_df.to_csv(output_path, index=False)
        log_info(f"Saved metrics to: {output_path}")
    
    return metrics_df