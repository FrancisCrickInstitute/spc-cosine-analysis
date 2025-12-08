# Compoute DMSO distance function

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
from scipy.spatial.distance import cosine
from ..utils.logging import log_section, log_info


def compute_dmso_distance(df, embedding_cols, config, dir_paths, is_reference=True, dmso_centroid=None):
    """
    Compute distances from DMSO centroid for all treatments.
    
    Args:
        df: DataFrame with embeddings
        embedding_cols: List of embedding column names
        config: Configuration dictionary
        dir_paths: Dictionary containing directory paths
        is_reference: Boolean indicating if this is reference data
        
    Returns:
        tuple: (distance_df, thresholds)
    """
    label = "REFERENCE" if is_reference else "TEST"
    log_section(f"COMPUTING DISTANCE FROM DMSO FOR {label} SET")
    
    # Extract DMSO embeddings - MODIFIED TO USE treatment column: DMSO@'conc.term'
    log_info("Extracting DMSO embeddings...")
    # Handle both "DMSO" and "DMSO@X.X" formats
    dmso_df = df[df['treatment'].str.startswith('DMSO') if 'treatment' in df.columns else pd.Series(False)]

    if len(dmso_df) == 0:
        log_info("No DMSO samples found in 'treatment' column (including DMSO@X.X variants). Checking 'moa' column...")
        if 'moa' in df.columns and df['moa'].str.contains('DMSO').any():
            dmso_df = df[df['moa'] == 'DMSO']
            log_info(f"Found {len(dmso_df)} DMSO samples in 'moa' column.")
        else:
            log_info("No DMSO samples found. Cannot calculate distances.")
            return None, None
    else:
        log_info(f"Found {len(dmso_df)} DMSO samples (including DMSO@X.X variants) in 'treatment' column.")
    
    # Use provided centroid or calculate from this dataset
    if dmso_centroid is None:
        dmso_centroid = dmso_df[embedding_cols].median().values
        log_info("Calculated DMSO centroid (medianoid) from this dataset")
    else:
        log_info("Using pre-computed DMSO centroid (from reference set)")
    
    # Calculate distances between DMSO samples and DMSO centroid
    dmso_distances = []
    for _, row in dmso_df.iterrows():
        sample_embedding = row[embedding_cols].values
        dist = cosine(dmso_centroid, sample_embedding)
        dmso_distances.append(dist)
    
    # Calculate DMSO distance thresholds
    thresholds = {
        '80': np.percentile(dmso_distances, 80),
        '90': np.percentile(dmso_distances, 90),
        '95': np.percentile(dmso_distances, 95),
        '99': np.percentile(dmso_distances, 99),
        '99.9': np.percentile(dmso_distances, 99.9),
        '99.99': np.percentile(dmso_distances, 99.99)
    }
    
    # ADD THIS LOGGING:
    log_info("DMSO distance thresholds calculated:")
    for percentile, threshold in thresholds.items():
        log_info(f"  {percentile}% threshold: {threshold:.4f}")
    
    # Make sure we indicate which dataset these thresholds came from    
    if is_reference:
        log_info("These thresholds calculated from REFERENCE dataset DMSO samples (will be used for reference compounds)")
    else:
        log_info("These thresholds calculated from TEST dataset DMSO samples (will be used for test compounds)")
        
    # Calculate distance for each treatment
    log_info("Calculating distances for all treatments...")
    treatment_distances = []
    
    treatments = df['treatment'].unique()
    
    for treatment in tqdm(treatments, desc="Computing DMSO distances"):
        # Skip DMSO
        if treatment == 'DMSO':
            continue
        
        # Skip invalid treatments
        if pd.isna(treatment) or treatment == '' or treatment == 'NaN':
            continue
            
        # Get treatment data
        treatment_df = df[df['treatment'] == treatment]
        
        # Skip if no data
        if len(treatment_df) == 0:
            continue
            
        # Calculate treatment centroid
        treatment_centroid = treatment_df[embedding_cols].median().values
        
        # Calculate cosine distance from DMSO
        dist = cosine(dmso_centroid, treatment_centroid)
        
        # Check if treatment exceeds thresholds
        is_significant = {
            percentile: dist > threshold 
            for percentile, threshold in thresholds.items()
        }
        
        # Create result
        result = {
            'treatment': treatment,
            'cosine_distance_from_dmso': dist,
            'sample_count': len(treatment_df)
        }
        
        # Add threshold booleans
        for percentile in thresholds.keys():
            result[f'exceeds_{percentile.replace(".", "_")}'] = is_significant[percentile]
        
        # Add metadata from first row (expanded list)
        first_row = treatment_df.iloc[0]
        for col in ['moa', 'compound_name', 'compound_uM', 'cell_type', 'plate', 'well', 'library',
                    # Add new PP metadata columns
                    'lib_plate_order', 'perturbation_name', 'chemical_name', 
                    'supplier_ID', 'control_type', 'control_name', 'is_control',
                    'annotated_target', 'annotated_target_description', 
                    'PP_ID', 'SMILES', 'chemical_description',
                    'compound_type',
                    'manual_annotation']:
            if col in first_row:
                result[col] = first_row[col]
            elif col == 'library' and 'plate' in first_row:
                # Try to get library from plate_definitions if missing
                plate_str = str(first_row['plate'])
                if config and 'plate_definitions' in config and plate_str in config['plate_definitions']:
                    result['library'] = config['plate_definitions'][plate_str].get('library', 'Unknown')
                else:
                    result['library'] = 'Unknown'
        
        treatment_distances.append(result)

        # EARLY EXIT: Check if we have any treatment distances
        if len(treatment_distances) == 0:
            log_info("WARNING: No valid treatment distances calculated. Returning empty DataFrame.")
            return pd.DataFrame(), thresholds
    
    # Convert to DataFrame
    distance_df = pd.DataFrame(treatment_distances)

    # DIAGNOSTIC: Verify DMSO distances were calculated
    log_info(f"\n{'='*80}")
    log_info("DMSO DISTANCE CALCULATION DIAGNOSTIC")
    log_info(f"Created distance DataFrame with shape: {distance_df.shape}")
    log_info(f"Columns: {distance_df.columns.tolist()}")
    if 'cosine_distance_from_dmso' in distance_df.columns:
        log_info(f"\nDMSO distance statistics:")
        log_info(f"  Count: {distance_df['cosine_distance_from_dmso'].notna().sum()}")
        log_info(f"  Min: {distance_df['cosine_distance_from_dmso'].min():.6f}")
        log_info(f"  Max: {distance_df['cosine_distance_from_dmso'].max():.6f}")
        log_info(f"  Mean: {distance_df['cosine_distance_from_dmso'].mean():.6f}")
        log_info(f"  Median: {distance_df['cosine_distance_from_dmso'].median():.6f}")
    else:
        log_info("ERROR: cosine_distance_from_dmso column NOT found!")
    log_info(f"{'='*80}\n")

    
    # Sort by distance (descending)
    distance_df = distance_df.sort_values('cosine_distance_from_dmso', ascending=False)
    
    # Save distance results
    if 'output_dir' in config:
        output_name = 'reference_dmso_distances.csv' if is_reference else 'test_dmso_distances.csv'
        output_path = dir_paths['analysis']['dmso_distances'] / output_name
        distance_df.to_csv(output_path, index=False)
        log_info(f"Saved {label} DMSO distance results to: {output_path}")
    
    return distance_df, thresholds