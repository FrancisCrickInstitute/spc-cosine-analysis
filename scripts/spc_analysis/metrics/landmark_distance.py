# Updated landmark_distance.py with complete landmark metadata and consistent column naming
# SPC VERSION

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
from ..utils.logging import log_info, log_section
from pathlib import Path
from ..utils.logging import log_section, log_info



def build_comprehensive_metadata_lookup(dmso_distances, mad_df, landmark_distances, df):
    """
    Build complete metadata from all calculated dataframes and raw data.
    
    CRITICAL FIX: This function properly merges metrics from different sources
    to ensure the parquet files contain all required data.
    """
    log_section("BUILDING COMPREHENSIVE METADATA LOOKUP FOR PARQUET")
    
    metadata_lookup = {}
    
    # Log what data we have
    log_info(f"DMSO distances available: {dmso_distances is not None}")
    log_info(f"MAD metrics available: {mad_df is not None}")
    log_info(f"Landmark distances available: {landmark_distances is not None}")
    log_info(f"Raw data available: {df is not None}")
    
    if dmso_distances is not None and 'treatment' in dmso_distances.columns:
        log_info(f"Processing {len(dmso_distances)} DMSO distance records")
        for _, row in dmso_distances.iterrows():
            treatment = row['treatment']
            if treatment not in metadata_lookup:
                metadata_lookup[treatment] = {}
            metadata_lookup[treatment]['cosine_distance_from_dmso'] = row['cosine_distance_from_dmso']
            # log_info(f"  Added DMSO distance for {treatment}: {row['cosine_distance_from_dmso']:.4f}") # Remove excessive logging
    
    if mad_df is not None and 'treatment' in mad_df.columns:
        log_info(f"Processing {len(mad_df)} MAD metric records")
        for _, row in mad_df.iterrows():
            treatment = row['treatment']
            if treatment not in metadata_lookup:
                metadata_lookup[treatment] = {}
            for metric in ['mad_cosine', 'var_cosine', 'std_cosine', 'median_distance', 'well_count']:
                if metric in row and pd.notna(row[metric]):
                    metadata_lookup[treatment][metric] = row[metric]
                    # log_info(f"  Added {metric} for {treatment}: {row[metric]:.4f}") # Remove excessive logging
    
    # Add base metadata from raw data for treatments that might be missing
    if df is not None and 'treatment' in df.columns:
        log_info("Adding base metadata from raw data")
        treatments_in_data = df['treatment'].unique()
        for treatment in treatments_in_data:
            if treatment not in metadata_lookup:
                metadata_lookup[treatment] = {}
            
            treatment_data = df[df['treatment'] == treatment]
            if len(treatment_data) > 0:
                first_row = treatment_data.iloc[0]
                base_columns = [
                    'compound_name', 'compound_uM', 'library', 'moa', 'cell_type',
                    'plate', 'well', 'PP_ID', 'SMILES', 'annotated_target',
                    'perturbation_name', 'chemical_name', 'chemical_description',
                    'compound_type', 'manual_annotation'
                ]
                
                for col in base_columns:
                    if col in first_row and pd.notna(first_row[col]):
                        if col not in metadata_lookup[treatment]:
                            metadata_lookup[treatment][col] = first_row[col]
    
    log_info(f"Built comprehensive metadata for {len(metadata_lookup)} treatments")
    return metadata_lookup

########################################################################################
# Deprecated old version of parquet full landmark distance files
########################################################################################

# def save_full_landmark_distances_for_viz(
#     all_distances_data,
#     test_metadata,
#     landmark_metadata_lookup,
#     is_reference,
#     config,
#     dir_paths,
#     dmso_distances=None,
#     mad_df=None,
#     raw_df=None
# ):
#     """
#     Save the full distance matrix to CSV for visualization purposes.
    
#     CRITICAL FIX: Now properly integrates metrics from multiple sources
#     to ensure complete data in parquet files.
#     """
#     log_info("="*80)
#     log_info("SAVING FULL LANDMARK DISTANCE MATRIX FOR VISUALIZATION")
#     log_info("="*80)
    
#     log_info(f"Processing {len(all_distances_data)} treatments")
#     log_info(f"Available metadata keys: {list(test_metadata.keys())[:5]}...")
    
#     # Build comprehensive metadata if additional sources provided
#     if dmso_distances is not None or mad_df is not None:
#         log_info("Enhancing metadata with calculated metrics...")
#         enhanced_metadata = build_comprehensive_metadata_lookup(dmso_distances, mad_df, None, raw_df)
        
#         # Merge enhanced metadata into test_metadata
#         for treatment, enhanced_data in enhanced_metadata.items():
#             if treatment in test_metadata:
#                 test_metadata[treatment].update(enhanced_data)
#                 #log_info(f"Enhanced metadata for {treatment}") # Comment out
#             else:
#                 test_metadata[treatment] = enhanced_data
#                 log_info(f"Added missing treatment {treatment} to metadata")
    
#     # Build rows for CSV
#     rows = []
#     treatments_with_missing_metrics = []
    
#     for treatment_data in tqdm(all_distances_data, desc="Building full distance matrix"):
#         treatment = treatment_data['treatment']
#         distances = treatment_data['distances']  # List of (landmark_name, distance) tuples
        
#         # Start with treatment metadata
#         row = test_metadata.get(treatment, {})
#         row['treatment'] = treatment
        
#         # CRITICAL: Check if we have the key metrics
#         missing_metrics = []
#         for metric in ['cosine_distance_from_dmso', 'mad_cosine', 'var_cosine', 'std_cosine']:
#             if metric not in row or pd.isna(row.get(metric)):
#                 missing_metrics.append(metric)
        
#         if missing_metrics:
#             treatments_with_missing_metrics.append((treatment, missing_metrics))
        
#         # Add distances to all landmarks
#         for i, (landmark_name, dist) in enumerate(distances, start=1):
#             # Add distance
#             row[f'landmark_{i}_distance'] = dist
#             row[f'landmark_{i}_name'] = landmark_name
            
#             # Add landmark metadata - UPDATED WITH ALL THREE NEW COLUMNS
#             if landmark_name in landmark_metadata_lookup:
#                 lm_meta = landmark_metadata_lookup[landmark_name]
#                 row[f'landmark_{i}_moa'] = lm_meta.get('moa', '')
#                 row[f'landmark_{i}_moa_first'] = lm_meta.get('moa_first', '')
#                 row[f'landmark_{i}_target'] = lm_meta.get('annotated_target', '')
#                 row[f'landmark_{i}_library'] = lm_meta.get('library', '')
#                 row[f'landmark_{i}_compound_name'] = lm_meta.get('compound_name', '')
#                 # ADD ALL THREE NEW COLUMNS HERE:
#                 row[f'landmark_{i}_PP_ID_uM'] = lm_meta.get('PP_ID_uM', '')
#                 row[f'landmark_{i}_annotated_target_description'] = lm_meta.get('annotated_target_description', '')  # ADD THIS
#                 row[f'landmark_{i}_annotated_target_description_truncated_10'] = lm_meta.get('annotated_target_description_truncated_10', '')  # MAKE SURE THIS IS HERE
#                 # NEW: 5 additional landmark metadata columns
#                 row[f'landmark_{i}_perturbation_name'] = lm_meta.get('perturbation_name', '')
#                 row[f'landmark_{i}_chemical_name'] = lm_meta.get('chemical_name', '')
#                 row[f'landmark_{i}_chemical_description'] = lm_meta.get('chemical_description', '')
#                 row[f'landmark_{i}_compound_type'] = lm_meta.get('compound_type', '')
#                 row[f'landmark_{i}_manual_annotation'] = lm_meta.get('manual_annotation', '')
#             else:
#                 row[f'landmark_{i}_moa'] = ''
#                 row[f'landmark_{i}_moa_first'] = ''
#                 row[f'landmark_{i}_target'] = ''
#                 row[f'landmark_{i}_library'] = ''
#                 row[f'landmark_{i}_compound_name'] = ''
#                 # ADD ALL THREE NEW COLUMNS HERE TOO:
#                 row[f'landmark_{i}_PP_ID_uM'] = ''
#                 row[f'landmark_{i}_annotated_target_description'] = ''  # ADD THIS
#                 row[f'landmark_{i}_annotated_target_description_truncated_10'] = ''  # ADD THIS
#                 # NEW: Empty values for 5 additional columns when landmark not found
#                 row[f'landmark_{i}_perturbation_name'] = ''
#                 row[f'landmark_{i}_chemical_name'] = ''
#                 row[f'landmark_{i}_chemical_description'] = ''
#                 row[f'landmark_{i}_compound_type'] = ''
#                 row[f'landmark_{i}_manual_annotation'] = ''
        
#         rows.append(row)
    
#     # Log missing metrics summary
#     if treatments_with_missing_metrics:
#         log_info(f"WARNING: {len(treatments_with_missing_metrics)} treatments missing key metrics:")
#         for treatment, missing in treatments_with_missing_metrics[:10]:  # Show first 10
#             log_info(f"  {treatment}: missing {missing}")
#         if len(treatments_with_missing_metrics) > 10:
#             log_info(f"  ... and {len(treatments_with_missing_metrics) - 10} more")
    
#     # Create DataFrame
#     full_df = pd.DataFrame(rows)
    
#     log_info(f"Created DataFrame: {full_df.shape[0]} rows × {full_df.shape[1]} columns")
    
#     # CRITICAL: Verify key columns are present and have data
#     key_columns_to_check = [
#         'cosine_distance_from_dmso', 'mad_cosine', 'var_cosine', 'std_cosine',
#         'compound_name', 'library', 'moa'
#     ]
    
#     log_info("DATA VERIFICATION IN PARQUET:")
#     for col in key_columns_to_check:
#         if col in full_df.columns:
#             non_null = full_df[col].notna().sum()
#             total = len(full_df)
#             log_info(f"  {col}: {non_null}/{total} non-null ({non_null/total*100:.1f}%)")
#             if non_null > 0:
#                 sample_val = full_df[col].iloc[0] if pd.notna(full_df[col].iloc[0]) else "N/A"
#                 log_info(f"    Sample: {sample_val}")
#         else:
#             log_info(f"  {col}: COLUMN MISSING!")
    
#     # Save to Parquet
#     output_dir = dir_paths['data']
#     filename = f'landmark_distances_full_{"reference" if is_reference else "test"}_for_viz.parquet'
#     output_path = output_dir / filename
    
#     log_info(f"Saving to: {output_path}")
#     full_df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
    
#     # Report file size
#     file_size_mb = output_path.stat().st_size / (1024**2)
#     log_info(f"File size: {file_size_mb:.2f} MB")
    
#     # Also save as CSV for debugging
#     csv_path = output_dir / f'landmark_distances_full_{"reference" if is_reference else "test"}_for_viz_DEBUG.csv'
#     full_df.head(100).to_csv(csv_path, index=False)  # Save first 100 rows for debugging
#     log_info(f"Saved debug sample to: {csv_path}")
    
#     return full_df


def save_slim_landmark_parquet_files(
    treatment_centroid_dict,
    landmarks,
    landmark_embeddings,
    landmark_metadata_lookup,
    embedding_cols,
    is_reference,
    config,
    dir_paths,
    df,  # NEW: Add raw dataframe to extract comprehensive metadata
    dmso_distances=None,
    mad_df=None
):
    """
    Save slim parquet files for landmark analysis (CellProfiler-style)
    
    Creates 3 files:
    1. landmark_metadata.parquet (saved once, alphabetically sorted)
    2. reference_distances.parquet OR test_distances.parquet
    3. CSV samples of each (first 200 rows)
    
    Format:
    - Each landmark = single column named '{treatment}_distance'
    - Landmarks sorted alphabetically (same order in test and reference)
    - Query metadata included (ALL metadata columns, matching landmark_metadata)
    
    Args:
        treatment_centroid_dict: Dict of {treatment: centroid_vector}
        landmarks: DataFrame of landmark treatments
        landmark_embeddings: Dict of {landmark_name: embedding_vector}
        landmark_metadata_lookup: Dict of {landmark_name: metadata_dict}
        embedding_cols: List of embedding column names
        is_reference: Whether this is reference or test data
        config: Configuration dictionary
        dir_paths: Directory paths dictionary
        df: Raw dataframe with all treatment metadata (NEW!)
        dmso_distances: Optional DMSO distance DataFrame
        mad_df: Optional MAD metrics DataFrame
    
    Returns:
        pd.DataFrame: The slim distance matrix
    """
    log_section(f"CREATING SLIM LANDMARK PARQUET FILES ({'REFERENCE' if is_reference else 'TEST'})")
    log_info("NEW FORMAT: Distances + landmark metadata saved separately")
    log_info("Expected size reduction: 68GB → 500MB per file!")
    log_info("="*80)
    
    output_dir = dir_paths['analysis']['landmark_distances']
    
    # ========================================================================
    # STEP 1: Save landmark metadata (only once, not for each dataset)
    # ========================================================================
    landmark_meta_path = output_dir / 'landmark_metadata.parquet'
    
    if not landmark_meta_path.exists():
        log_info("\nSaving landmark metadata (saved once, not repeated)...")
        
        # Build landmark metadata list
        landmark_metadata_list = []
        for landmark_name in landmark_metadata_lookup.keys():
            lm_meta = {'treatment': landmark_name}
            lm_meta.update(landmark_metadata_lookup[landmark_name])
            landmark_metadata_list.append(lm_meta)
        
        landmark_metadata_df = pd.DataFrame(landmark_metadata_list)
        
        # CRITICAL: Sort by treatment name for consistent ordering
        landmark_metadata_df = landmark_metadata_df.sort_values('treatment').reset_index(drop=True)
        
        landmark_metadata_df.to_parquet(landmark_meta_path, index=False)
        
        # Also save CSV sample (first 200 rows)
        landmark_meta_csv_path = output_dir / 'landmark_metadata_sample.csv'
        landmark_metadata_df.head(200).to_csv(landmark_meta_csv_path, index=False)
        
        landmark_size_mb = landmark_meta_path.stat().st_size / (1024 * 1024)
        log_info(f"✓ Saved landmark metadata: {landmark_meta_path}")
        log_info(f"  Size: {landmark_size_mb:.1f} MB ({len(landmark_metadata_df)} landmarks)")
        log_info(f"  Columns: {len(landmark_metadata_df.columns)}")
        log_info(f"  Landmarks sorted alphabetically for consistent ordering")
        log_info(f"✓ Saved CSV sample: {landmark_meta_csv_path} (first 200 rows)")
    else:
        landmark_metadata_df = pd.read_parquet(landmark_meta_path)
        log_info(f"✓ Landmark metadata already exists: {landmark_meta_path}")
        log_info(f"  Loaded {len(landmark_metadata_df)} landmarks")
    
    # ========================================================================
    # STEP 2: Build slim distance matrix
    # ========================================================================
    log_info(f"\nBuilding slim distance matrix for {len(treatment_centroid_dict)} treatments...")
    
    # Prepare landmark embeddings in alphabetical order
    landmark_names_ordered = sorted(list(landmark_embeddings.keys()))
    landmark_matrix = np.vstack([landmark_embeddings[name] for name in landmark_names_ordered])
    log_info(f"  Landmark matrix shape: {landmark_matrix.shape}")
    log_info(f"  Landmarks sorted alphabetically for consistent ordering")
    
    # Prepare treatment centroids matrix
    valid_treatments = list(treatment_centroid_dict.keys())
    treatment_centroids_matrix = np.vstack([treatment_centroid_dict[t] for t in valid_treatments])
    log_info(f"  Treatment matrix shape: {treatment_centroids_matrix.shape}")
    
    # VECTORIZED COMPUTATION: Calculate all distances at once
    log_info("  Computing all pairwise distances (vectorized)...")
    all_distances = cosine_distances(treatment_centroids_matrix, landmark_matrix)
    log_info(f"  Distance matrix shape: {all_distances.shape}")
    
    # ========================================================================
    # STEP 3: Build output dataframe with slim format
    # ========================================================================
    log_info("  Building slim output dataframe (single distance column per landmark)...")
    
    rows = []
    self_matches_excluded = 0
    landmark_treatments = set(landmarks['treatment'].unique())
    
    for treatment_idx, treatment in enumerate(tqdm(valid_treatments, desc="Processing treatments")):
        # Start with treatment info
        row = {'treatment': treatment}
        
        # ========================================================================
        # COMPREHENSIVE METADATA EXTRACTION (matching landmark_metadata.parquet)
        # ========================================================================
        # Get first row of raw data for this treatment to extract ALL metadata
        treatment_data = df[df['treatment'] == treatment]
        if len(treatment_data) > 0:
            first_row = treatment_data.iloc[0]
            
            # Add base metadata columns
            base_metadata_columns = [
                'SMILES', 'chemical_description', 'moa', 'annotated_target_description',
                'PP_ID', 'compound_name', 'compound_uM', 'supplier_ID', 'is_control',
                'lib_plate_order', 'perturbation_name', 'chemical_name', 'library',
                'annotated_target', 'well', 'plate', 'cell_type', 'manual_annotation'
            ]
            
            for col in base_metadata_columns:
                if col in first_row:
                    row[col] = first_row[col]
            
            # ====================================================================
            # GENERATE DERIVED COLUMNS (same logic as landmarks)
            # ====================================================================
            
            # 1. MOA derivatives
            if 'moa' in row and pd.notna(row['moa']):
                moa_value = str(row['moa'])
                
                # moa_truncated_10 (first 10 words)
                row['moa_truncated_10'] = ' '.join(moa_value.split()[:10])
                
                # moa_first (first gene/target before comma)
                row['moa_first'] = moa_value.split(',')[0].strip() if ',' in moa_value else moa_value
                
                # moa_compound_uM (MOA @ concentration)
                if 'compound_uM' in row and pd.notna(row['compound_uM']):
                    compound_um = row['compound_uM']
                    if compound_um != 0:
                        row['moa_compound_uM'] = f"{moa_value}@{compound_um}"
                    else:
                        row['moa_compound_uM'] = moa_value
                else:
                    row['moa_compound_uM'] = moa_value
            
            # 2. Annotated target description truncated
            if 'annotated_target_description' in row and pd.notna(row['annotated_target_description']):
                target_desc = str(row['annotated_target_description'])
                row['annotated_target_description_truncated_10'] = ' '.join(target_desc.split()[:10])
            
            # 3. Chemical description truncated
            if 'chemical_description' in row and pd.notna(row['chemical_description']):
                chem_desc = str(row['chemical_description'])
                row['chemical_description_truncated_10'] = ' '.join(chem_desc.split()[:10])
            
            # 4. PP_ID_uM (PP_ID @ concentration)
            if 'PP_ID' in row and 'compound_uM' in row:
                pp_id = row['PP_ID']
                compound_um = row['compound_uM']
                
                if pd.notna(pp_id) and str(pp_id) != 'nan':
                    if pd.notna(compound_um) and compound_um != 0:
                        row['PP_ID_uM'] = f"{pp_id}@{compound_um}"
                    else:
                        row['PP_ID_uM'] = str(pp_id)
                else:
                    row['PP_ID_uM'] = None
        
        # Add is_reference flag
        row['is_reference'] = is_reference
        
        # Add query metrics if available
        if dmso_distances is not None:
            dmso_match = dmso_distances[dmso_distances['treatment'] == treatment]
            if len(dmso_match) > 0:
                row['query_dmso_distance'] = dmso_match.iloc[0].get('cosine_distance_from_dmso')
        
        if mad_df is not None:
            mad_match = mad_df[mad_df['treatment'] == treatment]
            if len(mad_match) > 0:
                row['query_mad'] = mad_match.iloc[0].get('mad_cosine')
                
                # Add is_landmark flag if reference
                if is_reference:
                    row['is_landmark'] = mad_match.iloc[0].get('is_landmark', False)
        
        # Get distances for this treatment (already computed!)
        query_distances = all_distances[treatment_idx]
        
        # Create list of (landmark_name, distance) tuples
        landmark_distances = list(zip(landmark_names_ordered, query_distances))
        
        # Handle self-matching (exclude for reference landmarks)
        if is_reference and treatment in landmark_treatments:
            original_len = len(landmark_distances)
            landmark_distances = [(name, dist) for name, dist in landmark_distances 
                                if name != treatment]
            if len(landmark_distances) < original_len:
                self_matches_excluded += 1
        
        # NEW FORMAT: Add each landmark as {treatment}_distance column
        for lm_treatment, dist in landmark_distances:
            row[f'{lm_treatment}_distance'] = dist
        
        rows.append(row)
    
    if self_matches_excluded > 0:
        log_info(f"  Excluded {self_matches_excluded} self-matches for reference landmarks")
    
    # Create DataFrame
    distance_df = pd.DataFrame(rows)
    
    # Count distance columns
    distance_cols = [col for col in distance_df.columns 
                    if col.endswith('_distance') and col != 'query_dmso_distance']
    log_info(f"  ✓ Slim matrix created: {distance_df.shape}")
    log_info(f"  Distance columns: {len(distance_cols)} (one per landmark)")
    
    # ========================================================================
    # STEP 4: Save distance matrix
    # ========================================================================
    distance_filename = 'reference_distances.parquet' if is_reference else 'test_distances.parquet'
    distance_path = output_dir / distance_filename
    
    distance_df.to_parquet(distance_path, index=False)
    
    # Also save CSV sample
    distance_csv_path = output_dir / f'{"reference" if is_reference else "test"}_distances_sample.csv'
    distance_df.head(200).to_csv(distance_csv_path, index=False)
    
    distance_size_mb = distance_path.stat().st_size / (1024 * 1024)
    distance_size_gb = distance_size_mb / 1024
    
    log_info(f"\n✓ Saved distance matrix: {distance_path}")
    log_info(f"  Matrix shape: {distance_df.shape}")
    log_info(f"  File size: {distance_size_gb:.2f} GB")
    log_info(f"✓ Saved CSV sample: {distance_csv_path} (first 200 rows)")
    
    return distance_df


def save_treatment_centroids(
    treatment_centroid_dict,
    df,
    embedding_cols,
    is_reference,
    dir_paths
):
    """
    Save treatment centroids to parquet file for resume capability
    
    Args:
        treatment_centroid_dict: Dict of {treatment: centroid_vector}
        df: Original dataframe with treatment metadata
        embedding_cols: List of embedding column names
        is_reference: Whether this is reference or test data
        dir_paths: Directory paths dictionary
    
    Returns:
        pd.DataFrame: The centroids dataframe
    """
    centroid_filename = 'reference_centroids.parquet' if is_reference else 'test_centroids.parquet'
    centroid_path = dir_paths['analysis']['landmark_distances'] / centroid_filename
    
    if centroid_path.exists():
        log_info(f"✓ {centroid_filename} already exists, skipping")
        return pd.read_parquet(centroid_path)
    
    log_info(f"\nSaving {centroid_filename}...")
    
    # Build centroids dataframe with ALL metadata + features
    centroids_list = []
    for treatment, centroid_vector in treatment_centroid_dict.items():
        treatment_df = df[df['treatment'] == treatment]
        if len(treatment_df) == 0:
            continue
        
        first_row = treatment_df.iloc[0]
        centroid_row = {'treatment': treatment}
        
        # Add metadata (all columns that start with standard names or are in known list)
        metadata_cols = [col for col in first_row.index 
                        if col in ['library', 'moa', 'compound_name', 'compound_uM', 
                                  'cell_type', 'plate', 'well', 'PP_ID', 'SMILES',
                                  'annotated_target', 'perturbation_name', 'chemical_name']]
        for col in metadata_cols:
            if col in first_row:
                centroid_row[col] = first_row[col]
        
        # Add centroid features
        for i, val in enumerate(centroid_vector):
            centroid_row[embedding_cols[i]] = val
        
        # Add replicate count
        centroid_row['n_replicates'] = len(treatment_df)
        
        centroids_list.append(centroid_row)
    
    centroids_df = pd.DataFrame(centroids_list)
    centroids_df.to_parquet(centroid_path, index=False)
    
    file_size_mb = centroid_path.stat().st_size / (1024 * 1024)
    log_info(f"✓ Saved {centroid_filename} ({file_size_mb:.1f} MB)")
    log_info(f"  {len(centroids_df)} treatments with {len(embedding_cols)} features")
    
    return centroids_df


def compute_landmark_distance(df, landmarks, embedding_cols, is_reference, config, dir_paths, dmso_distances=None, mad_df=None):
    """
    Compute distances to landmarks for compounds.
    
    CRITICAL FIX: Now accepts dmso_distances and mad_df parameters to ensure
    complete metadata in output files.
    """
    log_section(f"COMPUTING LANDMARK DISTANCES FOR {'REFERENCE' if is_reference else 'TEST'} SET")
    
    # Add debug info about input parameters
    log_info(f"Input parameters debug:")
    log_info(f"  df shape: {df.shape if df is not None else 'None'}")
    log_info(f"  landmarks count: {len(landmarks) if landmarks is not None else 'None'}")
    log_info(f"  dmso_distances provided: {dmso_distances is not None}")
    log_info(f"  mad_df provided: {mad_df is not None}")
    
    if dmso_distances is not None:
        log_info(f"  dmso_distances columns: {dmso_distances.columns.tolist()}")
        log_info(f"  dmso_distances treatments: {len(dmso_distances['treatment'].unique())}")
    
    if mad_df is not None:
        log_info(f"  mad_df columns: {mad_df.columns.tolist()}")
        log_info(f"  mad_df treatments: {len(mad_df['treatment'].unique())}")
    
    # Skip if landmarks is None
    if landmarks is None or len(landmarks) == 0:
        log_info("No landmarks available. Cannot compute landmark distances.")
        return None
    
    # Extract landmark embeddings
    log_info(f"Using {len(landmarks)} landmarks")
    
    landmark_embeddings = {}
    landmark_treatments = landmarks['treatment'].unique()
    log_info(f"Found {len(landmark_treatments)} unique landmark treatments")
    
    # Build landmark metadata lookup from the landmarks dataframe
    landmark_metadata_lookup = {}
    
    # COMPLETE metadata columns to preserve for landmarks (no duplicates)
    metadata_cols_to_preserve = [
        # Chemical properties
        'SMILES', 
        'chemical_description', 
        
        # MOA and target information (primary columns - no duplicates)
        'moa',  # Primary MOA field from original metadata
        'annotated_target_description',  # Primary target description from PP file
        
        # Compound identification  
        'PP_ID', 
        'compound_name', 
        'compound_uM',
        'compound_type', 
        
        # Additional metadata from PP file
        'supplier_ID', 
        'control_type', 
        'control_name', 
        'is_control', 
        'lib_plate_order', 
        'perturbation_name', 
        'chemical_name',
        'library', 
        'annotated_target',
        'well',  
        'plate',  
        'cell_type',
        'manual_annotation'
    ]

    log_info("Building landmark metadata lookup with complete derived columns...")
    for _, landmark_row in landmarks.iterrows():
        treatment = landmark_row['treatment']
        if treatment not in landmark_metadata_lookup:
            landmark_metadata_lookup[treatment] = {}
            
            # Store base metadata columns
            for col in metadata_cols_to_preserve:
                if col in landmark_row and pd.notna(landmark_row[col]):
                    landmark_metadata_lookup[treatment][col] = landmark_row[col]
            
            # Create ALL derived columns for landmarks (matching what compounds get)
            
            # 1. MOA derivatives (from original moa column)
            if 'moa' in landmark_metadata_lookup[treatment]:
                moa_value = str(landmark_metadata_lookup[treatment]['moa'])
                
                # moa_truncated_10 (first 10 words)
                landmark_metadata_lookup[treatment]['moa_truncated_10'] = ' '.join(moa_value.split()[:10])
                
                # moa_first (first gene/target before comma)
                landmark_metadata_lookup[treatment]['moa_first'] = moa_value.split(',')[0].strip() if ',' in moa_value else moa_value
                
                # moa_compound_uM (MOA @ concentration)
                if 'compound_uM' in landmark_metadata_lookup[treatment]:
                    compound_um = landmark_metadata_lookup[treatment]['compound_uM']
                    if pd.notna(compound_um) and compound_um != 0:
                        landmark_metadata_lookup[treatment]['moa_compound_uM'] = f"{moa_value}@{compound_um}"
                    else:
                        landmark_metadata_lookup[treatment]['moa_compound_uM'] = moa_value
                else:
                    landmark_metadata_lookup[treatment]['moa_compound_uM'] = moa_value
            
            # 2. Annotated target description truncated (from PP file)
            if 'annotated_target_description' in landmark_metadata_lookup[treatment]:
                target_desc = str(landmark_metadata_lookup[treatment]['annotated_target_description'])
                landmark_metadata_lookup[treatment]['annotated_target_description_truncated_10'] = ' '.join(target_desc.split()[:10])
            
            # 3. Chemical description truncated
            if 'chemical_description' in landmark_metadata_lookup[treatment]:
                chem_desc = str(landmark_metadata_lookup[treatment]['chemical_description'])
                landmark_metadata_lookup[treatment]['chemical_description_truncated_10'] = ' '.join(chem_desc.split()[:10])
            
            # 4. PP_ID @ concentration (PP_ID_uM equivalent for landmarks)
            if 'PP_ID' in landmark_metadata_lookup[treatment] and 'compound_uM' in landmark_metadata_lookup[treatment]:
                pp_id = landmark_metadata_lookup[treatment]['PP_ID']
                compound_um = landmark_metadata_lookup[treatment]['compound_uM']
                
                if pd.notna(pp_id) and str(pp_id) != 'nan':
                    if pd.notna(compound_um) and compound_um != 0:
                        landmark_metadata_lookup[treatment]['PP_ID_uM'] = f"{pp_id}@{compound_um}"
                    else:
                        landmark_metadata_lookup[treatment]['PP_ID_uM'] = str(pp_id)
                else:
                    landmark_metadata_lookup[treatment]['PP_ID_uM'] = None
    
    log_info(f"Built complete metadata lookup for {len(landmark_metadata_lookup)} landmarks")
    
    # Log sample of derived columns created
    if landmark_metadata_lookup:
        sample_treatment = list(landmark_metadata_lookup.keys())[0]
        sample_meta = landmark_metadata_lookup[sample_treatment]
        derived_cols = [k for k in sample_meta.keys() if any(x in k for x in ['truncated', 'first', 'uM'])]
        log_info(f"Sample derived columns created: {derived_cols}")
    
    # Check if we have precomputed landmarks
    landmark_embeddings_path = dir_paths['analysis']['root'] / 'landmark_embeddings.npz'
    
    if is_reference:
        # For reference, compute landmark embeddings and save them
        for treatment in landmark_treatments:
            # Get treatment data
            treatment_df = df[df['treatment'] == treatment]
            
            # Skip if no data
            if len(treatment_df) == 0:
                log_info(f"Warning: Landmark treatment '{treatment}' not found in reference dataset")
                continue
                
            # Calculate treatment centroid
            treatment_centroid = treatment_df[embedding_cols].median().values
            landmark_embeddings[treatment] = treatment_centroid
        
        # Save the embeddings for later use with test set
        if landmark_embeddings:
            log_info(f"Saving {len(landmark_embeddings)} landmark embeddings for use with test set")
            np.savez(landmark_embeddings_path, **{k: v for k, v in landmark_embeddings.items()})
        
    else:
        # For test, load pre-computed landmark embeddings
        if landmark_embeddings_path.exists():
            log_info(f"Loading pre-computed landmark embeddings from {landmark_embeddings_path}")
            try:
                loaded = np.load(landmark_embeddings_path, allow_pickle=True)
                for treatment in landmark_treatments:
                    if treatment in loaded:
                        landmark_embeddings[treatment] = loaded[treatment]
                log_info(f"Successfully loaded {len(landmark_embeddings)} landmark embeddings")
            except Exception as e:
                log_info(f"Error loading landmark embeddings: {str(e)}")
        else:
            log_info("Warning: No pre-computed landmark embeddings found. Cannot compute test landmark distances.")
    
    log_info(f"Using {len(landmark_embeddings)} landmarks for distance calculation")
    
    # If we have no landmark embeddings, we can't continue
    if len(landmark_embeddings) == 0:
        log_info("Error: No valid landmark embeddings available. Cannot compute landmark distances.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Get unique treatments from the dataset
    treatments = df['treatment'].unique()
    # FILTER AWAY DMSO treatments
    treatments = [t for t in treatments if not pd.isna(t) and not str(t).startswith('DMSO')]
    log_info(f"Found {len(treatments)} unique treatments (DMSO excluded) to analyze")
    
    # Updated logic to exclude self-matching for reference landmarks
    log_info("UPDATED: Analyzing ALL treatments but excluding self-matches for reference landmarks")
    
    # Define threshold for similarity calculation
    similarity_threshold = config.get('similarity_threshold', 0.2)
    log_info(f"Similarity threshold for phenotypic makeup: {similarity_threshold}")

    # Calculate distances to landmarks
    landmark_distances = []

    # Print first few treatments for debugging
    log_info(f"Sample treatments to analyze: {treatments[:5] if len(treatments) > 5 else treatments}")
    
    # Track self-matching for debugging
    self_matches_excluded = 0
    
    # VECTORIZATION: Pre-compute all treatment centroids and landmark embeddings as matrices
    log_info("Pre-computing treatment centroids for vectorized distance calculation...")
    treatment_centroid_dict = {}
    for treatment in tqdm(treatments, desc="Computing treatment centroids"):
        # Skip invalid treatments
        if pd.isna(treatment) or treatment == '' or treatment == 'NaN':
            continue
            
        # Get treatment data
        treatment_df = df[df['treatment'] == treatment]
        
        # Skip if no data
        if len(treatment_df) == 0:
            continue
            
        # Calculate treatment centroid
        treatment_centroid_dict[treatment] = treatment_df[embedding_cols].median().values
    
    # Create ordered lists for vectorized computation
    valid_treatments = list(treatment_centroid_dict.keys())
    treatment_centroids_matrix = np.vstack([treatment_centroid_dict[t] for t in valid_treatments])
    
    # Prepare landmark embeddings as matrix
    landmark_names_list = list(landmark_embeddings.keys())
    landmark_matrix = np.vstack([landmark_embeddings[name] for name in landmark_names_list])
    
    # VECTORIZED COMPUTATION: Calculate all distances at once using sklearn
    log_info(f"Computing {len(valid_treatments)} x {len(landmark_names_list)} distance matrix...")
    distance_matrix = cosine_distances(treatment_centroids_matrix, landmark_matrix)
    log_info(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Now process each treatment using the pre-computed distances
    for treatment_idx, treatment in enumerate(tqdm(valid_treatments, desc="Processing landmark distances")):
        # Check if this treatment is itself a landmark
        is_self_landmark = treatment in landmark_treatments
        
        # Get distances for this treatment from the pre-computed matrix
        distances = []
        for landmark_idx, landmark_name in enumerate(landmark_names_list):
            # Skip self-matching for reference landmarks
            if is_reference and is_self_landmark and treatment == landmark_name:
                self_matches_excluded += 1
                continue  # Skip self-matching
                
            dist = distance_matrix[treatment_idx, landmark_idx]
            distances.append((landmark_name, dist))
        
        # Get treatment data for metadata
        treatment_df = df[df['treatment'] == treatment]
        
        # Skip if no landmarks found (shouldn't happen unless all were self-matches)
        if not distances:
            log_info(f"No valid landmark distances found for treatment: {treatment}")
            continue
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Get top 3 nearest landmarks
        top_landmarks = distances[:3]
        
        # Calculate phenotypic makeup if within threshold
        closest_dist = top_landmarks[0][1] if top_landmarks else 1.0
        
        # Determine validity for phenotypic makeup
        valid_for_makeup = closest_dist <= similarity_threshold

        # Create result dictionary
        result = {
            'treatment': treatment,
            'sample_count': len(treatment_df),
            'valid_for_phenotypic_makeup': valid_for_makeup,
            'is_self_landmark': is_self_landmark
        }
        
        # Add top 3 landmarks with COMPLETE metadata including all derived columns
        for i, (landmark_name, dist) in enumerate(top_landmarks[:3] if top_landmarks else []):
            position = ['closest', 'second_closest', 'third_closest'][i]
            
            # Basic landmark info
            result[f'{position}_landmark'] = landmark_name
            result[f'{position}_landmark_distance'] = dist
            result[f'{position}_is_landmark'] = landmark_name in landmark_treatments
            
            # Add ALL metadata with consistent naming
            if landmark_name in landmark_metadata_lookup:
                lm_meta = landmark_metadata_lookup[landmark_name]
                
                # Base metadata columns
                base_metadata = [
                    'SMILES', 'chemical_description', 'moa', 'annotated_target_description',
                    'PP_ID', 'compound_name', 'compound_uM', 'supplier_ID', 'control_type',
                    'control_name', 'is_control', 'lib_plate_order', 'perturbation_name', 'chemical_name',
                    'manual_annotation'
                ]
                
                # Derived columns (all the truncated and compound versions)
                derived_columns = [
                    'moa_truncated_10', 'moa_first', 'moa_compound_uM',
                    'annotated_target_description_truncated_10', 
                    'PP_ID_uM'
                ]
                
                # Add all columns to result
                for meta_key in base_metadata + derived_columns:
                    if meta_key in landmark_metadata_lookup[landmark_name]:
                        result[f'{position}_landmark_{meta_key}'] = landmark_metadata_lookup[landmark_name][meta_key]
                    else:
                        result[f'{position}_landmark_{meta_key}'] = None
                    
        # Simple flag - no complex calculations needed
        result['makeup_calculation_issue'] = None  # Remove this entirely if you want
        
        # Add metadata from first row (expanded list, no duplicate target_description)
        first_row = treatment_df.iloc[0]
        metadata_cols = ['moa', 'compound_name', 'compound_uM', 'cell_type', 'plate', 'well', 'library',
                        # PP metadata columns (no duplicates)
                        'lib_plate_order', 'perturbation_name', 'chemical_name', 
                        'supplier_ID', 'control_type', 'control_name', 'is_control',
                        'annotated_target_description',  # Primary target column
                        'PP_ID', 'SMILES', 'chemical_description',
                        'compound_type',
                        'manual_annotation']
        
        for col in metadata_cols:
            if col in first_row:
                result[col] = first_row[col]
        
        landmark_distances.append(result)
    
    # Log self-matching exclusion results
    if self_matches_excluded > 0:
        log_info(f"SUCCESS: Excluded {self_matches_excluded} self-matches for reference landmarks")
    else:
        log_info("No self-matches found to exclude")
    
    # Convert to DataFrame
    distance_df = pd.DataFrame(landmark_distances)
    
    # Sort by distance to closest landmark
    if 'closest_landmark_distance' in distance_df.columns:
        distance_df = distance_df.sort_values('closest_landmark_distance', ascending=True)
    
    # Log summary of landmark analysis with complete metadata
    if len(distance_df) > 0:
        # Count landmark metadata columns
        landmark_meta_cols = [col for col in distance_df.columns if 'landmark_' in col]
        derived_landmark_cols = [col for col in landmark_meta_cols if any(x in col for x in ['truncated', 'first', '_uM'])]
        
        log_info(f"Generated {len(landmark_meta_cols)} total landmark columns")
        log_info(f"Including {len(derived_landmark_cols)} derived columns (truncated, first, @concentration)")
        
        # Log sample of new derived columns
        sample_derived = [col for col in derived_landmark_cols if 'closest' in col][:5]
        log_info(f"Sample derived landmark columns: {sample_derived}")
        
        if 'is_self_landmark' in distance_df.columns:
            landmark_count = distance_df['is_self_landmark'].sum()
            log_info(f"Analyzed {landmark_count} landmarks and {len(distance_df) - landmark_count} non-landmarks")
            
            # Show sample of landmark distances (should now show different nearest neighbors)
            landmarks_analyzed = distance_df[distance_df['is_self_landmark'] == True]
            if len(landmarks_analyzed) > 0:
                log_info("Sample landmark-to-landmark distances (self-matches excluded):")
                for i, (_, row) in enumerate(landmarks_analyzed.head(3).iterrows()):
                    closest = row.get('closest_landmark', 'Unknown')
                    dist = row.get('closest_landmark_distance', 'Unknown')
                    closest_moa = row.get('closest_landmark_moa', 'Unknown')
                    log_info(f"  {row['treatment']} -> {closest} ({closest_moa}) (distance: {dist:.4f})")


        # Save SLIM parquet files (CellProfiler-style)
        log_info("\n" + "="*80)
        log_info("CREATING SLIM PARQUET FILES (CellProfiler-style)")
        log_info("="*80)
        save_slim_landmark_parquet_files(
            treatment_centroid_dict=treatment_centroid_dict,
            landmarks=landmarks,
            landmark_embeddings=landmark_embeddings,
            landmark_metadata_lookup=landmark_metadata_lookup,
            embedding_cols=embedding_cols,
            is_reference=is_reference,
            config=config,
            dir_paths=dir_paths,
            df=df,
            dmso_distances=dmso_distances,
            mad_df=mad_df
        )
        
        # NEW: Save treatment centroids
        log_info("\n" + "="*80)
        log_info("SAVING TREATMENT CENTROIDS")
        log_info("="*80)
        save_treatment_centroids(
            treatment_centroid_dict=treatment_centroid_dict,
            df=df,
            embedding_cols=embedding_cols,
            is_reference=is_reference,
            dir_paths=dir_paths
        )
    
    # Save results
    if 'output_dir' in config:
        output_name = 'reference_landmark_distances.csv' if is_reference else 'test_landmark_distances.csv'
        output_path = dir_paths['analysis']['landmark_distances'] / output_name
        distance_df.to_csv(output_path, index=False)
        log_info(f"Saved landmark distances with complete metadata to: {output_path}")
        
        # Log final column count summary
        total_cols = len(distance_df.columns)
        landmark_cols = len([col for col in distance_df.columns if 'landmark_' in col])
        log_info(f"Final output: {total_cols} total columns, {landmark_cols} landmark-related columns")
    
    return distance_df