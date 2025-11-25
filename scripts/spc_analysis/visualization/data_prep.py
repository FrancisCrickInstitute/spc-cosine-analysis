# Prepare data for umap/tSNE visualisations

import numpy as np
import pandas as pd
from pathlib import Path
from ..utils.logging import log_info, log_section
from ..data.treatment_aggregation import aggregate_to_treatment_level

def prepare_visualization_data(merged_df, landmarks=None, config=None, dir_paths=None):
    """
    Prepare visualization data by merging all metric data into a single dataframe and saving it as a CSV.
    
    UPDATED: Now also merges scoring data from compound_reference_scores.csv and compound_test_scores.csv
    
    This function:
    1. Loads all metrics files (dispersion metrics, DMSO distance)
    2. Creates standardized treatment columns for robust matching
    3. Applies multiple matching strategies (standard, case-insensitive, partial)
    4. Creates unified metric columns
    5. NEW: Merges scoring data from separate reference/test scoring files
    6. Saves the enriched dataframe as a CSV
    
    Args:
        merged_df: DataFrame with embeddings and metadata
        landmarks: DataFrame with landmark information (optional)
        config: Configuration dictionary
        dir_paths: Dictionary with directory paths
        
    Returns:
        DataFrame: Enriched dataframe with all metrics including scoring data
    """
    log_section("PREPARING VISUALIZATION DATA")
    
    # Create a copy of the dataframe to avoid modifying the original
    viz_df = merged_df.copy(deep=True)  # Use deep=True to ensure complete copy

    # REMOVE DUPLICATE COLUMNS - keep annotated_target_description as primary
    duplicate_columns_to_remove = ['target_description', 'target_description_truncated_10']

    for col in duplicate_columns_to_remove:
        if col in viz_df.columns:
            viz_df = viz_df.drop(columns=[col])
            log_info(f"Removed duplicate column: {col}")
    
    # Ensure we have annotated_target_description_truncated_10
    if 'annotated_target_description' in viz_df.columns and 'annotated_target_description_truncated_10' not in viz_df.columns:
        viz_df['annotated_target_description_truncated_10'] = viz_df['annotated_target_description'].apply(
            lambda x: ' '.join(str(x).split()[:10]) if pd.notna(x) and str(x) != 'nan' else None
        )
        log_info("Created 'annotated_target_description_truncated_10' column")
    
    # Add landmark labels if available - original method using provided landmarks
    if landmarks is not None and 'treatment' in landmarks.columns:
        log_info(f"Adding landmark labels based on {len(landmarks)} identified landmarks")
        landmark_treatments = set(landmarks['treatment'].unique())
        
        viz_df = viz_df.copy()  # Ensure we have a copy at the start
        viz_df['is_landmark'] = viz_df['treatment'].isin(landmark_treatments)
        viz_df['landmark_label'] = 'Other'
        viz_df.loc[viz_df['is_landmark'], 'landmark_label'] = 'Landmark'
        
        # If we have reference vs test split in the landmarks DataFrame
        if 'is_reference' in landmarks.columns:
            # Create a mapping of treatments to their reference/test status
            treatment_to_ref = {row['treatment']: row['is_reference'] for _, row in landmarks.iterrows()}
            
            # Apply this mapping to create more specific labels
            for treatment, is_ref in treatment_to_ref.items():
                viz_df.loc[viz_df['treatment'] == treatment, 'landmark_label'] = 'Reference Landmark' if is_ref else 'Test Landmark'
    else:
        log_info("No landmark information provided, all points labeled as 'All Data'")
        viz_df['is_landmark'] = False
        viz_df['landmark_label'] = 'All Data'
    
    # NEW: Load landmark files for each metric type and create corresponding label columns
    log_info("Loading landmark files for different metrics...")
    metrics = ['mad_cosine', 'std_cosine', 'var_cosine']
    
    # Map from internal metric names to file naming convention
    metric_file_names = {
        'mad_cosine': 'mad_cosine',
        'std_cosine': 'std_cosine', 
        'var_cosine': 'var_cosine'
    }
    
    # Initialize landmark label columns for each metric
    for metric in metrics:
        metric_short = metric.split("_")[0]
        viz_df.loc[:, f'landmark_label_{metric_short}'] = 'Other'
        viz_df.loc[:, f'is_landmark_{metric_short}'] = False
    
    # Load each metric's landmarks
    for metric in metrics:
        landmark_file = dir_paths['analysis']['root'] / f'landmarks_{metric_file_names[metric]}.csv'
        
        if landmark_file.exists():
            try:
                metric_landmarks = pd.read_csv(landmark_file)
                log_info(f"Loaded {len(metric_landmarks)} landmarks from {landmark_file}")
                
                if 'treatment' in metric_landmarks.columns:
                    metric_landmark_treatments = set(metric_landmarks['treatment'].unique())
                    metric_short = metric.split('_')[0]  # 'mad', 'std', or 'var'
                    
                    # Create boolean column
                    label_col = f'is_landmark_{metric_short}'
                    viz_df[label_col] = viz_df['treatment'].isin(metric_landmark_treatments)
                    
                    # Create categorical label column
                    cat_label_col = f'landmark_label_{metric_short}'
                    viz_df[cat_label_col] = 'Other'
                    viz_df.loc[viz_df[label_col], cat_label_col] = f'Landmark ({metric_short.upper()})'
                    
                    # Add reference/test status if available
                    if 'is_reference' in metric_landmarks.columns:
                        # Create mapping
                        treatment_to_ref = {row['treatment']: row['is_reference'] 
                                          for _, row in metric_landmarks.iterrows()}
                        
                        # Apply mapping
                        for treatment, is_ref in treatment_to_ref.items():
                            label = f'Reference Landmark ({metric_short.upper()})' if is_ref else f'Test Landmark ({metric_short.upper()})'
                            viz_df.loc[viz_df['treatment'] == treatment, cat_label_col] = label
                    
                    log_info(f"Added {viz_df[label_col].sum()} landmarks for metric {metric}")
                else:
                    log_info(f"No 'treatment' column found in {landmark_file}")
            except Exception as e:
                log_info(f"Error loading landmarks for {metric}: {str(e)}")
        else:
            log_info(f"Landmark file not found: {landmark_file}")
    
    # Clean up MOA column if it exists
    if 'moa' in viz_df.columns:
        # Fill NA values
        viz_df.loc[:, 'moa'] = viz_df['moa'].fillna('Unknown')
        log_info(f"Found {viz_df['moa'].nunique()} unique MOA values")
        
        # Create truncated MOA column (first 10 genes)
        viz_df['moa_truncated_10'] = viz_df['moa'].apply(
            lambda x: ' '.join(str(x).split()[:10]) if pd.notna(x) and str(x) != 'Unknown' else str(x)
        )
        log_info(f"Created 'moa_truncated_10' column with {viz_df['moa_truncated_10'].nunique()} unique values")
        
        # Create a column with just the first MOA
        viz_df['moa_first'] = viz_df['moa'].astype(str).apply(
            lambda x: x.split(',')[0].strip() if ',' in x else x
        )
        log_info(f"Created 'moa_first' column with {viz_df['moa_first'].nunique()} unique values")

    # Create combined MOA and compound_uM column
    if 'moa_first' in viz_df.columns and 'compound_uM' in viz_df.columns:
        log_info("Creating combined MOA and compound_uM column using moa_first")
        
        def create_moa_compound_label(row):
            # Get moa_first value, handle NaN
            moa_first = str(row['moa_first']) if pd.notna(row['moa_first']) else 'Unknown'
            
            # Get compound_uM value, handle NaN (treat as 0)
            compound_um = row['compound_uM'] if pd.notna(row['compound_uM']) else 0
            
            # If compound_uM is 0, just return moa_first
            if compound_um == 0:
                return moa_first
            else:
                # Format with @ symbol: "moa_first@compound_uM"
                return f"{moa_first}@{compound_um}"
        
        viz_df['moa_compound_uM'] = viz_df.apply(create_moa_compound_label, axis=1)
        log_info(f"Created 'moa_compound_uM' column with {viz_df['moa_compound_uM'].nunique()} unique values")
        
        # Log some examples
        sample_values = viz_df['moa_compound_uM'].value_counts().head(10)
        log_info("Sample moa_compound_uM values:")
        for value, count in sample_values.items():
            log_info(f"  {value}: {count} occurrences")

    elif 'moa_first' in viz_df.columns:
        log_info("Warning: 'compound_uM' column not found, cannot create moa_compound_uM column")
    elif 'compound_uM' in viz_df.columns:
        log_info("Warning: 'moa_first' column not found, cannot create moa_compound_uM column")
    else:
        log_info("Warning: Neither 'moa_first' nor 'compound_uM' columns found, cannot create moa_compound_uM column")


# =============================================================================
    # CRITICAL FIX: Load and merge DMSO distance data
    # =============================================================================
    log_info("Loading DMSO distance data...")

    # Load reference DMSO distances
    reference_dmso_file = dir_paths['analysis']['dmso_distances'] / 'reference_dmso_distances.csv'
    test_dmso_file = dir_paths['analysis']['dmso_distances'] / 'test_dmso_distances.csv'

    dmso_distances_loaded = False

    if reference_dmso_file.exists():
        try:
            ref_dmso_df = pd.read_csv(reference_dmso_file)
            log_info(f"Loaded reference DMSO distances: {len(ref_dmso_df)} rows")
            log_info(f"Reference DMSO columns: {ref_dmso_df.columns.tolist()}")
            
            # Merge reference DMSO distances
            if 'treatment' in ref_dmso_df.columns and 'cosine_distance_from_dmso' in ref_dmso_df.columns:
                # Select columns to merge (avoid duplicates)
                dmso_cols_to_merge = ['treatment', 'cosine_distance_from_dmso']
                # Add threshold columns
                threshold_cols = [col for col in ref_dmso_df.columns if col.startswith('exceeds_')]
                dmso_cols_to_merge.extend(threshold_cols)
                
                ref_dmso_subset = ref_dmso_df[dmso_cols_to_merge].copy()
                
                # Rename columns to indicate reference
                rename_map = {
                    'cosine_distance_from_dmso': 'reference_cosine_distance_from_dmso'
                }
                for col in threshold_cols:
                    rename_map[col] = f'reference_{col}'
                
                ref_dmso_subset = ref_dmso_subset.rename(columns=rename_map)
                
                # Merge into viz_df
                original_shape = viz_df.shape
                viz_df = viz_df.merge(ref_dmso_subset, on='treatment', how='left')
                log_info(f"Merged reference DMSO distances: {original_shape} -> {viz_df.shape}")
                
                # Check how many got values
                non_null = viz_df['reference_cosine_distance_from_dmso'].notna().sum()
                log_info(f"  {non_null} rows have reference DMSO distances ({non_null/len(viz_df)*100:.1f}%)")
                dmso_distances_loaded = True
            else:
                log_info("ERROR: Required columns missing in reference DMSO file")
        except Exception as e:
            log_info(f"Error loading reference DMSO distances: {str(e)}")
    else:
        log_info(f"Reference DMSO distance file not found: {reference_dmso_file}")

    # Load test DMSO distances
    if test_dmso_file.exists():
        try:
            test_dmso_df = pd.read_csv(test_dmso_file)
            log_info(f"Loaded test DMSO distances: {len(test_dmso_df)} rows")
            
            if 'treatment' in test_dmso_df.columns and 'cosine_distance_from_dmso' in test_dmso_df.columns:
                # Select columns to merge
                dmso_cols_to_merge = ['treatment', 'cosine_distance_from_dmso']
                threshold_cols = [col for col in test_dmso_df.columns if col.startswith('exceeds_')]
                dmso_cols_to_merge.extend(threshold_cols)
                
                test_dmso_subset = test_dmso_df[dmso_cols_to_merge].copy()
                
                # Rename columns to indicate test
                rename_map = {
                    'cosine_distance_from_dmso': 'test_cosine_distance_from_dmso'
                }
                for col in threshold_cols:
                    rename_map[col] = f'test_{col}'
                
                test_dmso_subset = test_dmso_subset.rename(columns=rename_map)
                
                # Merge into viz_df
                original_shape = viz_df.shape
                viz_df = viz_df.merge(test_dmso_subset, on='treatment', how='left')
                log_info(f"Merged test DMSO distances: {original_shape} -> {viz_df.shape}")
                
                non_null = viz_df['test_cosine_distance_from_dmso'].notna().sum()
                log_info(f"  {non_null} rows have test DMSO distances ({non_null/len(viz_df)*100:.1f}%)")
                dmso_distances_loaded = True
            else:
                log_info("ERROR: Required columns missing in test DMSO file")
        except Exception as e:
            log_info(f"Error loading test DMSO distances: {str(e)}")
    else:
        log_info(f"Test DMSO distance file not found: {test_dmso_file}")

    # Create unified cosine_distance_from_dmso column
    # Prefer reference, fall back to test
    if dmso_distances_loaded:
        log_info("Creating unified cosine_distance_from_dmso column...")
        
        if 'reference_cosine_distance_from_dmso' in viz_df.columns:
            viz_df['cosine_distance_from_dmso'] = viz_df['reference_cosine_distance_from_dmso']
            
            # Fill NaNs with test values if available
            if 'test_cosine_distance_from_dmso' in viz_df.columns:
                mask = viz_df['cosine_distance_from_dmso'].isna()
                viz_df.loc[mask, 'cosine_distance_from_dmso'] = viz_df.loc[mask, 'test_cosine_distance_from_dmso']
                
            non_null = viz_df['cosine_distance_from_dmso'].notna().sum()
            log_info(f"Created unified column with {non_null} non-null values ({non_null/len(viz_df)*100:.1f}%)")
        elif 'test_cosine_distance_from_dmso' in viz_df.columns:
            viz_df['cosine_distance_from_dmso'] = viz_df['test_cosine_distance_from_dmso']
            non_null = viz_df['cosine_distance_from_dmso'].notna().sum()
            log_info(f"Created unified column from test data: {non_null} non-null values")
    else:
        log_info("WARNING: No DMSO distance data was loaded!")

    # =============================================================================
    # PP DATA COLUMN CREATION (PP merge now happens in merge.py with well key)
    # =============================================================================
    # NOTE: PP data (PP_ID, SMILES, chemical_description, annotated_target_description, etc.)
    # is now merged in merge.py using compound_name + library + well keys to avoid duplicates.
    # This section only creates derived columns from the already-merged data.
    
    log_info("Processing PP-derived columns from pre-merged data...")
    
    # Check if PP_ID was already merged in merge.py
    if 'PP_ID' in viz_df.columns:
        pp_matches = viz_df['PP_ID'].notna().sum()
        log_info(f"Found PP_ID column with {pp_matches} non-null values ({pp_matches/len(viz_df)*100:.1f}%)")
        
        # CREATE PP_ID_uM COLUMN using the already-merged PP_IDs
        if 'compound_uM' in viz_df.columns:
            log_info("Creating combined PP_ID and compound_uM column")
            
            def create_pp_compound_label(row):
                # Get PP_ID value, handle NaN
                pp_id = str(row['PP_ID']) if pd.notna(row['PP_ID']) else None
                
                # Get compound_uM value, handle NaN (treat as 0)
                compound_um = row['compound_uM'] if pd.notna(row['compound_uM']) else 0
                
                # If no PP_ID, return None/empty
                if pp_id is None or pp_id == 'nan':
                    return None
                else:
                    # Format with @ symbol: "PP_ID@compound_uM"
                    return f"{pp_id}@{compound_um}"
            
            viz_df['PP_ID_uM'] = viz_df.apply(create_pp_compound_label, axis=1)
            log_info(f"Created 'PP_ID_uM' column with {viz_df['PP_ID_uM'].notna().sum()} non-null values")
            
            # Log some examples
            sample_values = viz_df['PP_ID_uM'].dropna().head(10).tolist()
            log_info(f"Sample PP_ID_uM values: {sample_values}")
        else:
            viz_df['PP_ID_uM'] = None
            log_info("compound_uM column not found, PP_ID_uM set to None")
        
        # Log SMILES and chemical_description status
        if 'SMILES' in viz_df.columns:
            smiles_matches = viz_df['SMILES'].notna().sum()
            log_info(f"SMILES data available for {smiles_matches} rows ({smiles_matches/len(viz_df)*100:.1f}%)")
        
        if 'chemical_description' in viz_df.columns:
            chem_desc_matches = viz_df['chemical_description'].notna().sum()
            log_info(f"Chemical description data available for {chem_desc_matches} rows ({chem_desc_matches/len(viz_df)*100:.1f}%)")
        
        # Show breakdown by library
        if 'library' in viz_df.columns:
            log_info("PP data breakdown by library:")
            for lib in viz_df['library'].unique():
                lib_df = viz_df[viz_df['library'] == lib]
                lib_pp = lib_df['PP_ID'].notna().sum()
                lib_smiles = lib_df['SMILES'].notna().sum() if 'SMILES' in lib_df.columns else 0
                log_info(f"  Library '{lib}': {lib_pp}/{len(lib_df)} PP matches, {lib_smiles} SMILES")
    
    else:
        log_info("WARNING: PP_ID column not found - PP data may not have been merged in merge.py")
        viz_df['PP_ID'] = None
        viz_df['PP_ID_uM'] = None
    
    # Ensure these columns exist (may already be present from merge.py)
    for col in ['SMILES', 'chemical_description']:
        if col not in viz_df.columns:
            viz_df[col] = None
            log_info(f"Created empty {col} column")
    
    # Handle target_description columns - use annotated_target_description as primary
    if 'target_description' not in viz_df.columns:
        if 'annotated_target_description' in viz_df.columns:
            viz_df['target_description'] = viz_df['annotated_target_description']
            log_info("Created target_description from annotated_target_description")
        else:
            viz_df['target_description'] = None
    
    if 'target_description_truncated_10' not in viz_df.columns:
        if 'target_description' in viz_df.columns:
            viz_df['target_description_truncated_10'] = viz_df['target_description'].apply(
                lambda x: ' '.join(str(x).split()[:10]) if pd.notna(x) and str(x) != 'nan' else None
            )
            log_info("Created 'target_description_truncated_10' column")
        else:
            viz_df['target_description_truncated_10'] = None
    
    # =============================================================================
    # END PP DATA COLUMN CREATION
    # =============================================================================





    ############################ DEPRECATED SECTION START ###############################
        
    # # =============================================================================
    # # END CRITICAL FIX
    # # =============================================================================

    # # Load PP numbers if file is provided
    # if config and 'pp_numbers_file' in config and config['pp_numbers_file']:
    #     pp_file_path = config['pp_numbers_file']
    #     log_info(f"Loading PP numbers from: {pp_file_path}")
        
    #     try:
    #         pp_df = pd.read_csv(pp_file_path)
    #         log_info(f"Loaded PP numbers file with {len(pp_df)} rows")
    #         log_info(f"PP file columns: {pp_df.columns.tolist()}")
            
    #         # UPDATED: Check for all required columns including SMILES and chemical description
    #         required_pp_cols = ['Metadata_PP_ID', 'Metadata_perturbation_name', 'Metadata_library']
    #         optional_pp_cols = ['Metadata_annotated_target_description', 'Metadata_SMILES', 'Metadata_chemical_description']
            
    #         # Check for required columns
    #         missing_required = [col for col in required_pp_cols if col not in pp_df.columns]
    #         if missing_required:
    #             log_info(f"Warning: Required columns missing from PP file: {missing_required}")
    #             viz_df['PP_ID'] = None
    #             viz_df['PP_ID_uM'] = None
    #             viz_df['target_description'] = None
    #             viz_df['target_description_truncated_10'] = None
    #             viz_df['SMILES'] = None
    #             viz_df['chemical_description'] = None
    #         else:
    #             # All required columns present, proceed with processing
    #             log_info("All required PP columns found, proceeding with merge...")
                
    #             # Clean the chemical names and library names for matching
    #             pp_df['chemical_name_clean'] = pp_df['Metadata_perturbation_name'].astype(str).str.strip()
    #             pp_df['library_clean'] = pp_df['Metadata_library'].astype(str).str.strip()
    #             viz_df['compound_name_clean'] = viz_df['compound_name'].astype(str).str.strip()
    #             viz_df['library_clean'] = viz_df['library'].astype(str).str.strip()
                
    #             # Start with required columns
    #             pp_columns_to_keep = ['chemical_name_clean', 'library_clean', 'Metadata_PP_ID']
                
    #             # Check for and add optional columns
    #             for optional_col in optional_pp_cols:
    #                 if optional_col in pp_df.columns:
    #                     log_info(f"Found optional column: {optional_col}")
    #                     if optional_col == 'Metadata_annotated_target_description':
    #                         pp_df['target_description_clean'] = pp_df[optional_col].astype(str).str.strip()
    #                         pp_columns_to_keep.append('target_description_clean')
    #                     elif optional_col == 'Metadata_SMILES':
    #                         pp_df['SMILES_clean'] = pp_df[optional_col].astype(str).str.strip()
    #                         pp_columns_to_keep.append('SMILES_clean')
    #                     elif optional_col == 'Metadata_chemical_description':
    #                         pp_df['chemical_description_clean'] = pp_df[optional_col].astype(str).str.strip()
    #                         pp_columns_to_keep.append('chemical_description_clean')
    #                 else:
    #                     log_info(f"Optional column not found: {optional_col}")
                
    #             log_info(f"PP columns to keep: {pp_columns_to_keep}")
                
    #             # Create mapping with compound+library as composite key, only keep non-null PP_IDs
    #             pp_mapping = pp_df[pp_df['Metadata_PP_ID'].notna()][pp_columns_to_keep].drop_duplicates()
    #             log_info(f"Found {len(pp_mapping)} unique compound-library-PP mappings")
                
    #             # Merge PP numbers using BOTH compound name AND library
    #             log_info("Merging PP data using compound name + library matching...")
    #             viz_df = viz_df.merge(
    #                 pp_mapping, 
    #                 left_on=['compound_name_clean', 'library_clean'], 
    #                 right_on=['chemical_name_clean', 'library_clean'], 
    #                 how='left'
    #             )
                
    #             # Rename and clean up columns
    #             viz_df['PP_ID'] = viz_df['Metadata_PP_ID']
                
    #             # Handle target description if it was available
    #             if 'target_description_clean' in viz_df.columns:
    #                 viz_df['target_description'] = viz_df['target_description_clean']
    #                 viz_df = viz_df.drop(columns=['target_description_clean'])
    #                 log_info("Added target_description column")
                    
    #                 # Create truncated target description
    #                 viz_df['target_description_truncated_10'] = viz_df['target_description'].apply(
    #                     lambda x: ' '.join(str(x).split()[:10]) if pd.notna(x) and str(x) != 'nan' else None
    #                 )
    #                 log_info(f"Created 'target_description_truncated_10' column")
    #             else:
    #                 viz_df['target_description'] = None
    #                 viz_df['target_description_truncated_10'] = None
                
    #             # CRITICAL FIX: Handle SMILES if it was available
    #             if 'SMILES_clean' in viz_df.columns:
    #                 viz_df['SMILES'] = viz_df['SMILES_clean']
    #                 viz_df = viz_df.drop(columns=['SMILES_clean'])
    #                 log_info("Added SMILES column")
                    
    #                 # Log some sample SMILES
    #                 smiles_sample = viz_df['SMILES'].dropna().head(5).tolist()
    #                 log_info(f"Sample SMILES values: {smiles_sample}")
    #             else:
    #                 viz_df['SMILES'] = None
    #                 log_info("SMILES column not available in PP file")
                
    #             # CRITICAL FIX: Handle chemical_description if it was available
    #             if 'chemical_description_clean' in viz_df.columns:
    #                 viz_df['chemical_description'] = viz_df['chemical_description_clean']
    #                 viz_df = viz_df.drop(columns=['chemical_description_clean'])
    #                 log_info("Added chemical_description column")
                    
    #                 # Log some sample chemical descriptions
    #                 chem_desc_sample = viz_df['chemical_description'].dropna().head(3).tolist()
    #                 log_info(f"Sample chemical descriptions: {chem_desc_sample}")
    #             else:
    #                 viz_df['chemical_description'] = None
    #                 log_info("chemical_description column not available in PP file")
                
    #             # Clean up temporary columns
    #             columns_to_drop = ['Metadata_PP_ID', 'compound_name_clean', 'library_clean']
    #             if 'chemical_name_clean' in viz_df.columns:
    #                 columns_to_drop.append('chemical_name_clean')
                
    #             viz_df = viz_df.drop(columns=[col for col in columns_to_drop if col in viz_df.columns])
                
    #             # Log results of matching
    #             pp_matches = viz_df['PP_ID'].notna().sum()
    #             smiles_matches = viz_df['SMILES'].notna().sum() if 'SMILES' in viz_df.columns else 0
    #             chem_desc_matches = viz_df['chemical_description'].notna().sum() if 'chemical_description' in viz_df.columns else 0
                
    #             log_info(f"Successfully matched {pp_matches} compounds with PP numbers using compound+library matching ({pp_matches/len(viz_df)*100:.1f}%)")
    #             log_info(f"SMILES data available for {smiles_matches} compounds ({smiles_matches/len(viz_df)*100:.1f}%)")
    #             log_info(f"Chemical description data available for {chem_desc_matches} compounds ({chem_desc_matches/len(viz_df)*100:.1f}%)")
                
    #             # Show sample matches by library
    #             if 'library' in viz_df.columns:
    #                 for lib in viz_df['library'].unique():
    #                     lib_matches = viz_df[viz_df['library'] == lib]['PP_ID'].notna().sum()
    #                     lib_total = len(viz_df[viz_df['library'] == lib])
    #                     lib_smiles = viz_df[viz_df['library'] == lib]['SMILES'].notna().sum() if 'SMILES' in viz_df.columns else 0
    #                     lib_chem_desc = viz_df[viz_df['library'] == lib]['chemical_description'].notna().sum() if 'chemical_description' in viz_df.columns else 0
    #                     log_info(f"  Library '{lib}': {lib_matches}/{lib_total} PP matches, {lib_smiles} SMILES, {lib_chem_desc} chem descriptions")
                
    #             # CREATE PP_ID_uM COLUMN using the correctly matched PP_IDs
    #             if 'PP_ID' in viz_df.columns and 'compound_uM' in viz_df.columns:
    #                 log_info("Creating combined PP_ID and compound_uM column")
                    
    #                 def create_pp_compound_label(row):
    #                     # Get PP_ID value, handle NaN
    #                     pp_id = str(row['PP_ID']) if pd.notna(row['PP_ID']) else None
                        
    #                     # Get compound_uM value, handle NaN (treat as 0)
    #                     compound_um = row['compound_uM'] if pd.notna(row['compound_uM']) else 0
                        
    #                     # If no PP_ID, return None/empty
    #                     if pp_id is None or pp_id == 'nan':
    #                         return None
    #                     else:
    #                         # Format with @ symbol: "PP_ID@compound_uM"
    #                         return f"{pp_id}@{compound_um}"
                    
    #                 viz_df['PP_ID_uM'] = viz_df.apply(create_pp_compound_label, axis=1)
    #                 log_info(f"Created 'PP_ID_uM' column with {viz_df['PP_ID_uM'].notna().sum()} non-null values")
                    
    #                 # Log some examples
    #                 sample_values = viz_df['PP_ID_uM'].dropna().head(10).tolist()
    #                 log_info(f"Sample PP_ID_uM values: {sample_values}")
    #             else:
    #                 viz_df['PP_ID_uM'] = None
                    
    #     except Exception as e:
    #         log_info(f"Error loading PP numbers: {str(e)}")
    #         viz_df['PP_ID'] = None
    #         viz_df['PP_ID_uM'] = None
    #         viz_df['target_description'] = None
    #         viz_df['target_description_truncated_10'] = None
    #         viz_df['SMILES'] = None
    #         viz_df['chemical_description'] = None
    # else:
    #     log_info("No PP numbers file specified")
    #     viz_df['PP_ID'] = None
    #     viz_df['PP_ID_uM'] = None
    #     viz_df['target_description'] = None
    #     viz_df['target_description_truncated_10'] = None
    #     viz_df['SMILES'] = None
    #     viz_df['chemical_description'] = None
    
    ############################ DEPRECATED SECTION END ###############################







    # Load gene descriptions if file is provided
    genome_gene_descriptions_file = config.get('genome_gene_descriptions')
    if genome_gene_descriptions_file:
        log_info(f"Loading gene descriptions from: {genome_gene_descriptions_file}")
        
        try:
            gene_desc_df = pd.read_csv(genome_gene_descriptions_file)
            log_info(f"Loaded gene descriptions file with {len(gene_desc_df)} rows")
            
            if 'Original Gene' in gene_desc_df.columns and 'Summary' in gene_desc_df.columns:
                # Clean the gene names for matching
                gene_desc_df['gene_clean'] = gene_desc_df['Original Gene'].astype(str).str.strip()
                
                # Create gene mapping - only keep non-null summaries
                # .drop_duplicates(subset=['gene_clean'], keep='first')
                # Only checks gene_clean column for duplicates
                # Keeps FIRST row for each gene
                
                gene_mapping = gene_desc_df[gene_desc_df['Summary'].notna()][['gene_clean', 'Summary']].drop_duplicates(subset=['gene_clean'], keep='first')
                log_info(f"Found {len(gene_mapping)} unique gene-description mappings")
                
                # Match with moa_first column if it exists
                if 'moa_first' in viz_df.columns:
                    viz_df['moa_first_clean'] = viz_df['moa_first'].astype(str).str.strip()
                    
                    # Merge gene descriptions into visualization data
                    viz_df = viz_df.merge(
                        gene_mapping, 
                        left_on='moa_first_clean', 
                        right_on='gene_clean', 
                        how='left'
                    )
                    
                    # Rename and clean up
                    viz_df['gene_description'] = viz_df['Summary']
                    viz_df = viz_df.drop(columns=['Summary', 'gene_clean', 'moa_first_clean'])
                    
                    # Log results
                    gene_matches = viz_df['gene_description'].notna().sum()
                    log_info(f"Successfully matched {gene_matches} entries with gene descriptions ({gene_matches/len(viz_df)*100:.1f}%)")
                    
                    # Show sample descriptions (truncated)
                    sample_descriptions = viz_df[viz_df['gene_description'].notna()]['gene_description'].head(3)
                    for i, desc in enumerate(sample_descriptions):
                        truncated = desc[:100] + "..." if len(desc) > 100 else desc
                        log_info(f"Sample description {i+1}: {truncated}")
                else:
                    log_info("Warning: 'moa_first' column not found, cannot match gene descriptions")
                    viz_df['gene_description'] = None
                    
            else:
                log_info("Warning: Required columns 'Original Gene' or 'Summary' not found in gene descriptions file")
                viz_df['gene_description'] = None
                
        except Exception as e:
            log_info(f"Error loading gene descriptions: {str(e)}")
            viz_df['gene_description'] = None
    else:
        log_info("No gene descriptions file specified")
        viz_df['gene_description'] = None


    # Add well row and column components if not already present
    if 'well' in viz_df.columns and 'well_row' not in viz_df.columns:
        log_info("Extracting well row and column components")
        # Extract well row (letter part) and column (number part)
        viz_df.loc[:, 'well_row'] = viz_df['well'].str[0]
        viz_df.loc[:, 'well_column'] = viz_df['well'].str[1:].str.lstrip('0')           
        
        log_info(f"Found {viz_df['well_row'].nunique()} unique well rows and {viz_df['well_column'].nunique()} unique well columns")
    
    # Create standardized treatment columns for matching
    log_info("Creating standardized treatment columns for robust matching")
    viz_df.loc[:, 'treatment_std'] = viz_df['treatment'].astype(str).str.strip()
    viz_df.loc[:, 'treatment_lower'] = viz_df['treatment_std'].str.lower()          
    
    # Create a table of unique treatments for easier matching
    treatment_table = viz_df[['treatment', 'treatment_std', 'treatment_lower']].drop_duplicates()
    log_info(f"Created treatment mapping table with {len(treatment_table)} unique treatments")
    
    # Load all metrics files - update path for new metrics files
    analysis_files = [
        ('reference_metrics', dir_paths['analysis']['mad'] / 'reference_metrics.csv', 'treatment'),
        ('test_metrics', dir_paths['analysis']['mad'] / 'test_metrics.csv', 'treatment'),
        ('reference_dmso', dir_paths['analysis']['dmso_distances'] / 'reference_dmso_distances.csv', 'treatment'),
        ('test_dmso', dir_paths['analysis']['dmso_distances'] / 'test_dmso_distances.csv', 'treatment')
    ]
    
    # Check for legacy mad files for backwards compatibility
    legacy_files = [
        ('reference_mad', dir_paths['analysis']['mad'] / 'reference_mad.csv', 'treatment'),
        ('test_mad', dir_paths['analysis']['mad'] / 'test_mad.csv', 'treatment')
    ]
    
    for legacy_key, file_path, key_column in legacy_files:
        if file_path.exists() and not any(key == legacy_key.replace('mad', 'metrics') for key, _, _ in analysis_files):
            log_info(f"Found legacy file {file_path}, adding to analysis files list")
            analysis_files.append((legacy_key, file_path, key_column))
    
    # Initialize metrics dict
    metrics_data = {}
    
    # Load each metrics file
    for metrics_key, file_path, key_column in analysis_files:
        if file_path.exists():
            log_info(f"Loading metrics from {file_path}")
            try:
                metrics_df = pd.read_csv(file_path)
                log_info(f"Loaded {len(metrics_df)} rows from {file_path}")
                
                if key_column in metrics_df.columns:
                    # Add standardized treatment columns
                    metrics_df['treatment_std'] = metrics_df[key_column].astype(str).str.strip()
                    metrics_df['treatment_lower'] = metrics_df['treatment_std'].str.lower()
                    
                    # Store metrics
                    metrics_data[metrics_key] = metrics_df
                    
                    # Log sample treatments
                    if len(metrics_df) >= 5:
                        log_info(f"SAMPLE TREATMENTS FROM {metrics_key.upper()} (FIRST 5):")
                        for i, (_, row) in enumerate(metrics_df.head(5).iterrows()):
                            log_info(f"  {i+1}. Original: '{row[key_column]}', Standardized: '{row['treatment_std']}', Lower: '{row['treatment_lower']}'")
                else:
                    log_info(f"Key column '{key_column}' not found in {file_path}")
            except Exception as e:
                log_info(f"Error loading {file_path}: {str(e)}")
        else:
            log_info(f"File not found: {file_path}")
    
    # Initialize columns for metrics - update to include all dispersion metrics
    metric_columns = {
        'reference_metrics': ['mad_cosine', 'var_cosine', 'std_cosine', 'median_distance', 'well_count'],
        'test_metrics': ['mad_cosine', 'var_cosine', 'std_cosine', 'median_distance', 'well_count'],
        'reference_mad': ['mad_cosine', 'median_distance', 'well_count'],  # Legacy support
        'test_mad': ['mad_cosine', 'median_distance', 'well_count'],       # Legacy support
        'reference_dmso': ['cosine_distance_from_dmso'],
        'test_dmso': ['cosine_distance_from_dmso']
    }
    
    # Create empty columns for all metrics
    for metrics_type, columns in metric_columns.items():
        if metrics_type in metrics_data:
            # Only create columns for metrics that actually exist in the loaded data
            available_columns = [col for col in columns if col in metrics_data[metrics_type].columns]
            
            prefix = metrics_type.split('_')[0] + '_'  # 'reference_' or 'test_'
            for col in available_columns:
                viz_df[prefix + col] = None
    
    # Initialize columns for unified metrics (no prefix)
    for col in ['cosine_distance_from_dmso', 'mad_cosine', 'var_cosine', 'std_cosine', 'median_distance', 'well_count']:
        viz_df[col] = None

    # Process each metrics file
    for metrics_key, metrics_df in metrics_data.items():
        if metrics_df is None or len(metrics_df) == 0:
            continue
            
        prefix = metrics_key.split('_')[0] + '_'  # 'reference_' or 'test_'
        columns = [col for col in metric_columns[metrics_key] if col in metrics_df.columns]
        
        if not columns:
            log_info(f"No valid columns found in {metrics_key} data, skipping...")
            continue
            
        log_info(f"Processing {metrics_key} data with columns: {columns}")
        
        # Step 1: Direct matching using treatment_std
        log_info(f"Applying {metrics_key} with direct matching...")
        
        # Count matches before update
        direct_matches = 0
        for _, metrics_row in metrics_df.iterrows():
            treatment_std = metrics_row['treatment_std']
            mask = viz_df['treatment_std'] == treatment_std
            if mask.any():
                direct_matches += 1
                
                # Apply all metrics for this treatment
                for col in columns:
                    if col in metrics_row:
                        viz_df.loc[mask, prefix + col] = metrics_row[col]
        
        log_info(f"  Applied {metrics_key} metrics to {direct_matches} treatments with direct matching")

        # Steps 2 and 3: Case-insensitive and partial matching
        # (Rest of the function remains the same)
        
        # Step 2: Case-insensitive matching (if needed)
        missing_treatments = viz_df[prefix + columns[0]].isna().sum()
        if missing_treatments > 0 and 'treatment_lower' in metrics_df.columns:
            log_info(f"  Trying case-insensitive matching for {missing_treatments} remaining treatments...")
            
            case_insensitive_matches = 0
            for _, metrics_row in metrics_df.iterrows():
                treatment_lower = metrics_row['treatment_lower']
                mask = (viz_df['treatment_lower'] == treatment_lower) & (viz_df[prefix + columns[0]].isna())
                match_count = mask.sum()
                
                if match_count > 0:
                    case_insensitive_matches += 1
                    
                    # Apply all metrics for this treatment
                    for col in columns:
                        if col in metrics_row:
                            viz_df.loc[mask, prefix + col] = metrics_row[col]
            
            log_info(f"  Applied {metrics_key} metrics to {case_insensitive_matches} treatments with case-insensitive matching")
        
        # Step 3: Partial string matching (if needed)
        missing_treatments = viz_df[prefix + columns[0]].isna().sum()
        if missing_treatments > 0:
            log_info(f"  Trying partial string matching for {missing_treatments} remaining treatments...")
            
            # Create a mapping of treatments without metrics
            missing_std_treatments = viz_df.loc[viz_df[prefix + columns[0]].isna(), 'treatment_std'].unique()
            
            # Sample if too many
            if len(missing_std_treatments) > 100:
                log_info(f"  Sampling 100 treatments out of {len(missing_std_treatments)} for partial matching")
                missing_std_treatments = np.random.choice(missing_std_treatments, 100, replace=False)
            
            partial_matches = 0
            for treatment_std in missing_std_treatments:
                # Find metrics rows where this treatment is contained or contains the metrics treatment
                matching_metrics_rows = metrics_df[
                    (metrics_df['treatment_std'].str.contains(treatment_std, regex=False)) | 
                    (metrics_df['treatment_std'].apply(lambda x: treatment_std.find(x) >= 0))
                ]
                
                if len(matching_metrics_rows) > 0:
                    # Use the first match
                    metrics_row = matching_metrics_rows.iloc[0]
                    
                    # Apply to all rows with this treatment
                    mask = (viz_df['treatment_std'] == treatment_std) & (viz_df[prefix + columns[0]].isna())
                    if mask.any():
                        partial_matches += 1
                        
                        # Apply all metrics for this treatment
                        for col in columns:
                            if col in metrics_row:
                                viz_df.loc[mask, prefix + col] = metrics_row[col]
            
            log_info(f"  Applied {metrics_key} metrics to {partial_matches} treatments with partial string matching")
    
    # Now create unified columns - updated to include all metrics
    log_info("Creating unified metric columns")
    
    unified_columns = [
        ('cosine_distance_from_dmso', ['reference_cosine_distance_from_dmso', 'test_cosine_distance_from_dmso']),
        ('mad_cosine', ['reference_mad_cosine', 'test_mad_cosine']),
        ('var_cosine', ['reference_var_cosine', 'test_var_cosine']),
        ('std_cosine', ['reference_std_cosine', 'test_std_cosine']),
        ('median_distance', ['reference_median_distance', 'test_median_distance']),
        ('well_count', ['reference_well_count', 'test_well_count'])
    ]
    
    for unified_name, source_columns in unified_columns:
        # Only process if at least one source column exists
        available_sources = [col for col in source_columns if col in viz_df.columns]
        if not available_sources:
            log_info(f"Skipping unified column '{unified_name}' - no source columns available")
            continue
            
        # Fill from reference data first if available
        if source_columns[0] in viz_df.columns:
            viz_df[unified_name] = viz_df[source_columns[0]]
        
        # Then fill in gaps from test data if available
        if len(source_columns) > 1 and source_columns[1] in viz_df.columns:
            # Only update where the unified column is still null
            null_mask = viz_df[unified_name].isna()
            viz_df.loc[null_mask, unified_name] = viz_df.loc[null_mask, source_columns[1]]
        
        # Report on the new column
        non_nan_count = viz_df[unified_name].notna().sum()
        log_info(f"Created unified column '{unified_name}' with {non_nan_count} non-NaN values ({non_nan_count/len(viz_df)*100:.1f}%)")
    
    # NEW: Log information about landmark label columns
    log_info("LANDMARK LABEL COLUMNS SUMMARY:")
    for metric in metrics:
        metric_short = metric.split('_')[0]
        label_col = f'is_landmark_{metric_short}'
        cat_label_col = f'landmark_label_{metric_short}'
        
        if label_col in viz_df.columns:
            count = viz_df[label_col].sum()
            log_info(f"  {cat_label_col}: {count} landmarks ({count/len(viz_df)*100:.1f}%)")
            log_info(f"  Value counts: {viz_df[cat_label_col].value_counts().to_dict()}")
    
    # NEW SECTION: Merge scoring data into visualization file
    log_section("MERGING SCORING DATA INTO VISUALIZATION FILE")
    
    # Load reference and test scoring files if they exist
    reference_scores_path = dir_paths['analysis']['root'] / 'compound_reference_scores.csv'
    test_scores_path = dir_paths['analysis']['root'] / 'compound_test_scores.csv'
    
    scoring_dfs_to_merge = []
    
    if reference_scores_path.exists():
        try:
            reference_scores_df = pd.read_csv(reference_scores_path, low_memory=False)
            reference_scores_df['score_data_source'] = 'reference'
            scoring_dfs_to_merge.append(reference_scores_df)
            log_info(f"Loaded reference scores: {len(reference_scores_df)} treatments")
        except Exception as e:
            log_info(f"Error loading reference scores: {str(e)}")
    
    if test_scores_path.exists():
        try:
            test_scores_df = pd.read_csv(test_scores_path, low_memory=False)
            test_scores_df['score_data_source'] = 'test'
            scoring_dfs_to_merge.append(test_scores_df)
            log_info(f"Loaded test scores: {len(test_scores_df)} treatments")
        except Exception as e:
            log_info(f"Error loading test scores: {str(e)}")
    
    if scoring_dfs_to_merge:
        # Combine all scoring data
        combined_scores_df = pd.concat(scoring_dfs_to_merge, ignore_index=True)
        log_info(f"Combined scoring data: {len(combined_scores_df)} total treatments")
        
        # Handle duplicates using treatment + library composite key (prefer test over reference)
        combined_scores_df = combined_scores_df.sort_values('score_data_source', ascending=False)  # test comes first
        combined_scores_df = combined_scores_df.drop_duplicates(subset=['treatment', 'library'], keep='first')
        log_info(f"After removing duplicates by treatment+library: {len(combined_scores_df)} unique treatment-library combinations")
        
        # Define scoring columns to merge - keep landmark columns without prefix
        scoring_columns_to_merge = [
            # Distance metrics (with score_ prefix)
            'cosine_distance_from_dmso', 'mad_cosine', 'var_cosine', 'std_cosine', 
            'median_distance', 'well_count',
            # Landmark information (NO prefix)
            'closest_landmark', 'second_closest_landmark', 'third_closest_landmark',
            'closest_landmark_distance', 'second_closest_landmark_distance', 'third_closest_landmark_distance',
            'closest_is_landmark', 'second_closest_is_landmark', 'third_closest_is_landmark',
            'is_self_landmark', 'valid_for_phenotypic_makeup',
            # Landmark metadata - Basic (NO prefix)  
            'closest_landmark_SMILES', 'second_closest_landmark_SMILES', 'third_closest_landmark_SMILES',
            'closest_landmark_chemical_description', 'second_closest_landmark_chemical_description', 
            'third_closest_landmark_chemical_description',
            'closest_landmark_chemical_description_truncated_10', 'second_closest_landmark_chemical_description_truncated_10',
            'third_closest_landmark_chemical_description_truncated_10',
            'closest_landmark_PP_ID', 'second_closest_landmark_PP_ID', 'third_closest_landmark_PP_ID',
            'closest_landmark_PP_ID_uM', 'second_closest_landmark_PP_ID_uM', 'third_closest_landmark_PP_ID_uM',
            'closest_landmark_compound_name', 'second_closest_landmark_compound_name', 'third_closest_landmark_compound_name',
            'closest_landmark_compound_uM', 'second_closest_landmark_compound_uM', 'third_closest_landmark_compound_uM',
            # Landmark metadata - MOA columns (NO prefix)
            'closest_landmark_moa', 'second_closest_landmark_moa', 'third_closest_landmark_moa',
            'closest_landmark_moa_truncated_10', 'second_closest_landmark_moa_truncated_10', 'third_closest_landmark_moa_truncated_10',
            'closest_landmark_moa_first', 'second_closest_landmark_moa_first', 'third_closest_landmark_moa_first',
            'closest_landmark_moa_compound_uM', 'second_closest_landmark_moa_compound_uM', 'third_closest_landmark_moa_compound_uM',
            # Landmark metadata - Annotated target columns (NO prefix)
            'closest_landmark_annotated_target', 'second_closest_landmark_annotated_target', 'third_closest_landmark_annotated_target',
            'closest_landmark_annotated_target_description', 'second_closest_landmark_annotated_target_description', 'third_closest_landmark_annotated_target_description',
            'closest_landmark_annotated_target_description_truncated_10', 'second_closest_landmark_annotated_target_description_truncated_10', 'third_closest_landmark_annotated_target_description_truncated_10',
            # Landmark metadata - Additional supplier/control columns (NO prefix)
            'closest_landmark_supplier_ID', 'second_closest_landmark_supplier_ID', 'third_closest_landmark_supplier_ID',
            'closest_landmark_control_type', 'second_closest_landmark_control_type', 'third_closest_landmark_control_type',
            'closest_landmark_control_name', 'second_closest_landmark_control_name', 'third_closest_landmark_control_name',
            'closest_landmark_is_control', 'second_closest_landmark_is_control', 'third_closest_landmark_is_control',
            'closest_landmark_lib_plate_order', 'second_closest_landmark_lib_plate_order', 'third_closest_landmark_lib_plate_order',
            'closest_landmark_perturbation_name', 'second_closest_landmark_perturbation_name', 'third_closest_landmark_perturbation_name',
            'closest_landmark_chemical_name', 'second_closest_landmark_chemical_name', 'third_closest_landmark_chemical_name',
            'closest_landmark_manual_annotation', 'second_closest_landmark_manual_annotation', 'third_closest_landmark_manual_annotation',
            'closest_landmark_compound_type', 'second_closest_landmark_compound_type', 'third_closest_landmark_compound_type',
            # Normalized metrics for scoring (WITH prefix)
            'normalized_dmso_distance', 'normalized_landmark_similarity',
            'normalized_mad_cosine', 'normalized_var_cosine', 'normalized_std_cosine',
            # Score metrics (WITH prefix)
            'harmonic_mean_2term', 'harmonic_mean_3term',
            'harmonic_mean_2term_mad_cosine', 'harmonic_mean_3term_mad_cosine',
            'harmonic_mean_2term_var_cosine', 'harmonic_mean_3term_var_cosine',
            'harmonic_mean_2term_std_cosine', 'harmonic_mean_3term_std_cosine',
            'ratio_score',
            # Source tracking (NO prefix)
            'score_data_source'
        ]
        
        # Filter to columns that actually exist in the combined scoring data
        available_scoring_cols = [col for col in scoring_columns_to_merge if col in combined_scores_df.columns]
        log_info(f"Found {len(available_scoring_cols)} scoring columns to merge")
        
        # Create the column mapping - landmark columns keep original names, others get score_ prefix
        scoring_columns_with_prefix = {}
        landmark_keywords = ['landmark', 'makeup_', 'is_self_landmark', 'valid_for_phenotypic', 'score_data_source']
        
        for col in available_scoring_cols:
            if col in ['treatment', 'library']:
                # Keep merge keys as-is
                scoring_columns_with_prefix[col] = col
            elif any(keyword in col for keyword in landmark_keywords):
                # Keep landmark-related columns without prefix
                scoring_columns_with_prefix[col] = col
            else:
                # Add score_ prefix to metric columns
                scoring_columns_with_prefix[col] = f'score_{col}'
         
        # Two changes - replacing commented out section above
                # 1. Added .copy() when creating scoring_data_to_merge
                # 2. .loc[:, 'treatment_library_key']

        # Rename columns in scoring dataframe
        combined_scores_renamed = combined_scores_df.rename(columns=scoring_columns_with_prefix)
        
        # Select only the columns we want to merge (must include both treatment and library)
        merge_columns = ['treatment', 'library'] + [scoring_columns_with_prefix[col] for col in available_scoring_cols if col not in ['treatment', 'library']]
        scoring_data_to_merge = combined_scores_renamed[merge_columns].copy()  # ← Added .copy() here
        
        log_info(f"Prepared {len(scoring_data_to_merge)} scoring records for merge using treatment+library keys")
        
        # With this:
        viz_df.loc[:, 'treatment_library_key'] = viz_df['treatment'].astype(str) + '|' + viz_df['library'].astype(str)
        scoring_data_to_merge.loc[:, 'treatment_library_key'] = scoring_data_to_merge['treatment'].astype(str) + '|' + scoring_data_to_merge['library'].astype(str)  # ← Added .loc[:, ]
        
        viz_keys = set(viz_df['treatment_library_key'].unique())
        scoring_keys = set(scoring_data_to_merge['treatment_library_key'].unique())
        missing_in_scoring = viz_keys - scoring_keys
        
        log_info(f"Treatment+Library combinations in viz data: {len(viz_keys)}")
        log_info(f"Treatment+Library combinations in scoring data: {len(scoring_keys)}")
        log_info(f"Combinations in viz but missing from scoring: {len(missing_in_scoring)}")
        
        if len(missing_in_scoring) < 20:
            log_info(f"Sample missing combinations: {list(missing_in_scoring)[:10]}")
        
        # Clean up diagnostic columns BEFORE merge
        # Fixed code to avoid the warning:
        viz_df = viz_df.drop(columns=['treatment_library_key']).copy()
        scoring_data_to_merge = scoring_data_to_merge.drop(columns=['treatment_library_key']).copy()

        # ADD THE DEBUGGING CODE HERE (before the merge):
        # ============ START DEBUGGING CODE ============
        log_info("=" * 80)
        log_info("DEBUG: Detailed merge key analysis before scoring merge")
        log_info("=" * 80)

        # Check viz_df details
        log_info("\n--- VISUALIZATION DATA ---")
        log_info(f"Total rows in viz_df: {len(viz_df)}")
        log_info(f"Unique treatments in viz_df: {viz_df['treatment'].nunique()}")

        if 'library' in viz_df.columns:
            viz_library_values = viz_df['library'].value_counts(dropna=False)
            log_info(f"Library value counts in viz_df:")
            for lib, count in viz_library_values.head(10).items():
                log_info(f"  '{lib}': {count} rows")
            if viz_df['library'].isna().any():
                log_info(f"  NaN/missing: {viz_df['library'].isna().sum()} rows")
        else:
            log_info("WARNING: No 'library' column in viz_df!")

        log_info(f"\nSample viz_df rows (first 5):")
        sample_cols = ['treatment', 'library'] if 'library' in viz_df.columns else ['treatment']
        log_info(viz_df[sample_cols].head().to_string())

        # Check scoring_data_to_merge details
        log_info("\n--- SCORING DATA ---")
        log_info(f"Total rows in scoring_data_to_merge: {len(scoring_data_to_merge)}")
        log_info(f"Unique treatments in scoring_data_to_merge: {scoring_data_to_merge['treatment'].nunique()}")

        if 'library' in scoring_data_to_merge.columns:
            score_library_values = scoring_data_to_merge['library'].value_counts(dropna=False)
            log_info(f"Library value counts in scoring_data_to_merge:")
            for lib, count in score_library_values.head(10).items():
                log_info(f"  '{lib}': {count} rows")
            if scoring_data_to_merge['library'].isna().any():
                log_info(f"  NaN/missing: {scoring_data_to_merge['library'].isna().sum()} rows")
        else:
            log_info("WARNING: No 'library' column in scoring_data_to_merge!")

        log_info(f"\nSample scoring_data_to_merge rows (first 5):")
        log_info(scoring_data_to_merge[sample_cols].head().to_string())

        # Check for exact matches using composite keys
        log_info("\n--- MERGE KEY MATCHING ANALYSIS ---")
        if 'library' in viz_df.columns and 'library' in scoring_data_to_merge.columns:
            # Create composite keys for comparison
            viz_keys = set(zip(viz_df['treatment'].astype(str), viz_df['library'].astype(str)))
            score_keys = set(zip(scoring_data_to_merge['treatment'].astype(str), scoring_data_to_merge['library'].astype(str)))
            
            matching_keys = viz_keys & score_keys
            viz_only_keys = viz_keys - score_keys
            score_only_keys = score_keys - viz_keys
            
            log_info(f"Total unique treatment+library combinations in viz_df: {len(viz_keys)}")
            log_info(f"Total unique treatment+library combinations in scoring: {len(score_keys)}")
            log_info(f"Matching treatment+library combinations: {len(matching_keys)} ({len(matching_keys)/len(viz_keys)*100:.1f}% of viz keys)")
            log_info(f"Combinations only in viz_df (will have NaN scores): {len(viz_only_keys)}")
            log_info(f"Combinations only in scoring (won't be used): {len(score_only_keys)}")
            
            # Show samples of non-matching keys
            if viz_only_keys and len(viz_only_keys) < 20:
                log_info(f"\nSample keys in viz but not in scoring (first 10):")
                for i, (treatment, library) in enumerate(list(viz_only_keys)[:10]):
                    log_info(f"  {i+1}. treatment='{treatment}', library='{library}'")
        else:
            log_info("Cannot perform composite key analysis - library column missing")
            
            # Fall back to treatment-only analysis
            viz_treatments = set(viz_df['treatment'].astype(str))
            score_treatments = set(scoring_data_to_merge['treatment'].astype(str))
            
            matching_treatments = viz_treatments & score_treatments
            log_info(f"Treatment-only matching: {len(matching_treatments)}/{len(viz_treatments)} ({len(matching_treatments)/len(viz_treatments)*100:.1f}%)")

        log_info("=" * 80)
        log_info("END DEBUG OUTPUT")
        log_info("=" * 80)
        
        # Perform the merge using treatment only (compounds are the same across libraries)
        log_info("Broadcasting treatment-level scores to well-level visualization data...")
        log_info("Note: Matching on treatment only - same compound gets same landmarks regardless of library")
        original_viz_shape = viz_df.shape

        # Drop library from scoring data to avoid conflicts, and deduplicate by treatment
        scoring_cols_to_merge = [col for col in scoring_data_to_merge.columns if col != 'library']
        scoring_for_merge = scoring_data_to_merge[scoring_cols_to_merge].drop_duplicates(subset=['treatment'])

        log_info(f"Deduplicating scoring data: {len(scoring_data_to_merge)} -> {len(scoring_for_merge)} unique treatments")

        # ============================================================================
        # CRITICAL FIX: Remove duplicate is_self_landmark column before merge
        # ============================================================================
        # Scoring data brings its own is_self_landmark, but viz_df may already have one
        # from main.py. Drop viz_df's version to avoid _x/_y suffixes.
        log_info("Checking for duplicate is_self_landmark column...")
        
        if 'is_self_landmark' in viz_df.columns and 'is_self_landmark' in scoring_for_merge.columns:
            log_info("  Duplicate is_self_landmark detected - dropping version from viz_df")
            log_info(f"  Before drop: viz_df has {viz_df['is_self_landmark'].sum()} landmarks")
            viz_df = viz_df.drop(columns=['is_self_landmark']).copy()
            log_info("  ✓ Dropped is_self_landmark from viz_df (will use version from scoring data)")
        elif 'is_self_landmark' in viz_df.columns:
            log_info("  Only viz_df has is_self_landmark - keeping it")
        elif 'is_self_landmark' in scoring_for_merge.columns:
            log_info("  Only scoring data has is_self_landmark - no conflict")
        else:
            log_info("  WARNING: Neither viz_df nor scoring data has is_self_landmark!")
        # ============================================================================

        viz_df = viz_df.merge(
            scoring_for_merge,
            on=['treatment'],
            how='left'
        )
        
        log_info(f"Merge completed: {original_viz_shape} -> {viz_df.shape}")
        
        # Verify landmark broadcasting success
        key_landmark_cols = ['closest_landmark', 'second_closest_landmark', 'third_closest_landmark']
        for col in key_landmark_cols:
            if col in viz_df.columns:
                matches = viz_df[col].notna().sum()
                log_info(f"Wells with {col}: {matches}/{len(viz_df)} ({matches/len(viz_df)*100:.1f}%)")
        
        # Overall success metric
        if 'closest_landmark' in viz_df.columns:
            success_rate = viz_df['closest_landmark'].notna().sum() / len(viz_df) * 100
            log_info(f"Overall landmark broadcasting success: {success_rate:.1f}%")
        
        # Log breakdown by score_data_source
        if 'score_data_source' in viz_df.columns:
            source_counts = viz_df['score_data_source'].value_counts()
            log_info("Scoring data source breakdown:")
            for source, count in source_counts.items():
                log_info(f"  {source}: {count} wells")
    
    else:
        log_info("No scoring files found. Visualization data will not include scoring metrics.")
        viz_df['score_data_source'] = None

   # Final metrics summary
    log_info("FINAL METRICS SUMMARY:")
    for col in viz_df.columns:
        if any(metric in col for metric in ['cosine_distance', 'mad_cosine', 'var_cosine', 'std_cosine', 'median_distance', 'well_count']):
            non_nan = viz_df[col].notna().sum()
            log_info(f"  {col}: {non_nan} non-NaN values ({non_nan/len(viz_df)*100:.1f}%)")

    # REMOVED: visualization_data.csv is redundant with spc_for_viz_app.csv
    log_info("Skipping visualization_data.csv - redundant with spc_for_viz_app.csv")

    # Create treatment-level aggregated data (BEFORE CELLPROFILER ADDED)
    log_info("Creating treatment-level aggregated data...")
    viz_df_agg = aggregate_to_treatment_level(viz_df)

    # Save treatment-level aggregated data
    agg_output_path = dir_paths['data'] / 'visualization_data_treatment_agg.csv'
    viz_df_agg.to_csv(agg_output_path, index=False)
    log_info(f"Saved treatment-level aggregated data to: {agg_output_path}")
    log_info(f"  Contains {len([c for c in viz_df_agg.columns if c.startswith('Z')])} embedding columns")

    # Create similarity matrix for hierarchical clustering
    log_info("Creating similarity matrix for hierarchical clustering...")
    from .hierarchical_clustering import create_similarity_matrix
    create_similarity_matrix(viz_df_agg, dir_paths['visualizations']['hierarchical_clustering']['cluster_map'])
     
    # ========================================================================
    # DEDUPLICATION FIX - Remove duplicate wells before saving
    # ========================================================================
    log_info("\n" + "="*80)
    log_info("CHECKING FOR DUPLICATE WELLS")
    log_info("="*80)
    
    # Create unique well identifier
    viz_df['temp_well_id'] = viz_df['plate'].astype(str) + '_' + viz_df['well'].astype(str)
    
    # Check for duplicates
    n_total = len(viz_df)
    n_dups = viz_df['temp_well_id'].duplicated().sum()
    
    if n_dups > 0:
        log_info(f"⚠️  WARNING: Found {n_dups} duplicate well IDs ({n_dups/n_total*100:.2f}%)")
        log_info("Analyzing duplicates...")
        
        # Show example duplicates for debugging
        dup_wells = viz_df[viz_df['temp_well_id'].duplicated(keep=False)]['temp_well_id'].unique()
        log_info(f"Example duplicate wells: {list(dup_wells[:5])}")
        
        # Count duplicates per well
        dup_counts = viz_df[viz_df['temp_well_id'].duplicated(keep=False)].groupby('temp_well_id').size()
        log_info(f"Duplicate counts - Min: {dup_counts.min()}, Max: {dup_counts.max()}, Mean: {dup_counts.mean():.1f}")
        
        # Remove duplicates (keep first occurrence)
        viz_df = viz_df[~viz_df['temp_well_id'].duplicated(keep='first')].copy()
        log_info(f"✓ Removed {n_dups} duplicates")
        log_info(f"After deduplication: {len(viz_df)} unique wells remaining")
    else:
        log_info("✓ No duplicates found - data is clean")
    
    # Remove temporary column
    viz_df = viz_df.drop(columns=['temp_well_id'])
    log_info("="*80 + "\n")
    
    # ========================================================================
    # END DEDUPLICATION FIX
    # ========================================================================

    # Also save a smaller version with just the key columns for easier inspection
    key_columns = [
        # Core SPC columns for checking merge
        'treatment', 'compound_name', 'compound_uM', 
        'moa', 'moa_truncated_10', 'moa_first', 'moa_compound_uM',  # Keep MOA derivatives
        'annotated_target_description', 'annotated_target_description_truncated_10',  # Primary target columns (no duplicates)
        'plate', 'well', 'well_row', 'well_column', 'library', 'cell_count', 'cell_pct',
        'landmark_label', 'is_landmark', 'PP_ID', 'PP_ID_uM', 'gene_description',
        'SMILES', 'chemical_description',
        'landmark_label_mad', 'is_landmark_mad',
        'landmark_label_std', 'is_landmark_std',
        'landmark_label_var', 'is_landmark_var',
        
        # SPC Metrics
        'cosine_distance_from_dmso', 'mad_cosine', 'var_cosine', 'std_cosine', 
        'median_distance', 'well_count',
        'reference_cosine_distance_from_dmso', 'reference_mad_cosine', 'reference_var_cosine', 'reference_std_cosine',
        'reference_median_distance', 'reference_well_count',
        'test_cosine_distance_from_dmso', 'test_mad_cosine', 'test_var_cosine', 'test_std_cosine',
        'test_median_distance', 'test_well_count',

        # Scoring columns
        'score_data_source', 'score_cosine_distance_from_dmso', 'score_closest_landmark_distance', 
        'score_closest_landmark', 'score_second_closest_landmark', 'score_third_closest_landmark',
        'score_is_self_landmark', 'score_closest_is_landmark',
        'score_harmonic_mean_2term', 'score_harmonic_mean_3term',
        'score_harmonic_mean_2term_mad_cosine', 'score_harmonic_mean_3term_mad_cosine',
        'score_harmonic_mean_2term_var_cosine', 'score_harmonic_mean_3term_var_cosine',
        'score_harmonic_mean_2term_std_cosine', 'score_harmonic_mean_3term_std_cosine',

        # COMPLETE landmark columns (for all three positions) - UPDATED with full metadata
        'closest_landmark', 'closest_landmark_distance', 'closest_is_landmark',
        'closest_landmark_moa', 'closest_landmark_moa_truncated_10', 'closest_landmark_moa_first', 'closest_landmark_moa_compound_uM',
        'closest_landmark_annotated_target_description', 'closest_landmark_annotated_target_description_truncated_10',
        'closest_landmark_SMILES', 'closest_landmark_chemical_description', 'closest_landmark_chemical_description_truncated_10',
        'closest_landmark_PP_ID', 'closest_landmark_PP_ID_uM', 'closest_landmark_compound_name', 'closest_landmark_compound_uM',
        'closest_landmark_supplier_ID', 'closest_landmark_control_type', 'closest_landmark_control_name', 'closest_landmark_is_control',
        
        # Second closest (same structure)
        'second_closest_landmark', 'second_closest_landmark_distance', 'second_closest_is_landmark',
        'second_closest_landmark_moa', 'second_closest_landmark_moa_truncated_10', 'second_closest_landmark_moa_first', 'second_closest_landmark_moa_compound_uM',
        'second_closest_landmark_annotated_target_description', 'second_closest_landmark_annotated_target_description_truncated_10',
        'second_closest_landmark_SMILES', 'second_closest_landmark_chemical_description', 'second_closest_landmark_chemical_description_truncated_10',
        'second_closest_landmark_PP_ID', 'second_closest_landmark_PP_ID_uM', 'second_closest_landmark_compound_name', 'second_closest_landmark_compound_uM',
        
        # Third closest (same structure)  
        'third_closest_landmark', 'third_closest_landmark_distance', 'third_closest_is_landmark',
        'third_closest_landmark_moa', 'third_closest_landmark_moa_truncated_10', 'third_closest_landmark_moa_first', 'third_closest_landmark_moa_compound_uM',
        'third_closest_landmark_annotated_target_description', 'third_closest_landmark_annotated_target_description_truncated_10',
        'third_closest_landmark_SMILES', 'third_closest_landmark_chemical_description', 'third_closest_landmark_chemical_description_truncated_10',
        'third_closest_landmark_PP_ID', 'third_closest_landmark_PP_ID_uM', 'third_closest_landmark_compound_name', 'third_closest_landmark_compound_uM',

        # CellProfiler metadata columns
        'cellprofiler_Metadata_plate_barcode', 'cellprofiler_Metadata_well', 'cellprofiler_Metadata_field',
        'cellprofiler_Metadata_lib_plate_order', 'cellprofiler_Metadata_replicate', 'cellprofiler_Metadata_compound_uM',
        'cellprofiler_Metadata_cell_type', 'cellprofiler_Metadata_library', 'cellprofiler_Metadata_perturbation_name',
        'cellprofiler_Metadata_chemical_name', 'cellprofiler_Metadata_supplier_ID', 'cellprofiler_Metadata_control_type',
        'cellprofiler_Metadata_control_name', 'cellprofiler_Metadata_is_control', 'cellprofiler_Metadata_library_from_csv',
        'cellprofiler_Metadata_annotated_target', 'cellprofiler_Metadata_annotated_target_truncated',
        'cellprofiler_Metadata_annotated_target_description', 'cellprofiler_Metadata_annotated_target_description_truncated',
        'cellprofiler_Metadata_PP_ID', 'cellprofiler_Metadata_SMILES', 'cellprofiler_Metadata_chemical_description'
    ]
    
    # Filter to columns that actually exist
    key_columns = [col for col in key_columns if col in viz_df.columns]
    
    # Remove embedding columns from small output
    embedding_cols = [col for col in viz_df.columns if col.startswith('Z') and col[1:].isdigit()]
    small_df = viz_df[[col for col in key_columns if col not in embedding_cols]]
    
    small_output_path = dir_paths['data'] / 'visualization_data_small.csv'
    small_df.to_csv(small_output_path, index=False)
    log_info(f"Saved smaller visualization data to: {small_output_path}")
    log_info(f"  Excluded {len(embedding_cols)} embedding columns")

    # REMOVED: CellProfiler merge is deprecated (see lines 1048-1050)
    log_info("CellProfiler data is now available in spc_for_viz_app.csv")


    # ========================================================================
    # HIERARCHICAL CHUNK CLUSTERING - MOVED TO main.py
    # ========================================================================
    # NOTE: Hierarchical clustering is now called from main.py AFTER spc_for_viz_app.csv is saved
    # This fixes the "Metadata not found" error caused by timing issues
    
    # Return both viz_df and viz_df_agg so main.py can use viz_df_agg for clustering
    return viz_df, viz_df_agg

