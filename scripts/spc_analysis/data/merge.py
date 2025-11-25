# Merge_datasets

import pandas as pd
import numpy as np
from ..utils.logging import log_section, log_info

# Complete updated merge_datasets function

def merge_datasets(config, dir_paths):
    """
    Merge embeddings, metadata, and harmony data.
    
    UPDATED: 
    - Conditional DMSO library assignment (only when plate has non-DMSO context)
    - No creation of duplicate target_description columns
    - Preserves annotated_target_description as primary target column
    
    Args:
        config: Configuration dictionary with file paths
        dir_paths: Directory paths dictionary
        
    Returns:
        DataFrame: Merged data with proper column naming
    """
    log_section("STEP 1: MERGING DATASETS")
    
    # Load metadata file
    metadata_file = config.get('metadata_file')
    log_info(f"Loading metadata from: {metadata_file}")
    metadata_df = pd.read_csv(metadata_file)
    log_info(f"Loaded metadata with shape: {metadata_df.shape}")
    log_info(f"Metadata columns: {metadata_df.columns.tolist()}")
    
    # Load embeddings file
    embeddings_file = config['embeddings_file']
    log_info(f"Loading embeddings from: {embeddings_file}")
    embeddings_array = np.load(embeddings_file)
    log_info(f"Loaded NPY embeddings with shape: {embeddings_array.shape}")
    
    # Detect embedding dimensionality
    embedding_dim = embeddings_array.shape[1]
    log_info(f"Detected embedding dimensionality: {embedding_dim}")
    
    # Create column names for embeddings
    embedding_cols = [f"Z{i+1}" for i in range(embedding_dim)]
    embeddings_df = pd.DataFrame(embeddings_array, columns=embedding_cols)
    
    # Ensure metadata and embeddings have the same number of rows
    if len(metadata_df) != len(embeddings_df):
        raise ValueError(f"Metadata has {len(metadata_df)} rows but embeddings has {len(embeddings_df)} rows")
    
    # Concatenate metadata with embeddings
    merged_df = pd.concat([metadata_df, embeddings_df], axis=1)
    log_info(f"Combined metadata and embeddings with final shape: {merged_df.shape}")
    
    # Load harmony data (for cell counts)
    if 'harmony_file' in config and config['harmony_file']:
        log_info(f"Loading harmony data from: {config['harmony_file']}")
        harmony_df = pd.read_csv(config['harmony_file'])
        log_info(f"Loaded harmony data with shape: {harmony_df.shape}")
        
        # Rename harmony columns to match our expected format (if needed)
        harmony_column_mapping = config.get('harmony_column_mapping', {
            'Metadata_Plate_ID': 'plate',
            'Metadata_Well_ID': 'well',
            'Nuclei Selected - Number of Objects': 'cell_count'
        })

        harmony_df = harmony_df.rename(columns=harmony_column_mapping)
        log_info(f"Renamed harmony columns. New columns include: {list(harmony_df.columns)}")
        
        # Standardize well format if needed
        if 'well_format' in config and config['well_format'] == 'letter_number':
            # For harmony data
            if 'well' in harmony_df.columns:
                sample_well = harmony_df['well'].iloc[0] if not harmony_df.empty else ""
                if sample_well and len(sample_well) == 2:  # A1 format
                    log_info("Converting wells from 'A1' to 'A01' format in harmony data")
                    harmony_df['well'] = harmony_df['well'].apply(
                        lambda x: x[0] + x[1:].zfill(2) if isinstance(x, str) and len(x) >= 2 else x
                    )
            
            # For merged_df data
            if 'well' in merged_df.columns:
                sample_well = merged_df['well'].iloc[0] if not merged_df.empty else ""
                if sample_well and len(sample_well) == 2:  # A1 format
                    log_info("Converting wells from 'A1' to 'A01' format in embeddings data")
                    merged_df['well'] = merged_df['well'].apply(
                        lambda x: x[0] + x[1:].zfill(2) if isinstance(x, str) and len(x) >= 2 else x
                    )
        
        # Merge harmony data with embeddings
        merge_keys = ['plate', 'well']
        log_info(f"Merging on keys: {merge_keys}")
        
        # Check if merge keys exist in both dataframes
        if all(key in merged_df.columns for key in merge_keys) and all(key in harmony_df.columns for key in merge_keys):
            # Only merge cell_count from harmony file (other metadata comes from PP file)
            harmony_columns_to_merge = merge_keys + ['cell_count']
            
            # Perform the merge with only the cell_count column
            merged_df = merged_df.merge(
                harmony_df[harmony_columns_to_merge], 
                on=merge_keys,
                how='left'
            )
            log_info(f"Merged with harmony data. New shape: {merged_df.shape}")
            log_info(f"Unique plate values in data: {sorted(merged_df['plate'].unique())[:10]}...")
        else:
            missing_keys_df = [key for key in merge_keys if key not in merged_df.columns]
            missing_keys_harmony = [key for key in merge_keys if key not in harmony_df.columns]
            
            log_info(f"Warning: Cannot merge with harmony data. Missing keys in data: {missing_keys_df}")
            log_info(f"Missing keys in harmony data: {missing_keys_harmony}")
    else:
        log_info("No harmony file specified. Skipping harmony data merge.")

    # Add library information from plate definitions
    if 'plate_definitions' in config:
        log_info("Adding library information from plate definitions...")
        log_info(f"Total plates defined in config: {len(config['plate_definitions'])}")
        
        # Convert plate column to string for matching
        merged_df['plate_str'] = merged_df['plate'].astype(str)
        log_info(f"Sample plate values in data: {sorted(merged_df['plate_str'].unique())[:10]}")
        
        # Initialize library column
        merged_df['library'] = 'Unknown'
        
        # Track assignments
        total_expected_assignments = 0
        
        # Map plates to libraries - UNCONDITIONALLY
        for plate, attributes in config['plate_definitions'].items():
            plate_mask = merged_df['plate_str'] == plate
            matches = plate_mask.sum()
            
            if matches > 0:
                total_expected_assignments += matches
                if 'library' in attributes:
                    library_name = attributes['library']
                    merged_df.loc[plate_mask, 'library'] = library_name
                    log_info(f"Assigned {matches} rows to library '{library_name}' for plate {plate}")
                    
                    # Count DMSO vs non-DMSO for logging only
                    dmso_count = merged_df[plate_mask & merged_df['treatment'].str.startswith('DMSO', na=False)].shape[0]
                    if dmso_count > 0:
                        log_info(f"  Including {dmso_count} DMSO treatments")
            else:
                log_info(f"WARNING: Plate {plate} defined in config but found 0 matching rows in data")
        
        # Remove temporary column
        merged_df = merged_df.drop(columns=['plate_str'])
        
        log_info(f"Total rows assigned to libraries: {total_expected_assignments}")
        
        # Log library distribution
        library_counts = merged_df['library'].value_counts(dropna=False)
        log_info(f"Final library distribution: {library_counts.to_dict()}")

    # Load PP numbers and metadata (NO target_description creation - keep annotated_target_description)
    if config and 'pp_numbers_file' in config and config['pp_numbers_file']:
        pp_file_path = config['pp_numbers_file']
        log_info(f"Loading PP numbers from: {pp_file_path}")
        
        try:
            pp_df = pd.read_csv(pp_file_path)
            log_info(f"Loaded PP numbers file with {len(pp_df)} rows")
            log_info(f"PP file columns: {pp_df.columns.tolist()}")
            
            # Define ALL metadata columns to preserve (NO target_description creation)
            metadata_columns_to_preserve = [
                'Metadata_lib_plate_order', 
                'Metadata_perturbation_name',
                'Metadata_chemical_name', 
                'Metadata_supplier_ID', 
                'Metadata_control_type',
                'Metadata_control_name', 
                'Metadata_is_control', 
                'Metadata_library',
                'Metadata_annotated_target', 
                'Metadata_annotated_target_description',
                'Metadata_PP_ID', 
                'Metadata_SMILES', 
                'Metadata_chemical_description',
                'Metadata_compound_type',
                'Metadata_manual_annotation',
            ]
            
            # Check which columns actually exist
            available_metadata = [col for col in metadata_columns_to_preserve if col in pp_df.columns]
            missing_metadata = [col for col in metadata_columns_to_preserve if col not in pp_df.columns]
            
            log_info(f"Found {len(available_metadata)} metadata columns to preserve")
            if missing_metadata:
                log_info(f"Missing metadata columns: {missing_metadata}")
            
            # Clean the matching columns
            if 'Metadata_perturbation_name' in pp_df.columns:
                pp_df['chemical_name_clean'] = pp_df['Metadata_perturbation_name'].astype(str).str.strip()
            if 'Metadata_library' in pp_df.columns:
                pp_df['library_clean'] = pp_df['Metadata_library'].astype(str).str.strip()

            merged_df['compound_name_clean'] = merged_df['compound_name'].astype(str).str.strip()

            # Handle library column - it might not exist yet
            if 'library' in merged_df.columns:
                merged_df['library_clean'] = merged_df['library'].astype(str).str.strip()
            else:
                merged_df['library_clean'] = 'Unknown'

            # Standardize well format in PP file to match merged_df
            if 'Metadata_well' in pp_df.columns:
                if 'well_format' in config and config['well_format'] == 'letter_number':
                    sample_well = pp_df['Metadata_well'].iloc[0] if not pp_df.empty else ""
                    if sample_well and len(str(sample_well)) == 2:  # A1 format
                        log_info("Converting wells from 'A1' to 'A01' format in PP file")
                        pp_df['Metadata_well'] = pp_df['Metadata_well'].apply(
                            lambda x: x[0] + x[1:].zfill(2) if isinstance(x, str) and len(x) >= 2 else x
                        )

            ###### CRITICAL FIX #######
                    
            # Must merge on:
                # compound_name
                # plate
                # well
            # As same compound_name can appear > 1 time: across differnet libraries, library plates, multiple wells/plate (e.g. JUMP)    
            
            # Create subset with only needed columns - ADD Metadata_well 
            pp_columns_to_merge = ['chemical_name_clean', 'library_clean', 'Metadata_well'] + available_metadata

            # Only keep rows with valid PP_ID if that column exists
            if 'Metadata_PP_ID' in pp_df.columns:
                pp_mapping = pp_df[pp_df['Metadata_PP_ID'].notna()][pp_columns_to_merge].drop_duplicates()
            else:
                pp_mapping = pp_df[pp_columns_to_merge].drop_duplicates()
                
            log_info(f"Found {len(pp_mapping)} unique compound-library-well-metadata mappings")

            # Merge PP data - ADD well to merge keys
            log_info("Merging PP metadata using compound name + library + well matching...")
            merged_df = merged_df.merge(
                pp_mapping, 
                left_on=['compound_name_clean', 'library_clean', 'well'], 
                right_on=['chemical_name_clean', 'library_clean', 'Metadata_well'], 
                how='left'
            )
            
            # Rename columns to remove Metadata_ prefix for cleaner names
            # IMPORTANT: Keep annotated_target_description as-is (primary target column)
            for col in available_metadata:
                clean_name = col.replace('Metadata_', '')
                if col in merged_df.columns:
                    merged_df[clean_name] = merged_df[col]
                    merged_df = merged_df.drop(columns=[col])
                    if clean_name == 'annotated_target_description':
                        log_info(f"Added PRIMARY target column: {clean_name} (no duplicates)")
                    else:
                        log_info(f"Added column: {clean_name}")
            
            # Clean up temporary columns
            columns_to_drop = ['compound_name_clean', 'library_clean', 'chemical_name_clean']
            merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])
            
            # Log success rate for key columns
            key_metadata_cols = ['PP_ID', 'SMILES', 'chemical_description', 'annotated_target', 
                                 'annotated_target_description', 'supplier_ID', 'control_type', 'compound_type']
            for col in key_metadata_cols:
                if col in merged_df.columns:
                    matches = merged_df[col].notna().sum()
                    log_info(f"{col}: {matches}/{len(merged_df)} matches ({matches/len(merged_df)*100:.1f}%)")
                    
        except Exception as e:
            log_info(f"Error loading PP numbers: {str(e)}")
            import traceback
            log_info(traceback.format_exc())
    else:
        log_info("No PP numbers file specified. Skipping PP metadata merge.")

    
    # Fill DMSO rows with library from same plate using groupby
    log_info("Filling DMSO library assignments using plate-level propagation...")
    # For each plate, propagate non-null library values to DMSO rows
    def fill_dmso_library(plate_group):
        # Get the non-DMSO library value(s) for this plate
        non_dmso_mask = ~plate_group['treatment'].str.startswith('DMSO', na=False)
        non_dmso_libraries = plate_group.loc[non_dmso_mask, 'library'].dropna().unique()
        
        if len(non_dmso_libraries) == 1:
            # Fill all DMSO rows with this library
            dmso_mask = plate_group['treatment'].str.startswith('DMSO', na=False)
            plate_group.loc[dmso_mask, 'library'] = non_dmso_libraries[0]
        elif len(non_dmso_libraries) > 1:
            log_info(f"Warning: Plate {plate_group['plate'].iloc[0]} has multiple libraries: {non_dmso_libraries}")
        
        return plate_group
    
    # Apply the function to each plate group
    merged_df = merged_df.groupby('plate', group_keys=False).apply(fill_dmso_library)
    
    # Log results
    dmso_data = merged_df[merged_df['treatment'].str.startswith('DMSO', na=False)]
    if len(dmso_data) > 0:
        dmso_with_library = dmso_data['library'].notna().sum()
        log_info(f"DMSO library assignment complete: {dmso_with_library}/{len(dmso_data)} DMSO treatments have library assignments")

    # Find embedding columns
    embedding_cols = [col for col in merged_df.columns if col.startswith('Z') and col[1:].isdigit()]
    log_info(f"Found {len(embedding_cols)} embedding dimensions: {embedding_cols[:5]}...")
    
    # Ensure all required columns exist
    required_columns = ['cell_type', 'plate', 'well', 'treatment', 'compound_name', 'compound_uM']
    missing_columns = [col for col in required_columns if col not in merged_df.columns]
    
    if missing_columns:
        log_info(f"Warning: Missing required columns: {missing_columns}")
        
        # Try to fix missing columns if possible
        for col in missing_columns:
            if col == 'compound_uM' and 'concentration' in merged_df.columns:
                log_info("Using 'concentration' column for 'compound_uM'")
                merged_df['compound_uM'] = merged_df['concentration']
            elif col == 'treatment' and 'compound_name' in merged_df.columns:
                log_info("Using 'compound_name' column for 'treatment'")
                merged_df['treatment'] = merged_df['compound_name']
            else:
                log_info(f"Creating empty column for '{col}'")
                merged_df[col] = "Unknown"

    # VERIFICATION: Ensure no duplicate target_description columns were created
    target_columns = [col for col in merged_df.columns if 'target_description' in col]
    log_info(f"Target-related columns in final merged data: {target_columns}")
    
    if 'target_description' in target_columns:
        log_info("WARNING: Duplicate target_description column detected!")
    if 'annotated_target_description' in target_columns:
        log_info("SUCCESS: Primary annotated_target_description column present")

    # Save the complete merged dataset
    if 'output_dir' in config:
        output_path = dir_paths['data'] / 'complete_merged_data.csv'
        merged_df.to_csv(output_path, index=False)
        log_info(f"Saved complete merged data to: {output_path}")
        
        # Save column list for reference
        column_list_path = dir_paths['data'] / 'column_list.txt'
        with open(column_list_path, 'w') as f:
            f.write("Column names in merged dataset:\n")
            for i, col in enumerate(merged_df.columns):
                f.write(f"{i+1}. {col}\n")
        log_info(f"Saved column list to: {column_list_path}")
    
    return merged_df