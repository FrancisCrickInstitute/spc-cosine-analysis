import os
import gc
import time
import json
import numpy as np
import pandas as pd
import warnings
import traceback
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import local modules
from spc_analysis.utils.logging import log_info, log_section, get_memory_usage_str
from spc_analysis.utils.directory import create_output_dir
from spc_analysis.data.loading import load_config
from spc_analysis.data.merge import merge_datasets
from spc_analysis.metrics.dispersion import calculate_replicate_metrics
from spc_analysis.metrics.distance import compute_dmso_distance
from spc_analysis.metrics.landmarks import identify_landmarks
from spc_analysis.metrics.landmark_distance import compute_landmark_distance
from spc_analysis.visualization.correlation import create_interactive_cell_correlation_plot
from spc_analysis.metrics.landmarks import generate_landmark_vs_dmso_distance_plots
from spc_analysis.visualization.dmso_metric_plots import generate_dmso_vs_dispersion_plots
from spc_analysis.visualization.distributions import generate_comprehensive_histograms, generate_dmso_cosine_distribution_plots
from spc_analysis.visualization.data_prep import prepare_visualization_data
from spc_analysis.visualization.dimensionality import generate_dimensionality_reduction_plots
from spc_analysis.metrics import calculate_scores
from spc_analysis.visualization.hierarchical_clustering import run_hierarchical_chunk_clustering
from spc_analysis.metrics.landmark_threshold_analysis import run_landmark_threshold_analysis

def main(config_path=None):
    """
    Main function to run the analysis pipeline with the new visualization data preparation.
    
    Args:
        config_path: Path to configuration file
    """
    script_start = time.time()
    log_section("Script Started")
    
    # Load configuration
    if config_path is None:
        # Default configuration
        config = {
            'metadata_file': '/nemo/stp/hts/working/Joe_Tuersley/code/spherical-phenotype-clustering-2/cosine_distance/test_data/embeddings/train_well_metadata_new.csv',
            'embeddings_file': '/nemo/stp/hts/working/Joe_Tuersley/code/spherical-phenotype-clustering-2/cosine_distance/test_data/embeddings/train_well_embeddings.npy',
            'harmony_file': '/nemo/stp/hts/working/Phenix/Joe_19022025_GSK_fragments_V3_clickable_V1/new_harmony_analysis/data/concatenated_plate_results.csv',
            'output_dir': './spc_analysis_output',
            'mad_threshold': 0.05,
            'dmso_threshold_percentile': '99',
            'similarity_threshold': 0.2,
            'harmony_column_mapping': {
                'Metadata_Plate_ID': 'plate',
                'Metadata_Well_ID': 'well',
                'Nuclei Selected - Number of Objects': 'cell_count'
            },
            'reference_plates': [],  # Empty means all plates are considered as one set
            'test_plates': [],       # Empty means all non-reference plates
            'analysis_type': 'all',   # 'all', 'dmso_distance', or 'similarity_matching'
            'metrics_type': 'all'     # 'mad', 'var', 'std', or 'all'
        }
    else:
        config = load_config(config_path)
    
    # Create output directory
    dir_paths = create_output_dir(config.get('output_dir', './spc_analysis_output'))
    config['output_dir'] = str(dir_paths['root'])
    
    # Save configuration
    with open(dir_paths['data'] / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Merge datasets
    merged_df = merge_datasets(config, dir_paths)

    # Create interactive cell correlation plot
    create_interactive_cell_correlation_plot(merged_df, config, dir_paths)
    
    # Check if data was loaded successfully
    if merged_df is None or merged_df.empty:
        log_info("Error: Failed to load or merge data. Exiting.")
        return
    
    # Get embedding columns
    embedding_cols = [col for col in merged_df.columns if col.startswith('Z') and col[1:].isdigit()]
    if not embedding_cols:
        log_info("Error: No embedding columns found. Exiting.")
        return
    
    log_info(f"Found {len(embedding_cols)} embedding dimensions")
    
    # Split into reference and test sets based on config
    reference_plates = config.get('reference_plates', [])
    test_plates = config.get('test_plates', [])

    # Extract reference and test plates from plate_definitions if available
    if 'plate_definitions' in config and (not reference_plates or not test_plates):
        log_info("Extracting reference and test plates from plate_definitions...")
        for plate, attributes in config['plate_definitions'].items():
            if attributes.get('type') == 'reference':
                reference_plates.append(plate)
            elif attributes.get('type') == 'test':
                test_plates.append(plate)
        log_info(f"Extracted {len(reference_plates)} reference plates and {len(test_plates)} test plates from plate_definitions")

    if reference_plates:
        log_info(f"Splitting data based on reference plates: {reference_plates[:3]}... +{len(reference_plates)-3} more")
        # Convert both sides to strings for comparison
        reference_df = merged_df[merged_df['plate'].astype(str).isin([str(p) for p in reference_plates])]
        
        # Determine test set
        if test_plates:
            log_info(f"Using specified test plates: {test_plates[:3]}... +{len(test_plates)-3} more")
            # Convert both sides to strings for comparison
            test_df = merged_df[merged_df['plate'].astype(str).isin([str(p) for p in test_plates])]
        else:
            log_info("No test plates specified. Using all non-reference plates as test set.")
            # Convert both sides to strings for comparison
            test_df = merged_df[~merged_df['plate'].astype(str).isin([str(p) for p in reference_plates])]
        
        log_info(f"Reference set: {len(reference_df)} samples from {reference_df['plate'].nunique()} plates")
        log_info(f"Test set: {len(test_df)} samples from {test_df['plate'].nunique()} plates")
        
        # Count unique compounds and treatments for reference set
        reference_compounds = reference_df['compound_name'].nunique() if 'compound_name' in reference_df.columns else 0
        reference_treatments = reference_df['treatment'].nunique() if 'treatment' in reference_df.columns else 0
        
        # Get unique compounds and treatments per library in reference set
        if 'library' in reference_df.columns:
            log_info("\n=== REFERENCE SET COMPOSITION ===")
            unique_ref_libraries = reference_df['library'].unique()
            log_info(f"Found {len(unique_ref_libraries)} unique libraries in reference set")
            
            for lib in unique_ref_libraries:
                lib_df = reference_df[reference_df['library'] == lib]
                lib_compounds = lib_df['compound_name'].nunique() if 'compound_name' in lib_df.columns else 0
                lib_treatments = lib_df['treatment'].nunique() if 'treatment' in lib_df.columns else 0
                log_info(f"  Library '{lib}': {lib_compounds} unique compounds, {lib_treatments} unique treatments")
            
            log_info(f"  TOTAL: {reference_compounds} unique compounds, {reference_treatments} unique treatments")
        
        # Count unique compounds and treatments for test set
        test_compounds = test_df['compound_name'].nunique() if 'compound_name' in test_df.columns else 0
        test_treatments = test_df['treatment'].nunique() if 'treatment' in test_df.columns else 0
        
        # Get unique compounds and treatments per library in test set
        if 'library' in test_df.columns:
            log_info("\n=== TEST SET COMPOSITION ===")
            unique_test_libraries = test_df['library'].unique()
            log_info(f"Found {len(unique_test_libraries)} unique libraries in test set")
            
            for lib in unique_test_libraries:
                lib_df = test_df[test_df['library'] == lib]
                lib_compounds = lib_df['compound_name'].nunique() if 'compound_name' in lib_df.columns else 0
                lib_treatments = lib_df['treatment'].nunique() if 'treatment' in lib_df.columns else 0
                log_info(f"  Library '{lib}': {lib_compounds} unique compounds, {lib_treatments} unique treatments")
            
            log_info(f"  TOTAL: {test_compounds} unique compounds, {test_treatments} unique treatments")
    else:
        log_info("No reference plates specified. Using all data as both reference and test set.")
        reference_df = merged_df
        test_df = merged_df

    # Determine analysis type
    analysis_type = config.get('analysis_type', 'all').lower()
    log_info(f"Analysis type: {analysis_type}")
    
    # Compute metrics for reference set
    reference_mad = calculate_replicate_metrics(reference_df, embedding_cols, True, config, dir_paths)

    log_info("REFERENCE MAD METRICS CHECK")
    log_info(f"reference_mad columns: {reference_mad.columns.tolist()}")
    for metric in ['mad_cosine', 'var_cosine', 'std_cosine']:
        log_info(f"'{metric}' in reference_mad: {metric in reference_mad.columns}")

    # Initialise before use
    reference_dmso_dist = None
    
    # If we have separate test set, compute metrics for it too
    test_mad = None

    if analysis_type in ['all', 'dmso_distance']:
        reference_dmso_dist, dmso_thresholds = compute_dmso_distance(reference_df, embedding_cols, config, dir_paths, is_reference=True)
        
        # CRITICAL: Add thresholds to config for later use
        if dmso_thresholds:
            config['dmso_thresholds'] = dmso_thresholds
            log_info(f"STORED DMSO thresholds in config: {dmso_thresholds}")
            # Log the specific threshold being used
            threshold_percentile = config.get('dmso_threshold_percentile', '99')
            if threshold_percentile in dmso_thresholds:
                log_info(f"Your {threshold_percentile}% threshold is: {dmso_thresholds[threshold_percentile]:.4f}")

        
        # Compute distance from DMSO for test set if separate
        test_dmso_dist = None
        if len(test_df) > 0 and (len(test_df) != len(reference_df) or not test_df.equals(reference_df)):
            test_dmso_dist, _ = compute_dmso_distance(test_df, embedding_cols, config, dir_paths, is_reference=False)
    else:
        reference_dmso_dist = None
        test_dmso_dist = None
    
    # Identify landmarks from reference set ONLY
    landmarks = None
    if analysis_type in ['all', 'similarity_matching'] and reference_mad is not None and reference_dmso_dist is not None:
        landmarks = identify_landmarks(reference_mad, reference_dmso_dist, config, dir_paths)
        log_info(f"Identified {len(landmarks) if landmarks is not None else 0} landmarks from REFERENCE set")

        # ======================================================================
        # ADD is_self_landmark COLUMN TO merged_df
        # ========================================================================
        if landmarks is not None and len(landmarks) > 0:
            log_info("Adding is_self_landmark column to merged_df for threshold analysis...")
            
            # Get landmark treatments
            landmark_treatments = set(landmarks['treatment'].unique())
            log_info(f"Marking {len(landmark_treatments)} treatments as landmarks in merged_df")
            
            # Add is_self_landmark column
            merged_df['is_self_landmark'] = merged_df['treatment'].isin(landmark_treatments)
            
            # Log how many were marked
            landmark_count = merged_df['is_self_landmark'].sum()
            log_info(f"Marked {landmark_count} rows as is_self_landmark=True in merged_df")
            log_info(f"Unique landmark treatments in merged_df: {merged_df[merged_df['is_self_landmark']]['treatment'].nunique()}")

            # ======================================================================
            # ADD is_landmark TO reference_metrics.csv
            # ======================================================================
            if reference_mad is not None:
                reference_mad['is_landmark'] = reference_mad['treatment'].isin(landmark_treatments)
                landmark_in_mad = reference_mad['is_landmark'].sum()
                log_info(f"Added is_landmark column to reference_mad: {landmark_in_mad} landmarks out of {len(reference_mad)} treatments")
                
                output_path = dir_paths['analysis']['mad'] / 'reference_metrics.csv'
                reference_mad.to_csv(output_path, index=False)
                log_info(f"Re-saved reference_metrics.csv with is_landmark column to: {output_path}")

            # ======================================================================
            # ADD mad_cosine AND is_landmark TO reference_dmso_distances.csv
            # ======================================================================
            if reference_dmso_dist is not None and reference_mad is not None:
                # Add mad_cosine from reference_mad
                mad_subset = reference_mad[['treatment', 'mad_cosine']].copy()
                reference_dmso_dist = reference_dmso_dist.merge(mad_subset, on='treatment', how='left')
                log_info(f"Added mad_cosine to reference_dmso_dist: {reference_dmso_dist['mad_cosine'].notna().sum()} non-null values")
                
                # Add is_landmark
                reference_dmso_dist['is_landmark'] = reference_dmso_dist['treatment'].isin(landmark_treatments)
                landmark_in_dmso = reference_dmso_dist['is_landmark'].sum()
                log_info(f"Added is_landmark column to reference_dmso_dist: {landmark_in_dmso} landmarks out of {len(reference_dmso_dist)} treatments")
                
                output_path = dir_paths['analysis']['dmso_distances'] / 'reference_dmso_distances.csv'
                reference_dmso_dist.to_csv(output_path, index=False)
                log_info(f"Re-saved reference_dmso_distances.csv with mad_cosine and is_landmark columns to: {output_path}")

        else:
            log_info("No landmarks identified - setting is_self_landmark=False for all")
            merged_df['is_self_landmark'] = False
    
   # Compute landmark distances - CRITICAL FIX: Pass the calculated metrics dataframes
    reference_landmark_dist = None
    test_landmark_dist = None

    if analysis_type in ['all', 'similarity_matching'] and landmarks is not None and len(landmarks) > 0:
        # For reference set, compute distances to landmarks
        log_info("Computing reference landmark distances WITH enhanced metadata...")
        reference_landmark_dist = compute_landmark_distance(
            reference_df, 
            landmarks, 
            embedding_cols, 
            True, 
            config, 
            dir_paths,
            dmso_distances=reference_dmso_dist,  # CRITICAL FIX: Pass DMSO distances
            mad_df=reference_mad                 # CRITICAL FIX: Pass MAD metrics
        )
        
        # For test set, compute distances to the SAME landmarks identified from reference set
        if len(test_df) > 0 and (len(test_df) != len(reference_df) or not test_df.equals(reference_df)):
            log_info("Computing test landmark distances WITH enhanced metadata...")
            test_landmark_dist = compute_landmark_distance(
                test_df, 
                landmarks, 
                embedding_cols, 
                False, 
                config, 
                dir_paths,
                dmso_distances=test_dmso_dist,    # CRITICAL FIX: Pass DMSO distances
                mad_df=test_mad                   # CRITICAL FIX: Pass MAD metrics
            )
            log_info(f"Computed distances for test set to {len(landmarks)} landmarks from reference set")


    # Run landmark threshold analysis if enabled
    if config.get('run_landmark_threshold_analysis', False):
        log_info("Running landmark threshold analysis...")
        
        # Combine reference and test landmark distances if both exist
        combined_landmark_dist = None
        if reference_landmark_dist is not None and test_landmark_dist is not None:
            # Use test landmark distances (which includes test compounds)
            combined_landmark_dist = test_landmark_dist
        elif reference_landmark_dist is not None:
            combined_landmark_dist = reference_landmark_dist
        elif test_landmark_dist is not None:
            combined_landmark_dist = test_landmark_dist
        
        if combined_landmark_dist is not None:
            run_landmark_threshold_analysis(
                merged_df=merged_df,
                embedding_cols=embedding_cols,
                landmark_distances=combined_landmark_dist,
                config=config,
                dir_paths=dir_paths
            )
        else:
            log_info("Skipping landmark threshold analysis - no landmark distances available")
    

    # Calculate scores for reference compounds
    reference_scores = None
    if analysis_type == 'all' and reference_dmso_dist is not None and reference_landmark_dist is not None:
        log_info("Calculating scores for REFERENCE compounds...")
        reference_scores = calculate_scores(
            reference_dmso_dist, 
            reference_landmark_dist, 
            reference_mad, 
            config, 
            dir_paths, 
            is_reference=True  # NEW: Specify this is reference data
        )

    # Calculate scores for test compounds
    test_scores = None
    if analysis_type == 'all' and test_dmso_dist is not None and test_landmark_dist is not None:
        log_info("Calculating scores for TEST compounds...")
        test_scores = calculate_scores(
            test_dmso_dist, 
            test_landmark_dist, 
            test_mad, 
            config, 
            dir_paths, 
            is_reference=False  # NEW: Specify this is test data
        )

    # Log analysis datasets before visualization
    log_section("STEP 5: PREPARING FOR VISUALIZATION")
    if reference_scores is not None:
        log_info(f"Reference scores available with {len(reference_scores)} rows and {len(reference_scores.columns)} columns")
        
    if test_scores is not None and len(test_df) > 0 and (len(test_df) != len(reference_df) or not test_df.equals(reference_df)):
        log_info(f"Test scores available with {len(test_scores)} rows and {len(test_scores.columns)} columns")

    # Generate landmark vs DMSO distance plots
    if reference_scores is not None:
        generate_landmark_vs_dmso_distance_plots(reference_scores, config, dir_paths, is_reference=True)

    if test_scores is not None and len(test_df) > 0 and (len(test_df) != len(reference_df) or not test_df.equals(reference_df)):
        generate_landmark_vs_dmso_distance_plots(test_scores, config, dir_paths, is_reference=False)

    # Generate DMSO distance vs dispersion metric plots
    log_info("Generating DMSO distance vs dispersion metric plots")
    generate_dmso_vs_dispersion_plots(
        reference_scores=reference_scores,
        test_scores=test_scores,
        config=config,
        dir_paths=dir_paths
    )

    # Generate comprehensive histograms for all metrics
    generate_comprehensive_histograms(
        reference_mad=reference_mad,
        test_mad=test_mad,
        reference_dmso_dist=reference_dmso_dist,
        test_dmso_dist=test_dmso_dist,
        reference_landmark_dist=reference_landmark_dist,
        test_landmark_dist=test_landmark_dist,
        reference_scores=reference_scores,
        test_scores=test_scores,
        config=config,
        dir_paths=dir_paths
    )

    # Generate DMSO cosine distribution plots
    generate_dmso_cosine_distribution_plots(
        reference_df=reference_df,
        test_df=test_df,
        embedding_cols=embedding_cols,
        config=config,
        dir_paths=dir_paths
    )

    # Step 1: Prepare base visualization data (creates visualization_data.csv)
    log_info("Preparing comprehensive visualization data with all metrics")
    visualization_data, viz_df_agg = prepare_visualization_data(merged_df, landmarks, config, dir_paths)

    # Step 2: Generate dimensionality reduction and UPDATE the visualization data
    log_info("Adding SPC dimensionality reduction to visualization data")
    updated_viz_data = generate_dimensionality_reduction_plots(
        merged_df=merged_df,
        landmarks=landmarks,
        embedding_cols=embedding_cols,
        config=config,
        dir_paths=dir_paths,
        viz_data=visualization_data  # Pass the prepared visualization data
    )

    # Fixed main.py snippet - Step 3: Save FINAL version without embedding columns

# Step 3: Save the FINAL version with both metrics AND SPC coordinates
    if updated_viz_data is not None and ('UMAP1' in updated_viz_data.columns or 'TSNE1' in updated_viz_data.columns):
        log_info("Saving final visualization data with SPC dimensionality reduction")
        
        # Check what SPC columns we have
        spc_cols = [col for col in updated_viz_data.columns if col in ['UMAP1', 'UMAP2', 'TSNE1', 'TSNE2', 'well_count']]
        log_info(f"SPC columns found: {spc_cols}")
        
        # ============================================================================
        # FIX: Remove embedding columns (Z1-Z384) before saving
        # ============================================================================
        log_info("Removing embedding columns from spc_for_viz_app.csv (not needed by viz app)")
        
        # Detect embedding columns
        embedding_cols = [col for col in updated_viz_data.columns 
                         if col.startswith('Z') and col[1:].isdigit()]
        
        log_info(f"Found {len(embedding_cols)} embedding columns to remove: Z1-Z{len(embedding_cols)}")
        
        # Create version without embeddings
        viz_data_no_embeddings = updated_viz_data.drop(columns=embedding_cols)
        
        # Save the version WITHOUT embedding columns
        final_output_path = dir_paths['data'] / 'spc_for_viz_app.csv'
        viz_data_no_embeddings.to_csv(final_output_path, index=False)
        
        log_info(f"Saved FINAL visualization data with SPC dimensionality reduction to: {final_output_path}")
        log_info(f"  Original shape: {updated_viz_data.shape}")
        log_info(f"  Final shape: {viz_data_no_embeddings.shape}")
        log_info(f"  Removed {len(embedding_cols)} embedding columns")
        log_info(f"  Estimated space saved: {len(embedding_cols) * len(viz_data_no_embeddings) * 8 / 1024 / 1024:.1f} MB")
        
        # Verify the SPC columns are there
        for col in ['UMAP1', 'UMAP2', 'TSNE1', 'TSNE2']:
            if col in viz_data_no_embeddings.columns:
                non_null = viz_data_no_embeddings[col].notna().sum()
                log_info(f"  {col}: {non_null} non-null values ({non_null/len(viz_data_no_embeddings)*100:.1f}%)")
                
    else:
        log_info("WARNING: No SPC dimensionality reduction coordinates found in updated data")
        log_info("Using original visualization_data as fallback")
        
        # ============================================================================
        # FIX: Remove embedding columns from fallback version too
        # ============================================================================
        log_info("Removing embedding columns from fallback spc_for_viz_app.csv")
        
        # Detect embedding columns
        embedding_cols = [col for col in visualization_data.columns 
                         if col.startswith('Z') and col[1:].isdigit()]
        
        log_info(f"Found {len(embedding_cols)} embedding columns to remove: Z1-Z{len(embedding_cols)}")
        
        # Create version without embeddings
        viz_data_no_embeddings = visualization_data.drop(columns=embedding_cols)
        
        # Use original visualization_data WITHOUT embeddings as fallback
        final_output_path = dir_paths['data'] / 'spc_for_viz_app.csv'
        viz_data_no_embeddings.to_csv(final_output_path, index=False)
        
        log_info(f"Saved visualization data to: {final_output_path}")
        log_info(f"  Original shape: {visualization_data.shape}")
        log_info(f"  Final shape: {viz_data_no_embeddings.shape}")
        log_info(f"  Removed {len(embedding_cols)} embedding columns")
        log_info(f"  Estimated space saved: {len(embedding_cols) * len(viz_data_no_embeddings) * 8 / 1024 / 1024:.1f} MB")

    # ========================================================================
    # HIERARCHICAL CHUNK CLUSTERING - After spc_for_viz_app.csv is saved
    # ========================================================================
    if config.get('create_hierarchical_chunks', False):
        log_info("Generating hierarchical chunk clustering...")
        run_hierarchical_chunk_clustering(viz_df_agg, config, dir_paths)
        log_info("✓ Hierarchical chunk clustering complete")
    else:
        log_info("Hierarchical chunk clustering disabled in config")
            
    # Print summary of results
    log_section("ANALYSIS SUMMARY")
    
    if reference_mad is not None:
        log_info(f"Reference metrics analysis: {len(reference_mad)} treatments")
        
        # Log metrics statistics
        metrics_types = ['mad_cosine', 'var_cosine', 'std_cosine']
        for metric in metrics_types:
            if metric in reference_mad.columns:
                log_info(f"  {metric} - Mean: {reference_mad[metric].mean():.6f}, Min: {reference_mad[metric].min():.6f}, Max: {reference_mad[metric].max():.6f}")
    
    if test_mad is not None:
        log_info(f"Test metrics analysis: {len(test_mad)} treatments")
        
        # Log metrics statistics for test too
        metrics_types = ['mad_cosine', 'var_cosine', 'std_cosine']
        for metric in metrics_types:
            if metric in test_mad.columns:
                log_info(f"  {metric} - Mean: {test_mad[metric].mean():.6f}, Min: {test_mad[metric].min():.6f}, Max: {test_mad[metric].max():.6f}")
    
    if landmarks is not None:
        log_info(f"Identified landmarks: {len(landmarks)} compounds")
    
    if reference_scores is not None:
        # Find top scoring compounds
        if 'harmonic_mean_3term' in reference_scores.columns:
            top_score_column = 'harmonic_mean_3term'
        elif 'harmonic_mean_2term' in reference_scores.columns:
            top_score_column = 'harmonic_mean_2term'
        else:
            top_score_column = 'ratio_score'
            
        top_compounds = reference_scores.nlargest(5, top_score_column)
        log_info(f"Top 5 reference compounds by {top_score_column}:")
        for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
            log_info(f"  {i}. {row['treatment']}: {row[top_score_column]:.4f}")
    
    # Print runtime
    log_section("THRESHOLD DEBUG SUMMARY") 
    log_info(f"Config dmso_threshold_percentile: {config.get('dmso_threshold_percentile', 'NOT SET')}")
    
    # Reference/Test thresholds
    if 'dmso_thresholds' in config:
        log_info("DMSO thresholds in config (from reference set):")
        for pct, val in config['dmso_thresholds'].items():
            log_info(f"  Reference {pct}%: {val:.4f}")
    else:
        log_info("ERROR: No dmso_thresholds found in config!")
    
    # ADD THIS - Combined thresholds
    if 'dmso_combined_thresholds' in config:
        log_info("Combined DMSO thresholds in config:")
        for pct, val in config['dmso_combined_thresholds'].items():
            log_info(f"  Combined {pct}%: {val:.4f}")
    else:
        log_info("No combined DMSO thresholds found in config (will be calculated during plotting)")
    
    log_section("Script Completion Summary")
    log_info(f"Total runtime: {time.time() - script_start:.2f} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SPC Cosine Distance Analysis')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)