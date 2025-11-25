"""
Hierarchical Chunk Clustering for SPC Analysis
Integrated version of standalone hierarchical_chunk_clustering_V7.py

This module creates chunked hierarchical clustering heatmaps with:
- 4 metadata colorbars (Row, Col, Plate, Anno)
- Multiple split types (test_only, reference_only, etc.)
- Single multi-page PDF per split
- Searchable vector PDFs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform, pdist  # ADD pdist HERE
from pathlib import Path
import re
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from ..utils.logging import log_info, log_section


# ============================================================================
# HELPER FUNCTIONS FOR DATASET TYPE CHECKING
# ============================================================================

def _is_test_compound(treatment, metadata_lookup, config):
    """
    Check if treatment is from test dataset using library_definitions from config
    
    Args:
        treatment: Treatment name
        metadata_lookup: Dictionary with treatment metadata (must have 'library' key)
        config: Config dict with library_definitions section
    
    Returns:
        bool: True if test compound, False otherwise
    """
    # Get test libraries from config
    test_libraries = config.get('library_definitions', {}).get('test_libraries', [])
    
    if not test_libraries:
        # Fallback: if no library_definitions, assume not test
        return False
    
    meta = metadata_lookup.get(treatment, {})
    library = meta.get('library')
    
    if pd.notna(library) and library in test_libraries:
        return True
    
    return False


def _is_reference_compound(treatment, metadata_lookup, config):
    """
    Check if treatment is from reference dataset using library_definitions from config
    
    Args:
        treatment: Treatment name
        metadata_lookup: Dictionary with treatment metadata (must have 'library' key)
        config: Config dict with library_definitions section
    
    Returns:
        bool: True if reference compound, False otherwise
    """
    # Get reference libraries from config
    reference_libraries = config.get('library_definitions', {}).get('reference_libraries', [])
    
    if not reference_libraries:
        # Fallback: if no library_definitions, assume not reference
        return False
    
    meta = metadata_lookup.get(treatment, {})
    library = meta.get('library')
    
    if pd.notna(library) and library in reference_libraries:
        return True
    
    return False


# ============================================================================
# SPLITS CONFIGURATION FUNCTION
# ============================================================================

def _create_splits_ordered(ordered_treatments, ordered_libraries, metadata_lookup, config):
    """
    Create splits using ordered data and config-defined libraries
    
    Args:
        ordered_treatments: List of treatment names in clustering order
        ordered_libraries: List of library names in clustering order
        metadata_lookup: Dictionary with treatment metadata
        config: Config dict with library_definitions section
    
    Returns:
        dict: Split configurations with filter functions and descriptions
    """
    log_info("Creating split configurations...")
    
    # Get library definitions from config
    test_libraries = config.get('library_definitions', {}).get('test_libraries', [])
    reference_libraries = config.get('library_definitions', {}).get('reference_libraries', [])
    
    # Get all landmarks (is_self_landmark==True)
    all_landmarks = set()
    for treatment in ordered_treatments:
        meta = metadata_lookup.get(treatment, {})
        if meta.get('is_self_landmark') == True:
            all_landmarks.add(treatment)
    
    # Get valid test treatments
    valid_test_treatments = set()
    for treatment in ordered_treatments:
        if _is_test_compound(treatment, metadata_lookup, config):
            meta = metadata_lookup.get(treatment, {})
            if meta.get('valid_for_phenotypic_makeup') == True:
                valid_test_treatments.add(treatment)
    
    # Find relevant landmarks for valid test compounds
    relevant_landmarks = set()
    for treatment in valid_test_treatments:
        meta = metadata_lookup.get(treatment, {})
        for landmark_col in ['closest_landmark_PP_ID_uM', 'second_closest_landmark_PP_ID_uM',
                            'third_closest_landmark_PP_ID_uM']:
            landmark_id = meta.get(landmark_col)
            if pd.notna(landmark_id):
                # Find treatment matching this landmark ID
                for t, m in metadata_lookup.items():
                    if m.get('PP_ID_uM') == landmark_id and t in ordered_treatments:
                        relevant_landmarks.add(t)
    
    log_info(f"  All landmarks (is_self_landmark==True): {len(all_landmarks)}")
    log_info(f"  Valid test treatments: {len(valid_test_treatments)}")
    log_info(f"  Relevant landmarks for valid test: {len(relevant_landmarks)}")
    
    # Define splits
    splits_ordered = {
        'test_and_reference': {
            'filter_fn': lambda idx: True,
            'description': 'All treatments (test + reference)'
        },
        'test_only': {
            'filter_fn': lambda idx: _is_test_compound(ordered_treatments[idx], metadata_lookup, config),
            'description': 'Test compounds only (from library_definitions)'
        },
        'reference_only': {
            'filter_fn': lambda idx: _is_reference_compound(ordered_treatments[idx], metadata_lookup, config),
            'description': 'Reference compounds only (from library_definitions)'
        },
        'reference_landmark': {
            'filter_fn': lambda idx: (
                _is_reference_compound(ordered_treatments[idx], metadata_lookup, config) and
                ordered_treatments[idx] in all_landmarks
            ),
            'description': 'Reference landmarks (is_self_landmark==True)'
        },
        'test_and_all_reference_landmarks': {
            'filter_fn': lambda idx: (
                _is_test_compound(ordered_treatments[idx], metadata_lookup, config) or
                ordered_treatments[idx] in all_landmarks
            ),
            'description': 'All test + All reference landmarks (is_self_landmark==True)'
        },
        'test_valid_and_relevant_landmarks': {
            'filter_fn': lambda idx: (
                ordered_treatments[idx] in valid_test_treatments or
                ordered_treatments[idx] in relevant_landmarks
            ),
            'description': 'Valid test + Relevant reference landmarks'
        }
    }
    
    # Log counts for each split
    log_info("\nSplit sizes:")
    for split_name, split_config in splits_ordered.items():
        split_indices = [i for i in range(len(ordered_treatments)) if split_config['filter_fn'](i)]
        log_info(f"  {split_name}: {len(split_indices)} treatments")
    
    return splits_ordered


# Function moved from deprecated seaborn_clustering script to here

def create_similarity_matrix(agg_df, output_dir):
    """
    Create and save treatment similarity matrix for hierarchical clustering.
    This replaces the functionality from seaborn_clustering.py
    
    Args:
        agg_df: Treatment-level aggregated DataFrame
        output_dir: Output directory Path object
        
    Returns:
        pd.DataFrame: Similarity matrix DataFrame, or None if creation fails
    """
    log_section("CREATING TREATMENT SIMILARITY MATRIX")
    
    if agg_df is None or len(agg_df) == 0:
        log_info("No aggregated data available for similarity matrix")
        return None
    
    # Filter out DMSO and invalid treatments
    agg_df_filtered = agg_df[~agg_df['treatment'].str.startswith('DMSO', na=False)].copy()
    log_info(f"Removed DMSO treatments: {len(agg_df)} -> {len(agg_df_filtered)}")
    
    # Remove treatments with NaN values in embeddings
    embedding_cols = [col for col in agg_df_filtered.columns if col.startswith('Z') and col[1:].isdigit()]
    if len(embedding_cols) == 0:
        log_info("No embedding columns found for similarity matrix")
        return None
    
    nan_mask = agg_df_filtered[embedding_cols].isna().any(axis=1)
    agg_df_clean = agg_df_filtered[~nan_mask].copy()
    log_info(f"Removed treatments with NaN embeddings: {len(agg_df_filtered)} -> {len(agg_df_clean)}")
    
    if len(agg_df_clean) < 2:
        log_info("Not enough valid treatments for similarity matrix")
        return None
    
    log_info(f"Final dataset for similarity matrix: {len(agg_df_clean)} treatments with {len(embedding_cols)} embedding dimensions")
    
    # Calculate pairwise cosine distances
    log_info("Calculating pairwise cosine distances...")
    embeddings_matrix = agg_df_clean[embedding_cols].values
    
    # Calculate cosine distance matrix
    cosine_distances = pdist(embeddings_matrix, metric='cosine')
    distance_matrix = squareform(cosine_distances)
    
    # Convert to similarity matrix (1 - distance) for better visualization
    similarity_matrix = 1 - distance_matrix
    
    # Create DataFrame with treatment labels
    treatment_labels = agg_df_clean['treatment'].values
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=treatment_labels,
        columns=treatment_labels
    )
    
    log_info(f"Created {similarity_df.shape} similarity matrix")
    
    # Save similarity matrix as Parquet
    similarity_parquet_path = output_dir / 'treatment_similarity_matrix.parquet'
    similarity_df.to_parquet(similarity_parquet_path, engine='pyarrow', compression='snappy')
    log_info(f"Saved similarity matrix to: {similarity_parquet_path}")
    
    return similarity_df



def save_split_as_single_pdf(split_dir, split_name, figures):
    """
    Save all figures for a split into a single high-quality PDF
    
    Args:
        split_dir: Path to the split directory
        split_name: Name of the split
        figures: List of matplotlib figure objects
    """
    output_path = split_dir / f"{split_name}_all_chunks.pdf"
    
    log_info(f"Creating PDF: {output_path}")
    log_info(f"Total chunks: {len(figures)}")
    
    # Create multi-page PDF
    with PdfPages(output_path) as pdf:
        for idx, fig in enumerate(figures, 1):
            log_info(f"  Adding chunk {idx}/{len(figures)} to PDF...")
            pdf.savefig(fig, dpi=300)  # High quality vector PDF
            plt.close(fig)  # Close figure to free memory
    
    log_info(f"✓ Saved: {output_path}")
    log_info("  File is searchable - all axis labels are preserved as text!")


def load_similarity_matrix(matrix_path):
    """Load pre-computed similarity matrix from parquet file"""
    log_section("LOADING SIMILARITY MATRIX FOR CHUNKED CLUSTERING")
    
    log_info(f"Loading from: {matrix_path}")
    similarity_matrix_df = pd.read_parquet(matrix_path)
    log_info(f"✓ Loaded matrix shape: {similarity_matrix_df.shape}")
    
    # Convert to numpy array and get treatment names
    similarity_matrix = similarity_matrix_df.values
    treatment_names = similarity_matrix_df.columns.tolist()
    
    log_info(f"✓ Matrix has {len(treatment_names)} treatments")
    
    return similarity_matrix, treatment_names


# ============================================================================
# CHANGES NEEDED IN create_enhanced_labels (around line 200)
# ============================================================================
# Add 'config' parameter and use library lists instead of dataset_type

def create_enhanced_labels(treatments, metadata_df, config):
    """
    Create enhanced labels using metadata and config-defined libraries
    
    UPDATED: Uses library_definitions from config instead of dataset_type column
    
    Args:
        treatments: List of treatment names
        metadata_df: DataFrame with treatment metadata
        config: Config dict with library_definitions section
    
    Returns:
        Tuple of (enhanced_labels list, libraries list)
    """
    log_info("Creating enhanced labels (using library_definitions from config)...")
    
    # Get library definitions from config
    test_libraries = config.get('library_definitions', {}).get('test_libraries', [])
    reference_libraries = config.get('library_definitions', {}).get('reference_libraries', [])
    
    log_info(f"  Test libraries: {test_libraries}")
    log_info(f"  Reference libraries: {reference_libraries}")
    
    enhanced_labels = []
    libraries = []
    
    # Create lookup for faster access
    metadata_lookup = {}
    for idx, row in metadata_df.iterrows():
        treatment = row.get('treatment')
        if pd.notna(treatment):
            metadata_lookup[treatment] = row
    
    for treatment in treatments:
        row = metadata_lookup.get(treatment, pd.Series())
        
        if row.empty:
            enhanced_labels.append(treatment)
            libraries.append(None)
            continue
        
        library = row.get('library')
        libraries.append(library)
        
        concentration = row.get('compound_uM', '')
        if pd.isna(concentration) or concentration == '':
            conc_match = re.search(r'@([0-9]+\.?[0-9]*)$', str(treatment))
            concentration = conc_match.group(1) if conc_match else "0.0"
        
        # Determine if test or reference using library lists
        is_test = pd.notna(library) and library in test_libraries
        is_ref = pd.notna(library) and library in reference_libraries
        
        if is_test:
            # Test compound: PP_ID@concentration
            pp_id = row.get('PP_ID', '')
            if pd.notna(pp_id) and pp_id != '':
                label = f"{pp_id}@{concentration}uM"
            else:
                label = f"{treatment}"
        
        elif is_ref:
            # Reference compound: MoA | Target | PP_ID@concentration
            pp_id = row.get('PP_ID', '')
            target_desc = row.get('annotated_target_description_truncated_10',
                                row.get('annotated_target_description', ''))
            moa = row.get('moa_first', '')
            
            if pd.notna(target_desc) and target_desc != "":
                target_with_conc = str(target_desc) + f"@{concentration}uM"
            else:
                target_with_conc = treatment
            
            parts = []
            if pd.notna(moa) and str(moa).strip() != "":
                parts.append(str(moa))
            parts.append(target_with_conc)
            if pd.notna(pp_id) and str(pp_id).strip() != "":
                parts.append(f"{pp_id}@{concentration}uM")
            
            label = " | ".join(parts) if parts else treatment
        
        else:
            # Unknown library type
            label = f"{treatment}"
        
        enhanced_labels.append(label)
    
    # Count by library type
    test_count = sum(1 for lib in libraries if pd.notna(lib) and lib in test_libraries)
    ref_count = sum(1 for lib in libraries if pd.notna(lib) and lib in reference_libraries)
    
    log_info(f"  Test compounds: {test_count}")
    log_info(f"  Reference compounds: {ref_count}")
    log_info(f"✓ Created enhanced labels for {len(enhanced_labels)} treatments")
    
    return enhanced_labels, libraries


def extract_well_info(metadata_df, treatments):
    """
    Extract well row (A-P) and column (01-24) from well column
    
    Args:
        metadata_df: Metadata DataFrame with 'well' column
        treatments: List of treatments to extract well info for
    
    Returns:
        tuple: (well_rows list, well_columns list)
    """
    well_rows = []
    well_columns = []
    
    for treatment in treatments:
        matching_rows = metadata_df[metadata_df['treatment'] == treatment]
        
        if len(matching_rows) > 0:
            well = matching_rows.iloc[0].get('well', '')
            if pd.notna(well) and well != '':
                # Extract row (first character)
                well_row = well[0] if len(well) > 0 else 'Unknown'
                # Extract column (remaining characters, remove leading zeros)
                well_col = well[1:].lstrip('0') if len(well) > 1 else 'Unknown'
            else:
                well_row = 'Unknown'
                well_col = 'Unknown'
        else:
            well_row = 'Unknown'
            well_col = 'Unknown'
        
        well_rows.append(well_row)
        well_columns.append(well_col)
    
    return well_rows, well_columns


def extract_metadata_for_colorbars(metadata_df, treatments):
    """
    Extract metadata for creating 4 colorbars: Row, Column, Plate, Annotation
    
    Args:
        metadata_df: Metadata DataFrame
        treatments: List of treatments
        
    Returns:
        dict: Dictionary with metadata lists
    """
    log_info("Extracting metadata for colorbars...")
    
    well_rows, well_columns = extract_well_info(metadata_df, treatments)
    
    plate_barcodes = []
    manual_annotations = []
    
    for treatment in treatments:
        matching_rows = metadata_df[metadata_df['treatment'] == treatment]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            plate = row.get('plate', 'Unknown')
            plate_barcodes.append(str(plate) if pd.notna(plate) else 'Unknown')
            
            # Get manual annotation
            manual_anno = row.get('manual_annotation', 'None')
            if pd.notna(manual_anno) and str(manual_anno).strip() != '' and str(manual_anno).lower() != 'nan':
                manual_annotations.append(str(manual_anno))
            else:
                manual_annotations.append('None')
        else:
            plate_barcodes.append('Unknown')
            manual_annotations.append('None')
    
    metadata_colorbars = {
        'well_row': well_rows,
        'well_column': well_columns,
        'plate_barcode': plate_barcodes,
        'manual_annotation': manual_annotations
    }
    
    log_info(f"  Well rows: {len(set(well_rows))} unique values")
    log_info(f"  Well columns: {len(set(well_columns))} unique values")
    log_info(f"  Plates: {len(set(plate_barcodes))} unique values")
    log_info(f"  Manual annotations: {len(set(manual_annotations))} unique values")
    
    return metadata_colorbars


def perform_global_clustering(similarity_matrix, treatments):
    """
    Perform hierarchical clustering on the full similarity matrix
    
    Args:
        similarity_matrix: Numpy array of similarity values
        treatments: List of treatment names
        
    Returns:
        tuple: (linkage_matrix, global_order, ordered_treatment_names)
    """
    log_section("PERFORMING GLOBAL HIERARCHICAL CLUSTERING")
    
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Ensure it's a proper distance matrix (symmetric, zero diagonal)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Convert to condensed distance matrix for linkage
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    log_info("Computing linkage matrix (method='average')...")
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Get optimal leaf ordering
    log_info("Determining optimal leaf order...")
    global_order = leaves_list(linkage_matrix)
    
    # Get ordered treatment names
    ordered_names = [treatments[i] for i in global_order]
    
    log_info(f"✓ Clustering complete")
    log_info(f"  Linkage matrix shape: {linkage_matrix.shape}")
    log_info(f"  Optimal ordering determined for {len(global_order)} treatments")
    
    return linkage_matrix, global_order, ordered_names


def create_colorbar_mappings(metadata_colorbars):
    """
    Create color mappings for the 4 colorbars
    
    Args:
        metadata_colorbars: Dictionary with metadata lists
        
    Returns:
        dict: Color mappings for each metadata type
    """
    color_mappings = {}
    
    # Well Row colors (use Dark2 palette for letters)
    unique_rows = sorted(set(metadata_colorbars['well_row']))
    if len(unique_rows) <= 8:
        row_palette = sns.color_palette("Dark2", len(unique_rows))
    else:
        row_palette = sns.color_palette("husl", len(unique_rows))
    color_mappings['well_row'] = dict(zip(unique_rows, row_palette))
    
    # Well Column colors (use coolwarm gradient for numbers)
    unique_cols = sorted(set(metadata_colorbars['well_column']), 
                        key=lambda x: int(x) if x.isdigit() else 999)
    col_palette = sns.color_palette("coolwarm", len(unique_cols))
    color_mappings['well_column'] = dict(zip(unique_cols, col_palette))
    
    # Plate colors
    unique_plates = sorted(set(metadata_colorbars['plate_barcode']))
    if len(unique_plates) <= 10:
        plate_palette = sns.color_palette("tab10", len(unique_plates))
    elif len(unique_plates) <= 20:
        plate_palette = sns.color_palette("tab20", len(unique_plates))
    else:
        plate_palette = sns.color_palette("husl", len(unique_plates))
    color_mappings['plate_barcode'] = dict(zip(unique_plates, plate_palette))
    
    # Manual annotation colors
    unique_annos = sorted(set(metadata_colorbars['manual_annotation']))
    if len(unique_annos) <= 10:
        anno_palette = sns.color_palette("Set1", len(unique_annos))
    else:
        anno_palette = sns.color_palette("viridis", len(unique_annos))
    color_mappings['manual_annotation'] = dict(zip(unique_annos, anno_palette))
    
    return color_mappings


def create_single_chunk_heatmap(chunk_data, chunk_labels, chunk_idx, total_chunks,
                               metadata_chunk, color_mappings, chunk_libraries,
                               chunk_treatments, metadata_lookup, config):
    """
    Create a single chunk heatmap with 4 metadata colorbars - UPDATED with dataset_type logic
    
    Args:
        chunk_treatments: List of treatment names for this chunk (NEW!)
        metadata_lookup: Dictionary with treatment metadata including dataset_type (NEW!)
    """
    n = len(chunk_labels)
    
    # Calculate figure size (match CP proportions)
    fig_width = 22
    fig_height = 22
    
    # Create figure with GridSpec - MATCH CP LAYOUT
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Define grid: 4 small rows for colorbars + 1 large row for heatmap (EXACTLY like CP)
    n_colorbar_rows = 4
    heatmap_row_height = 20
    colorbar_row_height = 0.3

    gs = GridSpec(
        n_colorbar_rows + 1,  # +1 for the main heatmap
        1, 
        figure=fig,
        height_ratios=[colorbar_row_height] * n_colorbar_rows + [heatmap_row_height],
        hspace=0.02  # Small space between subplots
    )

    # Create axes - all share the same x-axis (like CP)
    ax_main = fig.add_subplot(gs[n_colorbar_rows, 0])  # Bottom row for heatmap

    # Create colorbar axes that share the x-axis with main heatmap
    ax_top1 = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top2 = fig.add_subplot(gs[1, 0], sharex=ax_main)  
    ax_top3 = fig.add_subplot(gs[2, 0], sharex=ax_main)
    ax_top4 = fig.add_subplot(gs[3, 0], sharex=ax_main)

    # Turn off spines and ticks for colorbar axes (like CP)
    for ax in [ax_top1, ax_top2, ax_top3, ax_top4]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot main heatmap using seaborn (LIKE CP)
    heatmap = sns.heatmap(
        chunk_data,
        cmap='RdBu_r',  # CP uses RdBu_r
        center=0,        # CP centers at 0
        vmin=-1,         # CP range -1 to 1  
        vmax=1,
        xticklabels=False,
        yticklabels=chunk_labels,
        cbar_kws={
            'label': 'Cosine Similarity',
            'shrink': 0.5,
            'aspect': 15,
            'pad': 0.02
        },
        square=True,     # CP uses square cells
        ax=ax_main
    )

    # Color y-axis labels by dataset type (uses plate_definitions, not hardcoded lists!)
    for idx, (label, treatment) in enumerate(zip(ax_main.get_yticklabels(), chunk_treatments)):
        if _is_test_compound(treatment, metadata_lookup, config):
            label.set_color('blue')
        elif _is_reference_compound(treatment, metadata_lookup, config):
            label.set_color('green')

    # Set y-axis label properties (LIKE CP)
    ax_main.set_yticklabels(chunk_labels, rotation=0, fontsize=3)  # CP uses fontsize=3

    # Get axis limits AFTER seaborn has done all its adjustments
    heatmap_xlim = ax_main.get_xlim()
    heatmap_ylim = ax_main.get_ylim()

    # Create color bars using color mapping approach (LIKE CP)
    # Plot color bars using imshow with EXACT heatmap coordinates
    colorbar_configs = [
        ('well_row', 'Row', ax_top1),
        ('well_column', 'Column', ax_top2),
        ('plate_barcode', 'Plate', ax_top3)
    ]

    # First three colorbars (Row, Column, Plate) - normal processing
    for metadata_key, label, ax in colorbar_configs:
        colors = [color_mappings[metadata_key][val] for val in metadata_chunk[metadata_key]]
        
        # FIX: Convert RGB to RGBA if needed and handle array reshaping properly
        rgba_colors = []
        for color in colors:
            if len(color) == 3:  # RGB tuple
                rgba_colors.append(color + (1.0,))  # Add alpha channel
            else:  # Already RGBA
                rgba_colors.append(color)
        
        # Convert color list to array for imshow
        color_array = np.array(rgba_colors).reshape(1, -1, 4)
        
        # Use imshow with extent that matches EXACTLY the heatmap x-range
        im = ax.imshow(
            color_array,
            aspect='auto',
            extent=[heatmap_xlim[0], heatmap_xlim[1], 0, 1],
            interpolation='nearest'
        )
        
        # Force the EXACT same x-limits as main heatmap
        ax.set_xlim(heatmap_xlim)
        ax.set_ylim(0, 1)
        
        # Set label
        ax.set_ylabel(label, fontsize=6, rotation=0, ha='right', va='center')

    # SPECIAL HANDLING FOR ANNOTATION COLORBAR (ax_top4) - White for test compounds
    annotation_colors = []
    for i, (annotation_val, treatment) in enumerate(zip(metadata_chunk['manual_annotation'], chunk_treatments)):
        if _is_test_compound(treatment, metadata_lookup, config):
            # Test compounds get white color
            annotation_colors.append((1.0, 1.0, 1.0, 1.0))  # White RGBA
        else:
            # Reference compounds get normal annotation color
            color = color_mappings['manual_annotation'][annotation_val]
            if len(color) == 3:  # RGB tuple
                annotation_colors.append(color + (1.0,))  # Add alpha channel
            else:  # Already RGBA
                annotation_colors.append(color)

    # Convert annotation color list to array for imshow
    annotation_color_array = np.array(annotation_colors).reshape(1, -1, 4)

    # Use imshow with extent that matches EXACTLY the heatmap x-range
    im = ax_top4.imshow(
        annotation_color_array,
        aspect='auto',
        extent=[heatmap_xlim[0], heatmap_xlim[1], 0, 1],
        interpolation='nearest'
    )

    # Force the EXACT same x-limits as main heatmap
    ax_top4.set_xlim(heatmap_xlim)
    ax_top4.set_ylim(0, 1)

    # Set label
    ax_top4.set_ylabel('Annotation', fontsize=6, rotation=0, ha='right', va='center')

    # SPECIAL HANDLING FOR ANNOTATION COLORBAR (ax_top4) - White for test compounds
    annotation_colors = []
    for i, (annotation_val, treatment) in enumerate(zip(metadata_chunk['manual_annotation'], chunk_treatments)):
        if _is_test_compound(treatment, metadata_lookup, config):
            # Test compounds get white color
            annotation_colors.append((1.0, 1.0, 1.0, 1.0))  # White RGBA
        else:
            # Reference compounds get normal annotation color
            color = color_mappings['manual_annotation'][annotation_val]
            if len(color) == 3:  # RGB tuple
                annotation_colors.append(color + (1.0,))  # Add alpha channel
            else:  # Already RGBA
                annotation_colors.append(color)
    
    # Convert annotation color list to array for imshow
    annotation_color_array = np.array(annotation_colors).reshape(1, -1, 4)
    
    # Use imshow with extent that matches EXACTLY the heatmap x-range
    im = ax_top4.imshow(
        annotation_color_array,
        aspect='auto',
        extent=[heatmap_xlim[0], heatmap_xlim[1], 0, 1],
        interpolation='nearest'
    )
    
    # Force the EXACT same x-limits as main heatmap
    ax_top4.set_xlim(heatmap_xlim)
    ax_top4.set_ylim(0, 1)
    
    # Set label
    ax_top4.set_ylabel('Annotation', fontsize=6, rotation=0, ha='right', va='center')

    # Set IDENTICAL position for all axes in figure coordinates (LIKE CP)
    heatmap_pos = ax_main.get_position()
    
    # Force all colorbar axes to have EXACTLY the same x-position and width
    for ax in [ax_top1, ax_top2, ax_top3, ax_top4]:
        ax_pos = ax.get_position()
        ax.set_position([heatmap_pos.x0, ax_pos.y0, heatmap_pos.width, ax_pos.height])

    # Title (LIKE CP style)
    fig.suptitle(
        f"Cosine Similarity - Chunk {chunk_idx + 1}/{total_chunks}\n"
        f"Treatments {chunk_idx * len(chunk_labels) + 1}-{chunk_idx * len(chunk_labels) + len(chunk_labels)}\n"
        f"(Red = Similar, White = Orthogonal, Blue = Opposite)",
        fontsize=12,
        y=0.98
    )

    return fig


def create_chunked_heatmaps(similarity_matrix, treatments, enhanced_labels, libraries,
                           output_dir, chunk_size, splits, metadata_colorbars, 
                           metadata_lookup, config=None):
    """
    Create chunked heatmaps for different splits with 4 metadata colorbars
    
     Args:
        similarity_matrix: Ordered similarity matrix
        treatments: Ordered treatment names
        enhanced_labels: Ordered enhanced labels
        libraries: Ordered library names
        output_dir: Output directory Path object
        chunk_size: Size of each chunk
        splits: Dictionary of split configurations
        metadata_colorbars: Ordered metadata for colorbars
        metadata_lookup: Dictionary with treatment metadata including dataset_type
        config: Configuration dictionary (optional)
    """
    log_section("CREATING CHUNKED HEATMAPS")
    
    # Create color mappings once
    color_mappings = create_colorbar_mappings(metadata_colorbars)
    
    # Process each split
    for split_name, split_config in splits.items():
        log_info(f"\nProcessing split: {split_name}")
        log_info(f"  Description: {split_config['description']}")
        
        # Filter data for this split
        split_indices = [i for i in range(len(treatments)) if split_config['filter_fn'](i)]
        
        if len(split_indices) == 0:
            log_info(f"  Skipping {split_name} - no treatments match filter")
            continue
        
        log_info(f"  Split size: {len(split_indices)} treatments")
        
        # Create split subdirectory
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract split data - ADD split_treatments for dataset_type logic
        split_similarity = similarity_matrix[np.ix_(split_indices, split_indices)]
        split_labels = [enhanced_labels[i] for i in split_indices]
        split_libraries = [libraries[i] for i in split_indices]
        split_treatments = [treatments[i] for i in split_indices]  # NEW: needed for dataset_type checking
        split_metadata = {
            key: [values[i] for i in split_indices]
            for key, values in metadata_colorbars.items()
        }
        
        # Create chunks
        n_treatments = len(split_indices)
        n_chunks = (n_treatments + chunk_size - 1) // chunk_size  # Ceiling division
        log_info(f"  Creating {n_chunks} chunks of size {chunk_size}")
        
        figures = []
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_treatments)
            
            # Extract chunk data
            chunk_data = split_similarity[start_idx:end_idx, start_idx:end_idx]
            chunk_labels = split_labels[start_idx:end_idx]
            chunk_libraries = split_libraries[start_idx:end_idx]
            chunk_treatments = split_treatments[start_idx:end_idx]  # NEW: extract chunk treatments
            chunk_metadata = {
                key: values[start_idx:end_idx]
                for key, values in split_metadata.items()
            }
            
            fig = create_single_chunk_heatmap(
                chunk_data, chunk_labels, chunk_idx + 1, n_chunks,
                chunk_metadata, color_mappings, chunk_libraries,
                chunk_treatments, metadata_lookup, config  # Pass config through
            )
                    
            figures.append(fig)
        
        # Save all chunks as single PDF
        save_split_as_single_pdf(split_dir, split_name, figures)
    
    log_info("\n✓ All chunked heatmaps created successfully")


def run_hierarchical_chunk_clustering(agg_df, config, dir_paths, similarity_matrix_path=None):
    """
    Main function to run hierarchical chunk clustering with multiple splits
    """
    log_section("HIERARCHICAL CHUNK CLUSTERING")
    
    # Set up output directory
    output_dir = dir_paths['visualizations']['hierarchical_clustering']['cluster_map']
    
    # Determine similarity matrix path
    if similarity_matrix_path is None:
        similarity_matrix_path = output_dir / 'treatment_similarity_matrix.parquet'
    else:
        similarity_matrix_path = Path(similarity_matrix_path)
    

    # Define metadata path first (needed regardless of whether we create or load similarity matrix)
    metadata_filename = config.get('hierarchical_clustering_metadata_file', 'spc_for_viz_app.csv')
    metadata_path = dir_paths['data'] / metadata_filename
    
    if not metadata_path.exists():
        log_info(f"Metadata not found: {metadata_path}")
        return
    
    if not similarity_matrix_path.exists():
        log_info("Similarity matrix not found, creating it...")
        
        # Load and aggregate data
        metadata_df = pd.read_csv(metadata_path, low_memory=False)
        from ..data.treatment_aggregation import aggregate_to_treatment_level
        agg_df = aggregate_to_treatment_level(metadata_df)
        
        # Create similarity matrix
        similarity_df = create_similarity_matrix(agg_df, dir_paths['visualizations']['hierarchical_clustering']['cluster_map'])
        
        if similarity_df is None:
            log_info("Failed to create similarity matrix")
            return
    else:
        # Load existing similarity matrix
        similarity_df = pd.read_parquet(similarity_matrix_path)
    
    # Get chunk size from config
    chunk_size = config.get('hierarchical_chunk_size', 200)
    
    # REMOVED: No longer need library_definitions from config!
    # Dataset type comes from plate_definitions instead
    
    try:
        # 1. Use the similarity matrix we already loaded/created
        similarity_matrix = similarity_df.values
        treatment_names = similarity_df.columns.tolist()
        log_info(f"Using similarity matrix with {len(treatment_names)} treatments")
        
        # 2. Load metadata
        metadata_df = pd.read_csv(metadata_path, low_memory=False)
        log_info(f"Loaded metadata: {len(metadata_df)} rows, {len(metadata_df.columns)} columns")
        
        # 3. Filter out nan@ treatments
        log_info("\nFiltering treatments...")
        nan_treatments = [t for t in treatment_names if 'nan@' in t.lower()]
        log_info(f"Treatments containing 'nan@': {len(nan_treatments)}")
        
        treatments_to_keep = [i for i, t in enumerate(treatment_names) if 'nan@' not in t.lower()]
        filtered_treatments = [treatment_names[i] for i in treatments_to_keep]
        filtered_matrix = similarity_matrix[np.ix_(treatments_to_keep, treatments_to_keep)]
        log_info(f"After filtering: {len(filtered_treatments)} treatments remaining")
        
        # 4. Create enhanced labels (uses library_definitions from config)
        enhanced_labels, libraries = create_enhanced_labels(
            filtered_treatments, metadata_df, config
        )
        
        # 5. Extract metadata for color bars
        metadata_colorbars = extract_metadata_for_colorbars(metadata_df, filtered_treatments)
        
        # 6. Perform global clustering
        linkage_matrix, global_order, ordered_names = perform_global_clustering(
            filtered_matrix, filtered_treatments
        )
        
        # 7. Reorder everything based on clustering
        log_info("\nReordering data by clustering...")
        ordered_similarity = filtered_matrix[np.ix_(global_order, global_order)]
        ordered_enhanced_labels = [enhanced_labels[i] for i in global_order]
        ordered_libraries = [libraries[i] for i in global_order]
        ordered_treatments = [filtered_treatments[i] for i in global_order]
        
        # Reorder metadata for color bars
        ordered_metadata = {
            key: [values[i] for i in global_order]
            for key, values in metadata_colorbars.items()
        }
        log_info("✓ Data reordered by hierarchical clustering")
        
        # 8. Create metadata lookup for filtering
        log_info("\nCreating metadata lookup for filtering...")
        metadata_lookup = {}
        for idx, row in metadata_df.iterrows():
            treatment = row.get('treatment')
            if pd.notna(treatment):
                # CRITICAL FIX: Handle pandas merge suffix (_x, _y) for is_self_landmark column
                # When merging DataFrames that both have is_self_landmark, pandas adds suffixes
                # Priority: is_self_landmark_y (from landmark distances) > is_self_landmark > False
                is_landmark_val = row.get('is_self_landmark_y', row.get('is_self_landmark', False))

                metadata_lookup[treatment] = {
                    'library': row.get('library'),
                    'is_self_landmark': is_landmark_val,  # Use the resolved landmark value
                    'valid_for_phenotypic_makeup': row.get('valid_for_phenotypic_makeup'),
                    'closest_landmark_PP_ID_uM': row.get('closest_landmark_PP_ID_uM'),
                    'second_closest_landmark_PP_ID_uM': row.get('second_closest_landmark_PP_ID_uM'),
                    'third_closest_landmark_PP_ID_uM': row.get('third_closest_landmark_PP_ID_uM'),
                    'PP_ID_uM': row.get('PP_ID_uM')  # Needed for landmark matching
                }
        
        # 9. Create splits using the extracted function
        splits_ordered = _create_splits_ordered(
            ordered_treatments, ordered_libraries, metadata_lookup, config
        )
        
        # 10. Create chunked heatmaps
        create_chunked_heatmaps(
            ordered_similarity, ordered_treatments, ordered_enhanced_labels, ordered_libraries,
            output_dir, chunk_size=chunk_size, splits=splits_ordered,
            metadata_colorbars=ordered_metadata, 
            metadata_lookup=metadata_lookup,
            config=config
        )
        
        log_section("✓ HIERARCHICAL CHUNK CLUSTERING COMPLETED!")
        log_info(f"Results saved to: {output_dir}")
        log_info("\nGenerated PDFs:")
        log_info("  1. test_and_reference_all_chunks.pdf")
        log_info("  2. test_only_all_chunks.pdf")
        log_info("  3. reference_only_all_chunks.pdf")
        log_info("  4. reference_landmark_all_chunks.pdf")
        log_info("  5. test_and_all_reference_landmarks_all_chunks.pdf")
        log_info("  6. test_valid_and_relevant_landmarks_all_chunks.pdf")
        
    except Exception as e:
        log_info(f"\n❌ ERROR in hierarchical chunk clustering: {e}")
        import traceback
        log_info(traceback.format_exc())