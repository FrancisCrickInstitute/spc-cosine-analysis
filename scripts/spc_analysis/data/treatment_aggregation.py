# Treatment-level aggregation

import numpy as np
import pandas as pd
from ..utils.logging import log_info, log_section

def aggregate_to_treatment_level(viz_df):
    """
    Aggregate visualization data to treatment level.
    
    For Z columns (embeddings): Take the mean across treatments
    For other columns: Take the first value per treatment
    
    Args:
        viz_df: DataFrame with visualization data
        
    Returns:
        DataFrame: Treatment-level aggregated data
    """
    log_section("AGGREGATING DATA TO TREATMENT LEVEL")
    
    if viz_df is None or len(viz_df) == 0:
        log_info("No data to aggregate")
        return viz_df
    
    # Identify embedding columns (Z columns)
    embedding_cols = [col for col in viz_df.columns if col.startswith('Z') and col[1:].isdigit()]
    log_info(f"Found {len(embedding_cols)} embedding columns to average")
    
    # Identify non-embedding columns (everything else)
    non_embedding_cols = [col for col in viz_df.columns if col not in embedding_cols and col != 'treatment']
    log_info(f"Found {len(non_embedding_cols)} non-embedding columns")
    
    # Group by treatment
    grouped = viz_df.groupby('treatment')
    log_info(f"Found {len(grouped)} unique treatments")
    
    # Initialize result dictionary
    result_data = {}
    
    # Add treatment column
    result_data['treatment'] = list(grouped.groups.keys())
    
    # For embedding columns: calculate mean
    log_info("Calculating mean embeddings for each treatment...")
    for col in embedding_cols:
        if col in viz_df.columns:
            result_data[col] = grouped[col].mean().values
    
    # For non-embedding columns: take first value
    log_info("Taking first values for non-embedding columns...")
    for col in non_embedding_cols:
        if col in viz_df.columns:
            try:
                # Get first values for each group
                first_values = grouped[col].first()
                
                # Convert to list to ensure 1-dimensional array
                if hasattr(first_values, 'values'):
                    result_data[col] = first_values.values.tolist()
                else:
                    result_data[col] = list(first_values)
                    
            except Exception as e:
                log_info(f"Warning: Could not aggregate column {col}: {str(e)}")
                # Create list of None values with correct length
                result_data[col] = [None] * len(result_data['treatment'])
    
    # Create aggregated dataframe
    agg_df = pd.DataFrame(result_data)
    
    log_info(f"Aggregated data from {len(viz_df)} rows to {len(agg_df)} treatments")
    log_info(f"Aggregated dataframe shape: {agg_df.shape}")
    
    # Log sample of results
    log_info("Sample treatments after aggregation:")
    for i, treatment in enumerate(agg_df['treatment'].head(5)):
        original_count = len(viz_df[viz_df['treatment'] == treatment])
        log_info(f"  {i+1}. {treatment}: {original_count} samples -> 1 aggregated row")
    
    return agg_df