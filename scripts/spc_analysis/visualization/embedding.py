# Interactive UMAP & tSNE plots on embeddings

import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ..utils.logging import log_info

def generate_interactive_embedding_plot(data, x, y, color, hover_data, title, filename, dir_path, is_continuous=None, color_palette=None):
    """
    Generate interactive embedding plot using px.scatter with enhanced hover data.
    
    Args:
        data: DataFrame with data to plot
        x, y: Column names for x and y coordinates
        color: Column name for coloring points
        hover_data: List of columns to show in hover tooltip
        title: Plot title
        filename: Output filename
        dir_path: Output directory path
        is_continuous: Boolean flag to force continuous or discrete color scale (auto-detect if None)
        color_palette: Color palette to use (overrides default selection)
    """
    try:
        # Print diagnostic information
        log_info(f"Starting plot generation for {title}")
        log_info(f"Data shape: {data.shape}")
        log_info(f"Columns to use: x={x}, y={y}, color={color}")
        log_info(f"Color is continuous: {is_continuous}")
        log_info(f"Hover columns requested: {len(hover_data)} columns")
        
        if color not in data.columns:
            log_info(f"ERROR: Color column '{color}' not found in data!")
            log_info(f"Available columns: {data.columns.tolist()}")
            return None
        
        # Create a clean copy of data
        plot_data = data.copy()
        
        # Check if there's any non-NaN data in the color column
        non_nan_count = plot_data[color].count()
        log_info(f"Column '{color}' has {non_nan_count} non-NaN values out of {len(plot_data)} rows")
        
        if non_nan_count == 0:
            log_info(f"ERROR: No data in column '{color}' - all values are NaN")
            return None
        
        # Handle NaN values in color column - important for metrics that might not exist for all treatments
        if pd.api.types.is_numeric_dtype(plot_data[color]):
            # For numeric columns, we can use a special color for NaN values
            log_info(f"Handling NaN values in numeric column '{color}'")
            # We'll keep NaNs as is for now, px.scatter will handle them
        else:
            # For categorical columns, replace NaNs with "Unknown"
            nan_mask = plot_data[color].isna()
            if nan_mask.any():
                plot_data.loc[nan_mask, color] = "Unknown"
                log_info(f"Replaced {nan_mask.sum()} NaN values with 'Unknown' in '{color}'")
        
        # STEP 1: Handle the color column for MOA or similar fields with comma-separated values
        log_info(f"Processing color column: {color}")
        
        # First ensure all values are strings for categorical variables
        if is_continuous is False or (is_continuous is None and not pd.api.types.is_numeric_dtype(plot_data[color])):
            plot_data[color] = plot_data[color].astype(str)
            
            # For MOA or similar fields, take only first value before comma
            if color == 'moa' or (plot_data[color].str.contains(',').any() if len(plot_data) > 0 else False):
                log_info("Detected comma-separated values in color column, using first value only")
                # Create new column for coloring to preserve original
                color_col_name = f"{color}_first"
                plot_data[color_col_name] = plot_data[color].apply(
                    lambda val: val.split(',')[0].strip() if ',' in str(val) else val
                )
                
                # Show sample of original vs processed color values
                sample_colors = pd.DataFrame({
                    'original': plot_data[color].head(5),
                    'processed': plot_data[color_col_name].head(5)
                })
                log_info(f"Sample of color processing:\n{sample_colors}")
                
                # Update color variable to use the new column
                color_to_use = color_col_name
            else:
                color_to_use = color
        else:
            color_to_use = color
            # Ensure it's numeric for continuous coloring
            if is_continuous is True and not pd.api.types.is_numeric_dtype(plot_data[color_to_use]):
                log_info(f"Converting {color_to_use} to numeric for continuous coloring")
                plot_data[color_to_use] = pd.to_numeric(plot_data[color_to_use], errors='coerce')
        
        # STEP 2: Prepare hover data with proper formatting
        hover_data_dict = {}
        
        # First, check if we have reference and test data
        ref_columns = [col for col in hover_data if col.startswith('reference_')]
        test_columns = [col for col in hover_data if col.startswith('test_')]
        
        has_ref_data = any(plot_data[col].notna().any() for col in ref_columns if col in plot_data.columns)
        has_test_data = any(plot_data[col].notna().any() for col in test_columns if col in plot_data.columns)
        
        log_info(f"Reference data present: {has_ref_data}, Test data present: {has_test_data}")
        
        # Define format for each column type with expanded metrics
        numeric_formats = {
            # Specific formatting for metric columns - including new metrics
            'cosine_distance_from_dmso': ':.4f',
            'reference_cosine_distance_from_dmso': ':.4f',
            'test_cosine_distance_from_dmso': ':.4f',
            'mad_cosine': ':.4f',
            'reference_mad_cosine': ':.4f',
            'test_mad_cosine': ':.4f',
            'var_cosine': ':.4f',
            'reference_var_cosine': ':.4f',
            'test_var_cosine': ':.4f',
            'std_cosine': ':.4f',
            'reference_std_cosine': ':.4f',
            'test_std_cosine': ':.4f',
            'median_distance': ':.4f',
            'reference_median_distance': ':.4f',
            'test_median_distance': ':.4f',
            'compound_uM': ':.4f',
            # Default format for other numeric columns
            'default': ':.2f'
        }
        
        # Create hover_data_dict with proper formatting
        for col in hover_data:
            if col in plot_data.columns:
                # For numeric columns, use appropriate format
                if pd.api.types.is_numeric_dtype(plot_data[col]):
                    # Use specific format if defined, otherwise use default
                    format_spec = numeric_formats.get(col, numeric_formats['default'])
                    hover_data_dict[col] = format_spec
                else:
                    hover_data_dict[col] = True
        
        # STEP 3: Create the plot with px.scatter
        
        # Determine if we need continuous or discrete color
        if is_continuous is None:
            # Auto-detect: continuous if numeric and many unique values
            is_numeric = pd.api.types.is_numeric_dtype(plot_data[color_to_use])
            n_unique = plot_data[color_to_use].nunique()
            auto_continuous = is_numeric and n_unique > 12
            log_info(f"Auto-detected continuous={auto_continuous} (is_numeric={is_numeric}, n_unique={n_unique})")
        else:
            auto_continuous = is_continuous
        
        # Set color parameters based on continuous/discrete
        color_params = {}
        if auto_continuous:
            # For continuous coloring
            log_info("Using continuous color scale")
            if color_palette:
                color_params['color_continuous_scale'] = color_palette
            else:
                color_params['color_continuous_scale'] = 'Viridis'
                
            # For columns with NaN values, we'll use the range_color to ensure consistency
            if plot_data[color_to_use].isna().any():
                non_na_values = plot_data[color_to_use].dropna()
                if len(non_na_values) > 0:
                    min_val = non_na_values.min()
                    max_val = non_na_values.max()
                    color_params['range_color'] = [min_val, max_val]
                    log_info(f"Setting color range to [{min_val}, {max_val}] for continuous column with NaNs")
        else:
            # For discrete coloring
            log_info("Using discrete color scale")
            if color_palette:
                color_params['color_discrete_sequence'] = color_palette
            elif color_to_use == 'library':
                # Use qualitative color scale for library
                color_params['color_discrete_sequence'] = px.colors.qualitative.Bold
            elif 'landmark' in color_to_use:
                # Use a specific palette for landmarks
                color_params['color_discrete_sequence'] = px.colors.qualitative.Set1
            elif color_to_use == 'moa' or color_to_use == 'moa_first':
                # Use a larger palette for MOA
                color_params['color_discrete_sequence'] = px.colors.qualitative.Dark24
            elif 'well' in color_to_use:
                # Use Bold colors for well rows/columns
                color_params['color_discrete_sequence'] = px.colors.qualitative.Bold
            else:
                # Default to Light24 for other discrete variables
                color_params['color_discrete_sequence'] = px.colors.qualitative.Light24
            
            # For high cardinality categorical variables (like treatment and MOA) 
            # we need to handle them specially - but we won't limit categories since you want to see all
            if not auto_continuous and plot_data[color_to_use].nunique() > 100:
                log_info(f"High cardinality detected: {plot_data[color_to_use].nunique()} unique values")
                log_info("Colors will be recycled for this high-cardinality column")
        
        # Create the plot with hover_data_dict for proper formatting
        fig = px.scatter(
            plot_data,
            x=x,
            y=y,
            color=color_to_use,
            title=title,
            hover_data=hover_data_dict,
            height=800,
            width=1200,
            **color_params
        )
        
        # STEP 4: Customize the plot appearance
        # Make points smaller
        fig.update_traces(
            marker=dict(size=5),  # Smaller points
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Improve axis labels
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            legend_title=color.replace('_', ' ').title()
        )
        
        # Handle large discrete legends differently
        if not auto_continuous and plot_data[color_to_use].nunique() > 20:
            log_info(f"Optimizing legend for {plot_data[color_to_use].nunique()} categories")
            fig.update_layout(
                legend=dict(
                    itemsizing='constant',
                    itemwidth=30,
                    font=dict(size=8)
                )
            )
            # For very large numbers of categories, use scrolling legend
            if plot_data[color_to_use].nunique() > 50:
                fig.update_layout(
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.01,
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='rgba(0,0,0,0.2)',
                        borderwidth=1
                    )
                )
        
        # STEP 5: Save the plot
        output_path = dir_path / filename
        log_info(f"Saving plot to {output_path}")
        
        fig.write_html(
            output_path,
            include_plotlyjs='cdn',
            full_html=False,
            config={'displayModeBar': True}
        )
        
        log_info(f"Successfully saved plot to {output_path}")
        return fig  # Return the figure object for potential reuse
        
    except Exception as e:
        log_info(f"Error in plot generation: {str(e)}")
        log_info(f"Error traceback: {traceback.format_exc()}")
        
        # Provide a fallback attempt with minimal functionality
        try:
            log_info("Attempting fallback plot with minimal functionality")
            
            # Simple plot with bare minimum functionality
            fig = px.scatter(
                data,
                x=x,
                y=y,
                color=color,
                title=f"{title} (fallback)"
            )
            
            # Make points smaller
            fig.update_traces(marker=dict(size=5))
            
            # Save fallback plot
            fallback_path = dir_path / f"fallback_{filename}"
            fig.write_html(fallback_path, include_plotlyjs='cdn')
            log_info(f"Saved fallback plot to {fallback_path}")
            
            return fig
            
        except Exception as fallback_error:
            log_info(f"Fallback plot also failed: {str(fallback_error)}")
            return None

def create_combined_interactive_plot(data, x, y, color_columns, hover_data, base_title, filename, dir_path):
    """
    Create a single interactive plot with dropdown to select different coloring variables.
    
    Args:
        data: DataFrame with data to plot
        x, y: Column names for x and y coordinates
        color_columns: List of tuples (column_name, is_continuous, display_name, color_palette)
        hover_data: List of columns to show in hover tooltip
        base_title: Base title for the plot
        filename: Output filename
        dir_path: Output directory path
    """
    try:
        # Print diagnostic information
        log_info(f"Starting combined interactive plot generation for {base_title}")
        log_info(f"Data shape: {data.shape}")
        log_info(f"Number of color options: {len(color_columns)}")
        
        # Create a clean copy of data
        plot_data = data.copy()
        
        # Prepare hover part definitions - with clear labels and proper formatting
        hover_parts = []
        for col in hover_data:
            if col in plot_data.columns:
                # Format the column name for display - remove prefixes for metrics
                if col.startswith('reference_') or col.startswith('test_'):
                    # Remove the reference_ or test_ prefix for display
                    base_col = col.split('_', 1)[1]
                    display_name = base_col.replace('_', ' ').title()
                else:
                    display_name = col.replace('_', ' ').title()
                
                # For numeric columns, format appropriately
                if pd.api.types.is_numeric_dtype(plot_data[col]):
                    # Use 4 decimal places for specific metrics
                    if any(metric in col for metric in ['cosine_distance_from_dmso', 'mad_cosine', 'var_cosine', 'std_cosine', 
                                                      'median_distance', 'compound_uM', 'closest_landmark_distance']):
                        hover_parts.append(f"<b>{display_name}</b>: %{{customdata[{len(hover_parts)}]:.4f}}")
                    else:
                        hover_parts.append(f"<b>{display_name}</b>: %{{customdata[{len(hover_parts)}]:.2f}}")
                else:
                    hover_parts.append(f"<b>{display_name}</b>: %{{customdata[{len(hover_parts)}]}}")
        
        # Base hover template
        hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"
        
        # Create a base figure - we'll add the traces later
        fig = go.Figure()
        
        # Keep track of all buttons for the dropdown
        buttons = []
        visible_state = []  # For tracking which traces belong to which color option
        current_traces = 0
        
        # Process each color column
        for i, (col_name, is_continuous, display_name, color_palette) in enumerate(color_columns):
            if col_name not in plot_data.columns:
                log_info(f"Skipping {display_name} - column '{col_name}' not found")
                continue
            
            # Check if there's any data in this column
            non_nan_count = plot_data[col_name].count()
            if non_nan_count == 0:
                log_info(f"Skipping {display_name} - no non-NaN values")
                continue
                
            log_info(f"Processing color option: {display_name}")
            
            # Handle NaN values for this column
            plot_data_copy = plot_data.copy()
            if pd.api.types.is_numeric_dtype(plot_data_copy[col_name]) and plot_data_copy[col_name].isna().any():
                # Leave NaNs as is for numeric columns - they'll be colored gray by default
                pass
            elif plot_data_copy[col_name].isna().any():
                # For categorical columns, replace NaNs with "Unknown"
                nan_mask = plot_data_copy[col_name].isna()
                plot_data_copy.loc[nan_mask, col_name] = "Unknown"
                log_info(f"Replaced {nan_mask.sum()} NaN values with 'Unknown' in '{col_name}'")
            
            # Handle categorical variables
            if not is_continuous:
                # Process as categorical
                plot_data_copy[col_name] = plot_data_copy[col_name].astype(str)
                
                # For MOA or similar fields with commas, take only first value
                if col_name == 'moa' or (plot_data_copy[col_name].str.contains(',').any() if len(plot_data_copy) > 0 else False):
                    color_col_name = f"{col_name}_first"
                    if color_col_name not in plot_data_copy.columns:
                        plot_data_copy[color_col_name] = plot_data_copy[col_name].apply(
                            lambda val: val.split(',')[0].strip() if ',' in str(val) else val
                        )
                    col_to_use = color_col_name
                else:
                    col_to_use = col_name
            else:
                # Handle continuous fields
                col_to_use = col_name
                if not pd.api.types.is_numeric_dtype(plot_data_copy[col_to_use]):
                    log_info(f"Converting {col_to_use} to numeric for continuous coloring")
                    plot_data_copy[col_to_use] = pd.to_numeric(plot_data_copy[col_to_use], errors='coerce')
            
            # Create customdata for hover (same for all traces)
            hover_columns = [col for col in hover_data if col in plot_data_copy.columns]
            
            # Use numpy array for customdata to avoid pandas-specific issues
            customdata = np.array([plot_data_copy[col].values for col in hover_columns]).T
            log_info(f"Created customdata array with shape {customdata.shape}")
            
            # For discrete variables, create separate traces for each category
            if not is_continuous:
                # Get categories and colors
                categories = plot_data_copy[col_to_use].dropna().unique()
                log_info(f"Column '{col_to_use}' has {len(categories)} unique values")
                
                # Get appropriate color palette
                if color_palette is None:
                    if col_to_use == 'library' or 'well' in col_to_use:
                        colors = px.colors.qualitative.Bold
                    elif 'landmark' in col_to_use:
                        colors = px.colors.qualitative.Set1
                    elif col_to_use.startswith('moa'):
                        colors = px.colors.qualitative.Dark24
                    else:
                        colors = px.colors.qualitative.Light24
                else:
                    colors = color_palette
                
                # If we have more categories than colors, we'll recycle colors
                if len(categories) > len(colors):
                    log_info(f"Recycling colors - {len(categories)} categories but only {len(colors)} colors")
                
                # Count how many traces we added for this color option
                category_traces = 0
                
                # Add a trace for each category
                for j, category in enumerate(categories):
                    # Get data for this category
                    mask = plot_data_copy[col_to_use] == category
                    if mask.sum() == 0:
                        continue  # Skip empty categories
                    
                    # Get color (recycle if needed)
                    color_idx = j % len(colors)
                    color_value = colors[color_idx]
                    
                    # Add trace for this category
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data_copy.loc[mask, x],
                            y=plot_data_copy.loc[mask, y],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=color_value
                            ),
                            customdata=customdata[mask],
                            hovertemplate=hovertemplate,
                            name=str(category),
                            visible=(i == 0)  # Only first color option is visible initially
                        )
                    )
                    category_traces += 1
                
                # Update trace tracking
                if category_traces > 0:
                    visible_state.append((current_traces, current_traces + category_traces))
                    current_traces += category_traces
                    log_info(f"Added {category_traces} category traces for {display_name}")
                else:
                    log_info(f"No valid categories found for {display_name}")
            
            else:
                # For continuous variables, use a single trace with colorscale
                # Check if we have non-NaN values
                if plot_data_copy[col_to_use].notna().sum() > 0:
                    # Set a colorscale
                    if color_palette is None:
                        colorscale = 'Viridis'
                    else:
                        colorscale = color_palette
                    
                    # Create trace with color mapping
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data_copy[x],
                            y=plot_data_copy[y],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=plot_data_copy[col_to_use],
                                colorscale=colorscale,
                                colorbar=dict(title=display_name),
                                showscale=True
                            ),
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                            name=display_name,
                            visible=(i == 0)  # Only first color option is visible initially
                        )
                    )
                    
                    # Update trace tracking
                    visible_state.append((current_traces, current_traces + 1))
                    current_traces += 1
                    log_info(f"Added continuous trace for {display_name}")
                else:
                    log_info(f"Skipping {display_name} - no non-NaN values")
            
            # Create the button for this color option - only if we added traces
            if len(visible_state) > i:
                # Get the visibility indices for this option
                start, end = visible_state[i]
                
                # Create visibility array for all traces
                visibility = [False] * current_traces
                for idx in range(start, end):
                    visibility[idx] = True
                
                button = dict(
                    label=display_name,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"{base_title}<br>Colored by {display_name}"}
                    ]
                )
                buttons.append(button)
        
        # Add dropdown menu if we have buttons
        if buttons:
            fig.update_layout(
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=1.0,
                        xanchor="right",
                        y=1.15,
                        yanchor="top",
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='rgba(0,0,0,0.2)',
                        borderwidth=1
                    ),
                ]
            )
        else:
            log_info("No buttons created - no valid color options found.")
            # Add a dummy trace if we have no traces
            if current_traces == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode='markers',
                        marker=dict(size=0),
                        showlegend=False
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"{base_title}<br>Select a variable to color by",
            xaxis_title=x,
            yaxis_title=y,
            height=800,
            width=1200,
            hovermode='closest',
            template='plotly_white'
        )
        
        # Save the plot
        output_path = dir_path / filename
        log_info(f"Saving combined interactive plot to {output_path}")
        
        fig.write_html(
            output_path,
            include_plotlyjs='cdn',
            full_html=False,
            config={'displayModeBar': True}
        )
        
        log_info(f"Successfully saved combined interactive plot to {output_path}")
        
    except Exception as e:
        log_info(f"Error creating combined interactive plot: {str(e)}")
        import traceback
        log_info(f"Error traceback: {traceback.format_exc()}")
        
        # Save error information
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "data_shape": data.shape if hasattr(data, 'shape') else None,
            "color_columns": str(color_columns),
            "hover_data": hover_data
        }
        
        import json
        error_path = dir_path / f"error_info_{filename.replace('.html', '.json')}"
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2, default=str)
            
        log_info(f"Saved error information to {error_path}")