import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..utils.logging import log_section, log_info

def create_static_cell_count_correlation_plots(merged_df, dir_paths):
    """
    Create multiple static R² plots for cell count vs cell pct, each colored by a different column.
    All datapoints are plotted regardless of treatment count. No legend labels included to avoid size issues.
    
    Args:
        merged_df (pd.DataFrame): Merged dataframe with cell count and cell pct
        dir_paths (dict): Dictionary containing paths for saving visualizations
    """
    log_section("CREATING STATIC R² PLOTS FOR CELL COUNT VS CELL PCT")
    
    # Ensure required columns exist
    if 'cell_count' not in merged_df.columns or 'cell_pct' not in merged_df.columns:
        log_info("Warning: Cannot create cell count correlation plot - missing required columns")
        return
    
    # Calculate overall correlation and R²
    correlation = merged_df['cell_count'].corr(merged_df['cell_pct'])
    r_squared = correlation ** 2
    
    # List of columns to create plots for
    color_columns = ['treatment', 'moa', 'library', 'compound_name', 'compound_uM', 
                     'cell_count', 'cell_pct', 'well_letter', 'well_number']
    
    # Create a separate plot for each color column
    for color_col in color_columns:
        # Skip if column doesn't exist
        if color_col not in merged_df.columns:
            log_info(f"Skipping {color_col} plot - column not found in dataframe")
            continue
        
        log_info(f"Creating static R² plot colored by {color_col}")
        
        plt.figure(figsize=(12, 9))
        
        # Check if categorical or continuous column
        is_categorical = merged_df[color_col].dtype == 'object' or pd.api.types.is_categorical_dtype(merged_df[color_col])
        
        if is_categorical:
            # For categorical columns, use color parameter but no hue (to avoid legend)
            unique_count = merged_df[color_col].nunique()
            log_info(f"Column {color_col} has {unique_count} unique values")
            
            # Create a color map without using hue (to avoid legend)
            # Get unique values and their mapping to colors
            unique_values = merged_df[color_col].dropna().unique()
            
            # Use a colormap appropriate for the number of unique values
            if unique_count <= 10:
                cmap = plt.cm.tab10
            elif unique_count <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.viridis
            
            # Create a color dictionary for the values
            color_dict = {val: cmap(i/min(unique_count, cmap.N)) for i, val in enumerate(unique_values)}
            
            # Apply colors to each point directly (no legend)
            for value in unique_values:
                subset = merged_df[merged_df[color_col] == value]
                plt.scatter(
                    subset['cell_count'], 
                    subset['cell_pct'],
                    color=color_dict[value],
                    alpha=0.7, 
                    s=30
                )
        else:
            # For continuous columns, use a colormap
            scatter = plt.scatter(
                merged_df['cell_count'],
                merged_df['cell_pct'],
                c=merged_df[color_col],
                cmap='viridis',
                alpha=0.7,
                s=30
            )
            # Add a simple colorbar without excessive labels
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_col)
        
        plt.xlabel('Cell Count')
        plt.ylabel('Cell Percentage')
        plt.title(f'Cell Count vs Cell Percentage (R² = {r_squared:.4f})\nColored by {color_col}')
        plt.grid(True, alpha=0.3)
        
        # Add trendline
        if len(merged_df) > 0:
            z = np.polyfit(merged_df['cell_count'], merged_df['cell_pct'], 1)
            p = np.poly1d(z)
            plt.plot(
                sorted(merged_df['cell_count']),
                p(sorted(merged_df['cell_count'])),
                "r--",
                alpha=0.8,
                linewidth=2
            )
        
        # Save plot to correlation folder with specific name
        try:
            cell_corr_path = dir_paths['visualizations']['correlation'] / f'static_cell_count_vs_cell_pct_by_{color_col}.png'
            plt.savefig(cell_corr_path, dpi=150, bbox_inches='tight')
            log_info(f"Saved static cell correlation plot by {color_col} to: {cell_corr_path}")
        except ValueError as e:
            log_info(f"Error saving plot for {color_col}: {e}")
            # Try with a smaller DPI as a fallback
            try:
                plt.savefig(cell_corr_path, dpi=72, bbox_inches='tight')
                log_info(f"Saved static cell correlation plot at reduced quality for {color_col}")
            except Exception as e2:
                log_info(f"Could not save plot for {color_col} even at reduced quality: {e2}")
        
        plt.close()


def create_interactive_cell_correlation_plot(merged_df, config, dir_paths):
    """
    Creates an interactive R² plot for cell_count vs cell_pct with dropdown for different coloring schemes.
    Legend labels are included in HTML output but controlled to avoid overcrowding.
    
    Args:
        merged_df: DataFrame with all data
        config: Configuration dictionary
        dir_paths: Dictionary containing directory paths
    """
    # First, create all the static plots
    create_static_cell_count_correlation_plots(merged_df, dir_paths)
    
    log_section("CREATING INTERACTIVE R² PLOT FOR CELL COUNT VS CELL PCT")
    
    # Check if required columns exist
    if 'cell_count' not in merged_df.columns or 'cell_pct' not in merged_df.columns:
        log_info("Cannot create correlation plot - missing required columns 'cell_count' or 'cell_pct'")
        return
    
    try:
        # Check available columns for coloring
        potential_color_columns = ['library', 'treatment', 'moa', 'compound_name', 'compound_uM', 
                                  'cell_count', 'cell_pct']
        
        # Add well letter and number parts if 'well' column exists
        if 'well' in merged_df.columns:
            # Extract well letter and number parts if not already present
            if 'well_letter' not in merged_df.columns:
                merged_df['well_letter'] = merged_df['well'].str[0:1]
            if 'well_number' not in merged_df.columns:
                merged_df['well_number'] = merged_df['well'].str[1:].astype(str)
            potential_color_columns.extend(['well_letter', 'well_number'])
        
        # Filter to only available columns
        available_color_columns = [col for col in potential_color_columns if col in merged_df.columns]
        
        log_info(f"Available coloring options: {available_color_columns}")
        
        # Define custom titles for each coloring option
        color_titles = {
            'library': 'Library',
            'treatment': 'Treatment',
            'moa': 'Mechanism of Action',
            'well_letter': 'Well Row (Letter)',
            'well_number': 'Well Column (Number)',
            'compound_name': 'Compound Name',
            'compound_uM': 'Compound Concentration (µM)',
            'cell_count': 'Cell Count',
            'cell_pct': 'Cell Percentage'
        }
        
        # Calculate overall correlation and R²
        correlation = merged_df['cell_count'].corr(merged_df['cell_pct'])
        r_squared = correlation ** 2
        log_info(f"Overall R² between cell_count and cell_pct: {r_squared:.4f}")
        
        # Add trendline data
        z = np.polyfit(merged_df['cell_count'], merged_df['cell_pct'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(merged_df['cell_count'].min(), merged_df['cell_count'].max(), 100)
        y_range = p(x_range)
        
        # SIMPLIFIED APPROACH: Create a separate figure for each coloring option
        # and store them to HTML files
        
        # Prepare for creating multiple figures
        html_files = []
        
        # Create a separate HTML file for each coloring option
        for i, color_col in enumerate(available_color_columns):
            is_categorical = merged_df[color_col].dtype == 'object' or pd.api.types.is_categorical_dtype(merged_df[color_col])
            num_unique = merged_df[color_col].nunique()
            log_info(f"Creating figure for {color_col} with {num_unique} unique values")
            
            # Create a completely new figure for each coloring option
            if is_categorical:
                # For categorical variables
                if num_unique <= 20:
                    # Use a discrete color map for few categories
                    fig = px.scatter(
                        merged_df,
                        x='cell_count', 
                        y='cell_pct',
                        color=color_col,
                        color_discrete_sequence=px.colors.qualitative.Bold if num_unique <= 10 else px.colors.qualitative.Light24,
                        title=f"Cell Count vs Cell Pct (R² = {r_squared:.4f})<br>Colored by {color_titles.get(color_col, color_col)}",
                        labels={
                            'cell_count': 'Cell Count',
                            'cell_pct': 'Cell Percentage',
                            color_col: color_titles.get(color_col, color_col)
                        },
                        hover_data=['cell_count', 'cell_pct', color_col]
                    )
                else:
                    # For high cardinality categories, use a continuous color scale with category codes
                    # First create a copy of the dataframe with category codes
                    df_temp = merged_df.copy()
                    df_temp['category_code'] = pd.Categorical(df_temp[color_col]).codes
                    
                    # Create figure with continuous color scale for codes
                    fig = px.scatter(
                        df_temp,
                        x='cell_count', 
                        y='cell_pct',
                        color='category_code',
                        color_continuous_scale='Viridis',
                        title=f"Cell Count vs Cell Pct (R² = {r_squared:.4f})<br>Colored by {color_titles.get(color_col, color_col)}",
                        labels={
                            'cell_count': 'Cell Count',
                            'cell_pct': 'Cell Percentage',
                            'category_code': color_titles.get(color_col, color_col)
                        },
                        hover_data=['cell_count', 'cell_pct', color_col]
                    )
            else:
                # For continuous variables
                fig = px.scatter(
                    merged_df,
                    x='cell_count', 
                    y='cell_pct',
                    color=color_col,
                    color_continuous_scale='Viridis',
                    title=f"Cell Count vs Cell Pct (R² = {r_squared:.4f})<br>Colored by {color_titles.get(color_col, color_col)}",
                    labels={
                        'cell_count': 'Cell Count',
                        'cell_pct': 'Cell Percentage',
                        color_col: color_titles.get(color_col, color_col)
                    },
                    hover_data=['cell_count', 'cell_pct', color_col]
                )
            
            # Add trendline
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name=f'Trendline (R² = {r_squared:.4f})'
                )
            )
            
            # Update layout to remove redundant title on colorbar/legend
            fig.update_layout(
                width=1000,
                height=700,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    title=""  # Remove legend title (redundant with plot title)
                ),
                margin=dict(l=10, r=10, t=80, b=10),
                coloraxis_colorbar=dict(title="")  # Remove colorbar title for continuous variables
            )
            
            # Save each figure to its own HTML file
            filename = f'cell_correlation_{color_col}.html'
            file_path = dir_paths['visualizations']['correlation'] / filename
            fig.write_html(
                file_path,
                include_plotlyjs='cdn',
                full_html=True,
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'eraseshape']
                }
            )
            html_files.append((color_titles.get(color_col, color_col), filename))
            log_info(f"Saved plot colored by {color_col} to: {file_path}")
        
        # Create a simple HTML index file with links to each plot
        index_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cell Correlation Plots</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .plot-list { list-style-type: none; padding: 0; }
                .plot-list li { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .plot-list a { text-decoration: none; color: #0366d6; font-weight: bold; }
                .plot-list a:hover { text-decoration: underline; }
                .description { margin-top: 5px; color: #666; }
            </style>
        </head>
        <body>
            <h1>Cell Count vs Cell Percentage Correlation Plots</h1>
            <p>Select a coloring scheme to view the interactive plot:</p>
            <ul class="plot-list">
        """
        
        # Add links to each plot
        for title, filename in html_files:
            index_html += f"""
                <li>
                    <a href="{filename}" target="_blank">Colored by {title}</a>
                    <div class="description">Interactive scatter plot with R² = {r_squared:.4f}</div>
                </li>
            """
        
        index_html += """
            </ul>
        </body>
        </html>
        """
        
        # Save the index file
        index_path = dir_paths['visualizations']['correlation'] / 'cell_correlation_interactive.html'
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        log_info(f"Saved interactive cell correlation plot index to: {index_path}")
        
    except ImportError as e:
        log_info(f"Error: {e}. Make sure plotly is installed: pip install plotly")
    except Exception as e:
        log_info(f"Error creating interactive cell correlation plot: {e}")
        import traceback
        log_info(traceback.format_exc())