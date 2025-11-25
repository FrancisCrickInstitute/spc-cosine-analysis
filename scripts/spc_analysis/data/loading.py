# Load data

import os
import yaml
from ..utils.logging import log_info

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    log_info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Validate required configuration parameters
    required_params = ['embeddings_file', 'output_dir']
    missing_params = [param for param in required_params if param not in config]
    
    if missing_params:
        raise ValueError(f"Missing required configuration parameters: {missing_params}")
    
    # Check file path existence
    file_paths = ['embeddings_file', 'metadata_file', 'harmony_file', 'pp_numbers_file']
    for path_key in file_paths:
        if path_key in config and config[path_key]:
            file_path = config[path_key]
            if not os.path.exists(file_path):
                log_info(f"Warning: File path does not exist: {file_path}")
    
    # Print configuration for debugging
    log_info("Configuration loaded:")
    # Replace it with this:
    # Replace it with this:
    for key, value in config.items():
        # Special handling for plate_definitions
        if key == 'plate_definitions' and isinstance(value, dict):
            log_info(f"  {key}:")
            plate_count = len(value)
            log_info(f"    Defined {plate_count} plates:")
            for plate_id, plate_info in value.items():
                log_info(f"    Plate {plate_id} - {plate_info.get('library', 'Unknown')} ({plate_info.get('type', 'Unknown')})")
        # Truncate long lists for better readability
        elif isinstance(value, list) and len(value) > 5:
            log_info(f"  {key}: [{', '.join(str(x) for x in value[:3])}... +{len(value)-3} more]")
        else:
            log_info(f"  {key}: {value}")
            log_info(f"  {key}: {value}")
    
    return config