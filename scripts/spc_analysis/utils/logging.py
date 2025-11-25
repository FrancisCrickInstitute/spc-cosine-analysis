# Logging

import datetime

def get_timestamp():
    """Return current timestamp as string for logging."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_info(message):
    """Print timestamped message for logging."""
    print(f"[{get_timestamp()}] {message}")

def log_section(title):
    """Print section header for better log readability."""
    border = "=" * 80
    print(f"\n{border}")
    print(f"[{get_timestamp()}] {title}")
    print(f"{border}\n")

def get_memory_usage_str():
    """Return current memory usage of the process."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return f"{memory_info.rss / (1024 ** 3):.2f} GB"
    except ImportError:
        return "psutil not installed"
    except Exception as e:
        return f"Error getting memory usage: {str(e)}"