# src/utils/filename_utils.py

"""
Utility functions for filename handling and sanitization.
"""

import re

def sanitize_filename(name: str) -> str:
    """
    Sanitizes a string to be used as a safe filename by replacing or removing
    problematic characters.

    Parameters:
        name (str): The original filename string.

    Returns:
        str: The sanitized filename string.
    """
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    # Remove any characters that are not alphanumeric, underscores, or hyphens
    name = re.sub(r'[^\w\-]', '', name)
    
    # Optionally, convert to lowercase
    name = name.lower()
    
    return name
