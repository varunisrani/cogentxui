import json
import logging
import re

logger = logging.getLogger(__name__)

def clean_and_parse_json(text):
    """
    Extract and parse JSON from text, handling various edge cases.
    
    Args:
        text (str): Text that may contain JSON
        
    Returns:
        dict: Parsed JSON object or None if parsing fails
    """
    if not text:
        return None
        
    # If it's already a dict, return it
    if isinstance(text, dict):
        return text
        
    # Try to find JSON-like content in the string
    try:
        # First try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object using regex
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            try:
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Found JSON-like content but couldn't parse it: {json_str[:100]}...")
                
        # Try to fix common JSON formatting issues
        try:
            # Replace single quotes with double quotes
            fixed_text = text.replace("'", '"')
            # Fix unquoted property names
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            logger.warning("Couldn't parse JSON after fixing quotes")
            
        # Last resort: try to extract any JSON-like structure
        try:
            # Find the first { and last }
            start = text.find('{')
            end = text.rfind('}')
            
            if start >= 0 and end > start:
                json_str = text[start:end+1]
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", '"')
                # Fix unquoted property names
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            logger.error(f"All JSON parsing attempts failed for: {text[:100]}...")
            
    return None 