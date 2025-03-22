import os

def get_env_var(var_name, default=None):
    """Get an environment variable or return a default value."""
    return os.getenv(var_name, default) 