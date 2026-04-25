"""
Function definitions for LLM-assisted true LASA pairs generation.
"""

from src.proposer.prompt import SYSTEM_PROMPT
from os import environ
from pathlib import Path

# FIXME: adjust this as needed
MODEL = "claude-haiku-4-5-20251001"
RESULTS_DIR = Path("")
DELAY = 0.2

def propose_n(drug_name: str, n: int, api_key: str):
    """
    This function prompts the LLM to find potential LASA drugs given
    some drug name.
    
    FIXME: do this
    """
    for _ in range(0,n):
        print(SYSTEM_PROMPT)
    

def get_api_key(env_var_name: str):
    
    if (key := environ.get(key=env_var_name)) is None:
        raise KeyError(
            f"The environment variable {env_var_name} is required "
            "but is not found."
        )
    
    return key