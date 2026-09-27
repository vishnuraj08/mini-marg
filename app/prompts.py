"""
prompts.py — Loads the active prompt version from prompts.yaml

WHY A SEPARATE FILE?
  Any file that needs a prompt just calls get_prompt().
  The active version is controlled by prompts.yaml → active_version key.
  To switch from v2 back to v1: change one line in prompts.yaml. Done.
  No code changes, no redeployment needed.
"""

import yaml
from pathlib import Path

PROMPTS_PATH = Path(__file__).parent.parent / "prompts.yaml"


def get_prompt() -> dict:
    """
    Load prompts.yaml and return the active prompt version.
    
    Returns dict with keys: name, description, system, human
    """
    with open(PROMPTS_PATH) as f:
        data = yaml.safe_load(f)

    active = data["active_version"]          # e.g. "v2"
    prompt = data["prompts"][active]         # the v2 dict

    return {
        "version": active,
        "name": prompt["name"],
        "system": prompt["system"].strip(),
        "human": prompt["human"].strip(),
    }


def get_active_version() -> str:
    """Returns just the active version string e.g. 'v2'"""
    with open(PROMPTS_PATH) as f:
        data = yaml.safe_load(f)
    return data["active_version"]