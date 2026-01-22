"""Configuration loading utilities for TacticAI."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML configuration with inheritance and overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Optional dict of parameter overrides (e.g., from CLI args)

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle inheritance via 'defaults' key
    if 'defaults' in config:
        base_configs = config.pop('defaults')
        if isinstance(base_configs, str):
            base_configs = [base_configs]

        merged = {}
        for base in base_configs:
            # Resolve path relative to current config
            base_path = config_path.parent / f"{base}.yaml"
            if not base_path.exists():
                # Try configs directory
                base_path = config_path.parent.parent / f"{base}.yaml"
            base_config = load_config(str(base_path))
            merged = deep_merge(merged, base_config)

        config = deep_merge(merged, config)

    # Apply CLI overrides
    if overrides:
        config = deep_merge(config, overrides)

    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override dict into base dict.

    Args:
        base: Base dictionary
        override: Dictionary to merge on top of base

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config: Dict, prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested config to dot-notation keys.

    Useful for W&B logging and CLI overrides.
    Example: {'model': {'hidden_dim': 128}} -> {'model.hidden_dim': 128}

    Args:
        config: Nested configuration dictionary
        prefix: Prefix for keys (used in recursion)

    Returns:
        Flattened dictionary with dot-notation keys
    """
    items = {}
    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_config(value, new_key))
        else:
            items[new_key] = value
    return items


def parse_cli_overrides(args: list) -> Dict[str, Any]:
    """
    Parse CLI arguments in key=value format to nested dict.

    Args:
        args: List of strings like ['model.hidden_dim=256', 'training.lr=0.001']

    Returns:
        Nested dictionary of overrides
    """
    overrides = {}
    for arg in args:
        if '=' not in arg:
            continue
        key, value = arg.split('=', 1)

        # Try to parse as number/bool
        try:
            value = eval(value)
        except:
            pass

        # Handle nested keys (model.hidden_dim -> {'model': {'hidden_dim': value}})
        keys = key.split('.')
        current = overrides
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return overrides
