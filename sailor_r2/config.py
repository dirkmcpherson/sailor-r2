"""Config loading and merging for sailor-r2.

Loads SAILOR's YAML config format and adds backward-compatible attribute
access so both SAILOR pipeline code and r2dreamer model code can consume
the same config object.
"""

import copy
from pathlib import Path

from ruamel.yaml import YAML


class AttrDict(dict):
    """Dict subclass that supports attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def copy(self):
        return AttrDict(super().copy())


def _to_attrdict(d):
    """Recursively convert dicts to AttrDict."""
    if isinstance(d, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_attrdict(v) for v in d]
    return d


def load_config(config_path, overrides=None):
    """Load a YAML config file and apply CLI overrides.

    Args:
        config_path: Path to YAML config.
        overrides: list of ``"key.nested=value"`` strings.

    Returns:
        AttrDict config object.
    """
    yaml = YAML()
    with open(config_path) as f:
        raw = yaml.load(f)
    cfg = _to_attrdict(raw)
    if overrides:
        for ov in overrides:
            key, val = ov.split("=", 1)
            _set_nested(cfg, key, _parse_value(val))
    return cfg


def _set_nested(d, dotted_key, value):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, AttrDict())
    d[keys[-1]] = value


def _parse_value(s):
    """Try to parse a CLI override value as int/float/bool/string."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "null" or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def merge_suite_config(base, suite_name, suite_configs):
    """Merge a suite-specific config section into the base config.

    Args:
        base: base AttrDict config.
        suite_name: e.g. ``"robomimic"`` or ``"maniskill"``.
        suite_configs: dict mapping suite names to override dicts.

    Returns:
        merged AttrDict.
    """
    cfg = copy.deepcopy(base)
    if suite_name in suite_configs:
        suite = _to_attrdict(suite_configs[suite_name])
        _deep_update(cfg, suite)
    return cfg


def _deep_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
