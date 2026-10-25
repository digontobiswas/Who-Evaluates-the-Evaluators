"""Configuration loading.

Every experiment is described by one YAML file. A file may list other files under the
``extends`` key; those are merged first, in order, and the file itself is merged last. Values
are resolved with OmegaConf, so ``${paths.exp}`` refers to another key and
``${oc.env:WETE_ROOT}`` reads an environment variable. Nothing in the code base contains a
path: paths enter through environment variables set by ``scripts/env.sh``.

Command-line overrides use dotted keys, for example ``policy.seed=1 workers=8``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from omegaconf import DictConfig, OmegaConf

EXTENDS_KEY = "extends"

PathLike = Union[str, Path]


class ConfigError(RuntimeError):
    """Raised when a config file is missing, malformed or cyclic."""


def _load_file(path: Path, stack: List[Path]) -> DictConfig:
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    if path in stack:
        chain = " -> ".join(str(p) for p in stack + [path])
        raise ConfigError(f"Cyclic 'extends' chain: {chain}")

    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise ConfigError(f"Top level of {path} must be a mapping.")

    parents = loaded.pop(EXTENDS_KEY, None)
    if parents is None:
        parent_paths: List[str] = []
    elif isinstance(parents, str):
        parent_paths = [parents]
    else:
        parent_paths = [str(p) for p in parents]

    merged = OmegaConf.create({})
    for parent in parent_paths:
        parent_cfg = _load_file(path.parent / parent, stack + [path])
        merged = OmegaConf.merge(merged, parent_cfg)
    return OmegaConf.merge(merged, loaded)  # type: ignore[return-value]


def load_config(path: PathLike, overrides: Optional[Sequence[str]] = None) -> DictConfig:
    """Load ``path`` with its ``extends`` chain and apply dotted ``key=value`` overrides.

    Interpolations are left unresolved so that an override of one key propagates to every key
    that refers to it. Call :func:`resolve` before using the values.
    """
    cfg = _load_file(Path(path), stack=[])
    if overrides:
        bad = [o for o in overrides if "=" not in o]
        if bad:
            raise ConfigError(f"Overrides must look like key=value, got: {bad}")
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))  # type: ignore[assignment]
    return cfg


def resolve(cfg: DictConfig) -> Dict[str, Any]:
    """Return a plain dictionary with every interpolation resolved.

    Fails loudly on a missing value (``???``) or an unset environment variable, which is the
    behaviour we want on the cluster: a job should stop before it starts computing.
    """
    try:
        container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except Exception as exc:  # OmegaConf raises several unrelated exception types here.
        raise ConfigError(f"Could not resolve config: {exc}") from exc
    assert isinstance(container, dict)
    return container  # type: ignore[return-value]


def save(cfg: Union[DictConfig, Dict[str, Any]], path: PathLike) -> None:
    """Write a config next to the results it produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg, path)


def require(cfg: Dict[str, Any], dotted_keys: Iterable[str]) -> None:
    """Check that every dotted key exists and is not ``None``."""
    missing = []
    for dotted in dotted_keys:
        node: Any = cfg
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node or node[part] is None:
                missing.append(dotted)
                break
            node = node[part]
    if missing:
        raise ConfigError(f"Config is missing required keys: {sorted(missing)}")
