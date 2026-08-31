from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

# Persistent locker for trained model packages, keyed by frontend-minted id.
# Layout: MODELS_DIR/<id>/{tokenizer.json, weights.pt, meta.json}
# The mount target is ops config (env), not code, so the same store works for a
# local dir, a Docker volume, or a RunPod network volume.
MODELS_DIR = Path(
    os.environ.get("YAPUNY_MODELS_DIR", Path(__file__).resolve().parent.parent / "models")
)

TOKENIZER_FILE = "tokenizer.json"
WEIGHTS_FILE = "weights.pt"
META_FILE = "meta.json"


@dataclass
class Package:
    id: str
    structure_hash: str
    tokenizer: Tokenizer
    weights: dict


def _dir(model_id: str) -> Path:
    return MODELS_DIR / model_id


def _write_meta(d: Path, meta: dict) -> None:
    # write to a temp file then atomically replace, so meta.json is never half-written
    tmp = d / (META_FILE + ".tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, d / META_FILE)


def save(model_id: str, tokenizer: Tokenizer, weights: dict, structure_hash: str) -> None:
    """Persist a package. meta is written LAST with status=trained as the commit marker.

    An in-progress write is marked status=writing first, so a crash mid-save (or a
    re-save that dies partway) leaves an invalid package that reads as absent.
    """
    d = _dir(model_id)
    d.mkdir(parents=True, exist_ok=True)

    _write_meta(d, {"structure_hash": structure_hash, "status": "writing"})
    tokenizer.save(str(d / TOKENIZER_FILE))
    torch.save(weights, d / WEIGHTS_FILE)
    _write_meta(d, {"structure_hash": structure_hash, "status": "trained"})


def read_meta(model_id: str) -> dict | None:
    """Return meta only for a committed package (status==trained), else None."""
    path = _dir(model_id) / META_FILE
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("status") != "trained":
        return None
    return meta


def exists(model_id: str) -> bool:
    return read_meta(model_id) is not None


def load(model_id: str) -> Package | None:
    """Load a committed package, or None if absent/incomplete."""
    meta = read_meta(model_id)
    if meta is None:
        return None

    d = _dir(model_id)
    tok_path = d / TOKENIZER_FILE
    w_path = d / WEIGHTS_FILE
    if not tok_path.exists() or not w_path.exists():
        return None

    tokenizer = Tokenizer.from_file(str(tok_path))
    weights = torch.load(w_path, map_location="cpu", weights_only=True)
    return Package(
        id=model_id,
        structure_hash=meta["structure_hash"],
        tokenizer=tokenizer,
        weights=weights,
    )


def list_ids() -> list[str]:
    """Ids of all committed packages the worker holds."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir() and exists(p.name))


def delete(model_id: str) -> bool:
    """Remove a package. Returns False if it did not exist."""
    d = _dir(model_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True
