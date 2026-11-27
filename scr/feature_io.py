"""Save feature IDs and extraction metadata together; never infer row order."""
from pathlib import Path
import importlib.metadata
import numpy as np
try:
    from .save_results import write_json
except ImportError:
    from save_results import write_json


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def save_features(output, ids, features, metadata):
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Feature output must end in .npz")
    if output.exists():
        raise FileExistsError(output)
    ids, features = [str(x).strip() for x in ids], np.asarray(features, dtype=np.float32)
    if not ids or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError("Feature IDs must be unique and nonempty")
    if features.ndim != 2 or features.shape[0] != len(ids) or not features.shape[1] or not np.isfinite(features).all():
        raise ValueError("Feature matrix shape/values do not match IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(ids, dtype=str), features=features)
    write_json(str(output) + ".metadata.json", {**metadata, "rows": len(ids), "dimension": features.shape[1],
                                                "ids_stored_with_features": True})
