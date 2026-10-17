from .model import *
# Local, pinned Triangle dependency shipped with this release.
from pathlib import Path as _Path
import sys as _sys
_vendor = str(_Path(__file__).resolve().parents[1] / "vendor/Triangle-Simulator")
if _vendor not in _sys.path:
    _sys.path.insert(0, _vendor)
