"""Shared test configuration.

`tools/` holds repository-hygiene scripts rather than shipped code, so it is deliberately
not part of the installed package and not on the import path. The tests still have to
reach it — a gate nobody tests is a gate that silently stops working.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
