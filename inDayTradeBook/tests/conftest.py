from __future__ import annotations

import sys
from pathlib import Path

FEATURE_ROOT = Path(__file__).resolve().parents[1]
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))
