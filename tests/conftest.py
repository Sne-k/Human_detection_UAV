"""
Shared setup for the test suite.

The suite is fast and self-contained: it needs no dataset, no trained
weights and no GPU, so it can run anywhere the code can. Tests that need
those artefacts live in the evaluation scripts, which measure; these check.
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
