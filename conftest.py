"""Root conftest: ensure src/ is on sys.path for all tests."""

import sys
from pathlib import Path

# Insert src at the front so our packages (data, models, etc.) take
# precedence over any system packages with the same name.
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Clear stale module cache entries for our packages
for _mod in list(sys.modules):
    if _mod == "data" or _mod.startswith("data."):
        del sys.modules[_mod]
