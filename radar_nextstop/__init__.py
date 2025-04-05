import sys
from pathlib import Path

# Get the current file's directory and move up to the project's root
ROOT_DIR = Path(__file__).resolve().parents[1]

# Add the root directory to sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# You can also dynamically import submodules if needed
__all__ = []
