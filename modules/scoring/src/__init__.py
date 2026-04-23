import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCORING_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
_scoring_root_str = str(_SCORING_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)
if _scoring_root_str not in sys.path:
    sys.path.insert(0, _scoring_root_str)
