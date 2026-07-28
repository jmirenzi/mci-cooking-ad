import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = REPO_ROOT / "dataset" / "processed" / "breakfast" / "hsmm_params.npz"

pytestmark = pytest.mark.skipif(
    not PARAMS_PATH.exists(), reason="dataset/ is gitignored; no fitted breakfast checkpoint present"
)


def test_rollout_demo_healthy_scenario_exits_zero():
    # Invoke via `uv run`, matching this repo's own convention (coding_spec.md), rather than
    # sys.executable directly -- uv run resolves the project's editable install itself instead
    # of relying on whatever happens to be cached in site-packages at call time.
    result = subprocess.run(
        ["uv", "run", "python", "run_rollout_demo.py", "--scenario", "healthy", "--no-calibrate"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
