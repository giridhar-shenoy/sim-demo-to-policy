"""Record smallbc demos: a thin wrapper around ``sdp.policies.smallbc.record`` (the
experiment's original copy is in git history, tag v0.1.0).

    python experiments/smallbc/record.py --out experiments/smallbc/data/demo5 --episodes 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402,F401  (sets MUJOCO_GL / SDP_MENAGERIE_DIR)

from sdp.policies.smallbc.record import *  # noqa: E402,F401,F403
from sdp.policies.smallbc.record import main  # noqa: E402

if __name__ == "__main__":
    main()
