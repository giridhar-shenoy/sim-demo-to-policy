"""Train smallbc: a thin wrapper around ``sdp.policies.smallbc.train`` (the experiment's
original copy is in git history, tag v0.1.0).

    python experiments/smallbc/train.py --data experiments/smallbc/data/demo5 \
        --out experiments/smallbc/runs/smoke --steps 100
"""

from sdp.policies.smallbc.train import *  # noqa: F401,F403
from sdp.policies.smallbc.train import main

if __name__ == "__main__":
    main()
