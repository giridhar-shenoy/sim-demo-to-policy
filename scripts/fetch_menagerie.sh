#!/usr/bin/env bash
# Fetch the Franka Panda model from mujoco_menagerie at a pinned commit.
# The meshes are ~33 MB, so they are fetched rather than vendored.
#   usage: scripts/fetch_menagerie.sh [dest]   (default: third_party/mujoco_menagerie)
# Point SDP_MENAGERIE_DIR at the result (or at any existing checkout of the same commit).
set -euo pipefail
COMMIT=f3b6013629c7b94f8ee93fd977b161fa8350f9bc
DEST=${1:-third_party/mujoco_menagerie}
if [ -f "$DEST/franka_emika_panda/panda.xml" ]; then
  echo "menagerie already present at $DEST"
  exit 0
fi
mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git -C "$DEST" sparse-checkout set franka_emika_panda
git -C "$DEST" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$DEST" checkout -q FETCH_HEAD
echo "menagerie $COMMIT (franka_emika_panda) at $DEST"
