#!/usr/bin/env bash
# Fetch one archive from the v0.1.0 GitHub Release (plain curl, no gh auth) and extract it.
#   usage: scripts/fetch_release_asset.sh <asset> <dest_dir>   (asset name without .tar.gz)
# Prints the extracted directory, <dest_dir>/<asset>; skips the download if it already
# exists. The archive is downloaded and extracted in a temporary directory and moved into
# place only when complete, so an interrupted fetch leaves nothing behind.
#   e.g. scripts/fetch_release_asset.sh act_cpu_v7_abs outputs/release_v0.1.0
set -euo pipefail
ASSET=$1
DEST=$2
TAG=${TAG:-v0.1.0}
BASE=https://github.com/giridhar-shenoy/sim-demo-to-policy/releases/download/$TAG
if [[ ! -d $DEST/$ASSET ]]; then
  mkdir -p "$DEST"
  tmp=$(mktemp -d "$DEST/.fetch.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT
  echo "fetching $ASSET.tar.gz from release $TAG" >&2
  curl -fsSL -o "$tmp/$ASSET.tar.gz" "$BASE/$ASSET.tar.gz"
  tar -xzf "$tmp/$ASSET.tar.gz" -C "$tmp"
  [[ -d $tmp/$ASSET ]] || { echo "$ASSET.tar.gz has no top-level $ASSET/" >&2; exit 1; }
  mv "$tmp/$ASSET" "$DEST/$ASSET"
fi
echo "$DEST/$ASSET"
