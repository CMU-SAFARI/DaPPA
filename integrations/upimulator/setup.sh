#!/usr/bin/env bash
# Clone uPIMulator at the tested commit, add DaPPA support, and build it.
#
# Usage: ./setup.sh [DEST]     (default: integrations/upimulator/uPIMulator)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/uPIMulator}"
REPO="https://github.com/VIA-Research/uPIMulator.git"
COMMIT="870d916334e9ff0b190f555f951a9ec3c4257781"
SIM="$DEST/golang/uPIMulator"

if [[ ! -d "$DEST/.git" ]]; then
    git clone "$REPO" "$DEST"
fi
git -C "$DEST" checkout -q "$COMMIT"

# Patch existing uPIMulator sources and add the DAPPA benchmark.
git -C "$DEST" apply --directory=golang/uPIMulator "$HERE/upimulator.patch"
cp -r "$HERE/files/." "$SIM/"

(cd "$SIM" && python3 script/build.py)
echo "uPIMulator with DaPPA support built at $SIM/build/uPIMulator"
