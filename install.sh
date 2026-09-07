#!/bin/sh
# Yapuny worker installer (macOS / Linux)
set -e

REPO_TARBALL="https://github.com/ketjandr/yapuny/archive/refs/heads/main.tar.gz"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# uv installs to ~/.local/bin; make sure this shell sees it and the `yapuny` command it creates
export PATH="$HOME/.local/bin:$PATH"

# the GPU fusion kernels (triton) only have reliable wheels on Linux; skip that extra on macOS
EXTRA=""
if [ "$(uname -s)" = "Linux" ]; then EXTRA="[gpu]"; fi

echo "Installing the Yapuny worker (auto-detecting GPU)..."
# --force overwrites the existing tool; --refresh re-fetches the (mutable) main tarball so re-running
# this line actually pulls the latest code instead of reinstalling a cached build
UV_TORCH_BACKEND=auto uv tool install --force --refresh --python 3.11 "yapuny${EXTRA} @ ${REPO_TARBALL}"

echo ""
echo "Done. Starting the worker on http://localhost:8000"
echo "Next time, just run:  yapuny"
echo ""
exec yapuny
