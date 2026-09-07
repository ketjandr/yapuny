#!/bin/sh
# Yapuny worker installer (macOS / Linux).
set -e

REPO="ketjandr/yapuny"
REPO_TARBALL="https://github.com/${REPO}/archive/refs/heads/main.tar.gz"
STATE="$HOME/.yapuny"
SHA_FILE="$STATE/installed_sha"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# uv installs to ~/.local/bin; make sure this shell sees it and the `yapuny` command it creates
export PATH="$HOME/.local/bin:$PATH"

# latest commit on main (empty if offline / rate-limited); the sha we installed last time
latest=$(curl -fsSL "https://api.github.com/repos/${REPO}/commits/main" 2>/dev/null \
  | grep -m1 '"sha"' | sed -E 's/.*"sha": *"([0-9a-f]+)".*/\1/')
current=""
[ -f "$SHA_FILE" ] && current=$(cat "$SHA_FILE")
short=$(printf '%.7s' "$latest")

# already current: skip the reinstall and just start (fast path)
if [ -n "$latest" ] && [ "$latest" = "$current" ] && command -v yapuny >/dev/null 2>&1; then
  echo "Worker already up to date (${short}). Starting..."
  exec yapuny
fi

if [ -n "$current" ] && [ -n "$latest" ]; then
  echo "New worker version available (${short}) - updating..."
else
  echo "Installing the Yapuny worker..."
fi

# the GPU fusion kernels (triton) only have reliable wheels on Linux; skip that extra on macOS
EXTRA=""
[ "$(uname -s)" = "Linux" ] && EXTRA="[gpu]"
# --force overwrites the existing tool; --refresh re-fetches the (mutable) main tarball
UV_TORCH_BACKEND=auto uv tool install --force --refresh --python 3.11 "yapuny${EXTRA} @ ${REPO_TARBALL}"

# record what we just installed so the next run can tell update vs up-to-date
mkdir -p "$STATE"
[ -n "$latest" ] && printf '%s' "$latest" > "$SHA_FILE"

echo ""
echo "Done. Starting the worker on http://localhost:8000"
echo "Next time, just run:  yapuny"
echo ""
exec yapuny
