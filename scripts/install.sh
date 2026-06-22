#!/usr/bin/env bash
#
# pmc — one-line install for the terminal agent.
#
# Goal: a friend types one command, and a few minutes later they're
# talking to their agent in their own terminal.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/alialmef/thepersonalmodelcompany/main/scripts/install.sh | bash
#
# What this does:
#   1. Ensures uv (the Python package manager) is installed
#   2. Clones (or updates) the repo into ~/.pmc/src
#   3. Installs the `pmc` CLI into the user's uv tool store
#   4. Runs `pmc configure` to set up provider + key
#   5. Tells them to run `pmc chat`
#
# What this does NOT do (yet):
#   - It does NOT install the Rust extractors. Those live in the Mac
#     app and need notarization + FDA. Until the Mac app ships, the
#     CLI runs against an empty graph and the agent will say so.

set -euo pipefail

PMC_DIR="${PMC_DIR:-$HOME/.pmc/src}"
REPO_URL="${PMC_REPO:-https://github.com/alialmef/thepersonalmodelcompany.git}"
BRANCH="${PMC_BRANCH:-main}"

say() { printf '\033[1;36m%s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$1"; }
err() { printf '\033[1;31m%s\033[0m\n' "$1" >&2; }

# ---------------------------------------------------------------------------
# 1. uv
# ---------------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    say "installing uv (python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin or ~/.cargo/bin depending on platform
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    err "uv install appears to have failed. PATH may not have updated."
    err "open a new shell and re-run this installer."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. clone or update
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$PMC_DIR")"

if [ -d "$PMC_DIR/.git" ]; then
    say "updating pmc at $PMC_DIR..."
    git -C "$PMC_DIR" fetch --quiet origin "$BRANCH"
    git -C "$PMC_DIR" checkout --quiet "$BRANCH"
    git -C "$PMC_DIR" reset --hard --quiet "origin/$BRANCH"
else
    say "cloning pmc to $PMC_DIR..."
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$PMC_DIR"
fi

# ---------------------------------------------------------------------------
# 3. install
# ---------------------------------------------------------------------------

say "installing the pmc CLI via uv..."
cd "$PMC_DIR"
# `uv tool install --reinstall` makes the `pmc` command available on
# PATH (uv puts it in ~/.local/bin), isolated from your system python.
uv tool install --reinstall --quiet "$PMC_DIR"

if ! command -v pmc >/dev/null 2>&1; then
    warn "the pmc binary isn't on your PATH yet."
    warn "add this line to your shell rc and re-open the terminal:"
    warn "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. build the Rust extractor (pmc-ingest)
# ---------------------------------------------------------------------------

# pmc-ingest is the binary that walks the user's Mac data sources and
# writes the personal knowledge graph. Building it here means
# `pmc connect` works the first time without the user having to
# manually invoke cargo. Requires Rust on the machine; we install it
# if it's missing.

if ! command -v cargo >/dev/null 2>&1; then
    say "installing Rust toolchain (one-time)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # rustup installs to ~/.cargo/bin
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if command -v cargo >/dev/null 2>&1; then
    say "building pmc-ingest (Rust extractor)..."
    if (cd "$PMC_DIR/desktop" && cargo build --example pmc_ingest --release --quiet); then
        say "✓ pmc-ingest built."
    else
        warn "cargo build failed. you can retry later with:"
        warn "    cd $PMC_DIR/desktop && cargo build --example pmc_ingest --release"
    fi
else
    warn "cargo (Rust) couldn't be installed automatically."
    warn "to enable graph extraction, install Rust manually:"
    warn "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

# ---------------------------------------------------------------------------
# 5. ready
# ---------------------------------------------------------------------------

cat <<'EOF'

installed.

next:
    pmc doctor      # check that everything is set up correctly
    pmc configure   # set your model provider + api key
    pmc connect     # let pmc read your Mac (requires Full Disk Access)
    pmc sandbox     # build the portrait
    pmc install-mcp claude    # plug into Claude Desktop (or cursor / continue)

then restart your agent and ask it "what do you know about me?"

EOF
