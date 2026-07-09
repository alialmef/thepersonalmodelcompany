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
#   4. Installs the pmc-ingest extractor: downloads the prebuilt
#      universal macOS binary from GitHub Releases, falling back to a
#      from-source cargo build (needs Rust + Xcode Command Line Tools)
#   5. Tells them to run `pmc onboard`

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
# 4. the Rust extractor (pmc-ingest)
# ---------------------------------------------------------------------------

# pmc-ingest is the binary that walks the user's Mac data sources and
# writes the personal knowledge graph. Without it, `pmc connect` cannot
# do anything and the graph stays empty — so this step matters.
#
# Preferred path: download the prebuilt universal (arm64 + x86_64)
# binary from GitHub Releases. Fallback: build from source, which needs
# Rust and the Xcode Command Line Tools and takes several minutes.

PMC_BIN_DIR="$HOME/.pmc/bin"
INGEST_URL="${PMC_INGEST_URL:-https://github.com/alialmef/thepersonalmodelcompany/releases/latest/download/pmc-ingest-macos-universal.tar.gz}"

install_prebuilt_ingest() {
    say "downloading prebuilt pmc-ingest..."
    mkdir -p "$PMC_BIN_DIR"
    local tmp
    tmp="$(mktemp -d)"
    if curl -sfL "$INGEST_URL" -o "$tmp/pmc-ingest.tar.gz" \
        && tar -xzf "$tmp/pmc-ingest.tar.gz" -C "$tmp" pmc-ingest \
        && install -m 755 "$tmp/pmc-ingest" "$PMC_BIN_DIR/pmc-ingest" \
        && "$PMC_BIN_DIR/pmc-ingest" --help >/dev/null 2>&1; then
        rm -rf "$tmp"
        say "✓ pmc-ingest installed ($PMC_BIN_DIR/pmc-ingest)."
        return 0
    fi
    rm -rf "$tmp"
    return 1
}

build_ingest_from_source() {
    # Xcode Command Line Tools are required to link macOS frameworks.
    if ! xcode-select -p >/dev/null 2>&1; then
        err "building from source needs the Xcode Command Line Tools."
        err "install them with:    xcode-select --install"
        err "then re-run this installer."
        return 1
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        say "installing Rust toolchain (one-time)..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        err "cargo (Rust) couldn't be installed automatically."
        err "install Rust manually, then re-run this installer:"
        err "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        return 1
    fi
    say "building pmc-ingest from source (several minutes on first build)..."
    if (cd "$PMC_DIR/desktop" && cargo build --example pmc_ingest --release --quiet); then
        mkdir -p "$PMC_BIN_DIR"
        install -m 755 "$PMC_DIR/desktop/target/release/examples/pmc_ingest" \
            "$PMC_BIN_DIR/pmc-ingest"
        say "✓ pmc-ingest built and installed ($PMC_BIN_DIR/pmc-ingest)."
        return 0
    fi
    err "cargo build failed. you can retry later with:"
    err "    cd $PMC_DIR/desktop && cargo build --example pmc_ingest --release"
    return 1
}

if ! install_prebuilt_ingest; then
    warn "prebuilt download failed — falling back to a source build."
    if ! build_ingest_from_source; then
        err ""
        err "pmc-ingest is NOT installed. the pmc CLI will still work,"
        err "but 'pmc connect' cannot extract your data until it is."
        err "fix the issue above and re-run this installer."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 5. ready
# ---------------------------------------------------------------------------

cat <<'EOF'

installed.

next — one command walks you through everything:

    pmc onboard

it'll run doctor, take your API key, extract your data (after FDA),
build your portrait, and plug into Claude/Cursor/Continue. confirms
at each step. then restart your agent and ask it "what do you know
about me?"

(prefer step-by-step?  pmc doctor → pmc configure → pmc connect →
 pmc sandbox → pmc install-mcp claude)

EOF
