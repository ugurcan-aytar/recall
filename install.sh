#!/usr/bin/env bash
#
# install.sh — one-line installer for recall.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ugurcan-aytar/recall/main/install.sh | bash
#
# Optional environment variables:
#   RECALL_VERSION  — version tag to install (default: latest)
#   RECALL_PREFIX   — install prefix (default: /usr/local on root, ~/.local otherwise)
#
set -euo pipefail

REPO="ugurcan-aytar/recall"
BINARY="recall"

# ---- detect platform ------------------------------------------------------
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64)  ARCH="x86_64"  ;;
  arm64|aarch64) ARCH="arm64"   ;;
  *) echo "install.sh: unsupported arch: $ARCH" >&2; exit 1 ;;
esac
case "$OS" in
  linux|darwin) ;;
  *) echo "install.sh: unsupported os: $OS" >&2; exit 1 ;;
esac

# ---- resolve version -------------------------------------------------------
VERSION="${RECALL_VERSION:-}"
if [[ -z "$VERSION" ]]; then
  VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
             | grep -oE '"tag_name": *"v[^"]+"' \
             | head -1 \
             | sed -E 's/.*"tag_name": *"(v[^"]+)".*/\1/')"
  if [[ -z "$VERSION" ]]; then
    echo "install.sh: could not resolve latest version (no releases yet?)" >&2
    echo "             set RECALL_VERSION=v0.X.Y and re-run." >&2
    exit 1
  fi
fi
VERSION_NO_V="${VERSION#v}"
echo "→ installing recall ${VERSION} for ${OS}/${ARCH}"

# ---- pick install prefix --------------------------------------------------
PREFIX="${RECALL_PREFIX:-}"
if [[ -z "$PREFIX" ]]; then
  if [[ "$EUID" -eq 0 ]]; then
    PREFIX="/usr/local"
  else
    PREFIX="$HOME/.local"
  fi
fi
BIN_DIR="$PREFIX/bin"
mkdir -p "$BIN_DIR"

# ---- download + verify -----------------------------------------------------
TARBALL="${BINARY}_${VERSION_NO_V}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/$REPO/releases/download/$VERSION/$TARBALL"
SUMS_URL="https://github.com/$REPO/releases/download/$VERSION/checksums.txt"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "→ downloading $TARBALL"
curl -fsSL --proto '=https' "$URL" -o "$TMPDIR/$TARBALL"

echo "→ verifying SHA-256"
curl -fsSL --proto '=https' "$SUMS_URL" -o "$TMPDIR/checksums.txt"
EXPECTED="$(grep "$TARBALL" "$TMPDIR/checksums.txt" | awk '{print $1}')"
if [[ -z "$EXPECTED" ]]; then
  echo "install.sh: checksum line for $TARBALL not found in checksums.txt" >&2
  exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$TMPDIR/$TARBALL" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "$TMPDIR/$TARBALL" | awk '{print $1}')"
else
  echo "install.sh: neither sha256sum nor shasum found — cannot verify" >&2
  exit 1
fi
if [[ "$EXPECTED" != "$ACTUAL" ]]; then
  echo "install.sh: checksum mismatch (expected $EXPECTED, got $ACTUAL)" >&2
  exit 1
fi
echo "  ok ($ACTUAL)"

# ---- install ---------------------------------------------------------------
tar -xzf "$TMPDIR/$TARBALL" -C "$TMPDIR"
mv "$TMPDIR/$BINARY" "$BIN_DIR/$BINARY"
chmod +x "$BIN_DIR/$BINARY"

echo "→ installed: $BIN_DIR/$BINARY"
"$BIN_DIR/$BINARY" version || true

# ---- PATH hint -------------------------------------------------------------
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo
     echo "  hint: $BIN_DIR is not in your PATH. Add this to your shell rc:"
     echo "        export PATH=\"$BIN_DIR:\$PATH\""
     ;;
esac
