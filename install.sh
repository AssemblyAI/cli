#!/bin/bash

set -e # Exit on any error

# Canonical installer for the AssemblyAI CLI (`assembly`).
#
# Default: installs the latest published code as an isolated tool with uv (or
#   pipx), bootstrapping uv when neither is present.
# Dev mode (--install-method git / --dev): clones the repo (or reuses the
#   checkout you run this from) and installs it editable (`uv tool install -e .`),
#   so local source edits take effect without reinstalling.
# Either way it then installs the optional system deps via Homebrew if available.
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/AssemblyAI/cli/main/install.sh | bash
#   ./install.sh --dev                                   # editable, from a clone
#   curl -LsSf .../install.sh | bash -s -- --install-method git

REPO_URL="https://github.com/AssemblyAI/cli.git"
PACKAGE="git+${REPO_URL}"
PYTHON_VERSION="3.13"

# Install method: "release" (default, publish-style) or "git" (editable clone).
# Overridable by env or the flags parsed below.
INSTALL_METHOD="${AAI_INSTALL_METHOD:-release}"
GIT_DIR="${AAI_GIT_DIR:-$HOME/.local/share/assembly-cli}"
# Passed to the installer as `-e` only in dev mode (empty array otherwise).
EDITABLE=()

usage() {
	cat <<'EOF'
Install the AssemblyAI CLI (assembly).

Usage: install.sh [options]

Options:
  --install-method <release|git>  release (default): install the latest
                                  published code. git: clone the repo and
                                  install it editable (development mode).
  --dev, -e, --editable, --git    Shortcut for --install-method git.
  --release                       Shortcut for --install-method release.
  --dir <path>                    Clone directory for dev mode
                                  (default: ~/.local/share/assembly-cli).
  -h, --help                      Show this help.

Environment:
  AAI_INSTALL_METHOD=release|git  Same as --install-method.
  AAI_GIT_DIR=<path>              Same as --dir.
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
	--install-method | --method)
		[ $# -ge 2 ] || {
			echo "Missing value for $1" >&2
			exit 2
		}
		INSTALL_METHOD="$2"
		shift
		;;
	--dev | -e | --editable | --git) INSTALL_METHOD="git" ;;
	--release | --published) INSTALL_METHOD="release" ;;
	--dir | --git-dir)
		[ $# -ge 2 ] || {
			echo "Missing value for $1" >&2
			exit 2
		}
		GIT_DIR="$2"
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Unknown option: $1" >&2
		usage >&2
		exit 2
		;;
	esac
	shift
done

case "$INSTALL_METHOD" in
release | git) ;;
*)
	echo "Invalid --install-method: $INSTALL_METHOD (use 'release' or 'git')" >&2
	exit 2
	;;
esac

# Best-effort check for the PortAudio shared library (no `command` to probe, so
# look via pkg-config, the dynamic linker cache, then well-known lib paths).
has_portaudio() {
	if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists portaudio-2.0 2>/dev/null; then
		return 0
	fi
	local f
	case "$(uname -s)" in
	Darwin)
		for f in /opt/homebrew/lib/libportaudio*.dylib /usr/local/lib/libportaudio*.dylib; do
			[ -e "$f" ] && return 0
		done
		;;
	Linux)
		if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q libportaudio; then
			return 0
		fi
		for f in /usr/lib/libportaudio.so* /usr/lib/*/libportaudio.so*; do
			[ -e "$f" ] && return 0
		done
		;;
	esac
	return 1
}

# Populate MISSING_DEPS with the optional system deps not already on the system.
MISSING_DEPS=()
detect_missing_deps() {
	MISSING_DEPS=()
	command -v ffmpeg >/dev/null 2>&1 || MISSING_DEPS+=("ffmpeg")
	has_portaudio || MISSING_DEPS+=("portaudio")
	command -v cloudflared >/dev/null 2>&1 || MISSING_DEPS+=("cloudflared")
}

# Homebrew also pulls in ffmpeg, portaudio, and cloudflared. The uv/pipx installs
# do not, so detect any that are missing. If Homebrew is available we install the
# ones it actually carries (brew needs no sudo); for anything left we print how to
# install it — without touching the system or invoking sudo on the user's behalf.
install_system_deps() {
	detect_missing_deps
	[ ${#MISSING_DEPS[@]} -eq 0 ] && return 0

	if command -v brew >/dev/null 2>&1; then
		# Only ask Homebrew for formulae it actually has, so an unavailable one
		# can't fail the whole batch; `brew info` exits non-zero for unknown names.
		local brew_pkgs=() dep
		for dep in "${MISSING_DEPS[@]}"; do
			brew info --formula "$dep" >/dev/null 2>&1 && brew_pkgs+=("$dep")
		done
		if [ ${#brew_pkgs[@]} -gt 0 ]; then
			echo ""
			echo "Installing optional system dependencies with Homebrew: ${brew_pkgs[*]}"
			brew install "${brew_pkgs[@]}" || true
		fi
		# Re-detect so we only advise on whatever brew couldn't provide.
		detect_missing_deps
		[ ${#MISSING_DEPS[@]} -eq 0 ] && return 0
	fi

	local missing=("${MISSING_DEPS[@]}")
	echo ""
	echo "Optional system dependencies are missing: ${missing[*]}"
	echo "(core 'assembly transcribe' works without them)"
	echo "  - ffmpeg:      decode non-WAV / URL audio"
	echo "  - portaudio:   microphone capture for stream / dictate / agent"
	echo "  - cloudflared: public tunnel for 'assembly share'"
	echo ""
	echo "Install them with:"

	# Split into packages a system package manager carries vs. cloudflared,
	# which needs Cloudflare's own repo on Linux.
	local pkgs=()
	local need_cloudflared=0
	local dep
	for dep in "${missing[@]}"; do
		case "$dep" in
		cloudflared) need_cloudflared=1 ;;
		*) pkgs+=("$dep") ;;
		esac
	done

	case "$(uname -s)" in
	Darwin)
		echo "  brew install ${missing[*]}"
		;;
	Linux)
		if command -v apt-get >/dev/null 2>&1; then
			# PortAudio's runtime lib is libportaudio2 on Debian/Ubuntu.
			local apt_pkgs=("${pkgs[@]/portaudio/libportaudio2}")
			[ ${#apt_pkgs[@]} -gt 0 ] && echo "  sudo apt-get install ${apt_pkgs[*]}"
		elif command -v dnf >/dev/null 2>&1; then
			[ ${#pkgs[@]} -gt 0 ] && echo "  sudo dnf install ${pkgs[*]}"
		else
			[ ${#pkgs[@]} -gt 0 ] && echo "  install with your package manager: ${pkgs[*]}"
		fi
		if [ "$need_cloudflared" -eq 1 ]; then
			echo "  cloudflared: see https://pkg.cloudflare.com or"
			echo "    https://github.com/cloudflare/cloudflared/releases"
		fi
		;;
	*)
		echo "  ${missing[*]}"
		;;
	esac
}

# Resolve the source for a development (editable) install: reuse the checkout we
# are run from if it is the CLI repo, otherwise clone/update GIT_DIR. Sets PACKAGE
# to the local path and EDITABLE so the installer passes `-e`.
prepare_git_source() {
	if [ -f pyproject.toml ] && grep -q '^name = "aai-cli"' pyproject.toml 2>/dev/null; then
		PACKAGE="$(pwd)"
		echo "Development install from current checkout: $PACKAGE"
	else
		if ! command -v git >/dev/null 2>&1; then
			echo "Development install needs git to clone $REPO_URL" >&2
			exit 1
		fi
		if [ -d "$GIT_DIR/.git" ]; then
			echo "Updating existing clone at $GIT_DIR"
			git -C "$GIT_DIR" pull --ff-only
		else
			echo "Cloning $REPO_URL to $GIT_DIR"
			mkdir -p "$(dirname "$GIT_DIR")"
			git clone "$REPO_URL" "$GIT_DIR"
		fi
		PACKAGE="$GIT_DIR"
		echo "Development install from $PACKAGE"
	fi
	EDITABLE=(-e)
}

# Install `assembly` as an isolated tool. Prefer uv (it manages an isolated
# Python for us), then fall back to an existing pipx, and only bootstrap uv if
# neither is already present. EDITABLE is empty for a release install and `-e`
# for a dev install.
install_with_uv() {
	# "$1" is the uv executable to invoke.
	"$1" tool install -U "${EDITABLE[@]}" "$PACKAGE" --python "$PYTHON_VERSION"
}

[ "$INSTALL_METHOD" = "git" ] && prepare_git_source

if command -v uv >/dev/null 2>&1; then
	# `uv self update` errors out when uv was installed via an external package
	# manager (Homebrew, apt, …) — it can't replace a binary it doesn't own. That
	# is not fatal to us: a managed uv is already kept current by its manager, so
	# swallow the failure and proceed straight to installing the CLI.
	uv self update 2>/dev/null || true
	install_with_uv uv
elif command -v pipx >/dev/null 2>&1; then
	# --force makes a re-run upgrade in place: the git source's version may not
	# change between commits, so a plain `pipx install` would refuse as "already
	# installed" and never pick up new code.
	pipx install --force "${EDITABLE[@]}" "$PACKAGE"
else
	echo "Neither uv nor pipx found. Installing uv..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	echo "uv installation complete!"
	echo ""

	if [ -x "$HOME/.local/bin/uv" ]; then
		install_with_uv "$HOME/.local/bin/uv"
	else
		echo "Please restart your shell and run this script again"
		echo ""
		exit 0
	fi
fi

install_system_deps || true

echo ""
echo "For help and support, see the AssemblyAI CLI repository"
echo "https://github.com/AssemblyAI/cli"
echo ""
echo "Read the docs at https://www.assemblyai.com/docs"
echo ""
echo "The AssemblyAI CLI is installed!"
echo "Run 'assembly login' to sign in, then 'assembly transcribe --sample' to try it"
