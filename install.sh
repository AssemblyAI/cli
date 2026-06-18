#!/bin/bash

set -e # Exit on any error

# Canonical installer for the AssemblyAI CLI (`assembly`).
# Installs the app with uv (or pipx) if either is present, bootstrapping uv when
# neither is — then installs the optional system deps via Homebrew if available.

PACKAGE="git+https://github.com/AssemblyAI/cli.git"
PYTHON_VERSION="3.13"

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

# Install `assembly` as an isolated tool. Prefer uv (it manages an isolated
# Python for us), then fall back to an existing pipx, and only bootstrap uv if
# neither is already present.
install_with_uv() {
	# "$1" is the uv executable to invoke.
	"$1" tool install -U "$PACKAGE" --python "$PYTHON_VERSION"
}

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
	pipx install --force "$PACKAGE"
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
