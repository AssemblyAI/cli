#!/bin/bash

set -e # Exit on any error

# Canonical installer for the AssemblyAI CLI (`assembly`).
# Installs the app as a uv tool, bootstrapping uv first if it is missing.

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

# Homebrew also pulls in ffmpeg, portaudio, and cloudflared. The uv install does
# not, so detect any that are missing and print how to install them — without
# touching the system or invoking sudo on the user's behalf.
advise_system_deps() {
	local missing=()
	command -v ffmpeg >/dev/null 2>&1 || missing+=("ffmpeg")
	has_portaudio || missing+=("portaudio")
	command -v cloudflared >/dev/null 2>&1 || missing+=("cloudflared")

	[ ${#missing[@]} -eq 0 ] && return 0

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

if ! command -v uv &>/dev/null; then
	echo "uv is not installed. Installing..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	echo "uv installation complete!"
	echo ""

	if [ -x "$HOME/.local/bin/uv" ]; then
		"$HOME/.local/bin/uv" tool install -U "$PACKAGE" --python "$PYTHON_VERSION"
	else
		echo "Please restart your shell and run this script again"
		echo ""
		exit 0
	fi
else
	# `uv self update` errors out when uv was installed via an external package
	# manager (Homebrew, apt, …) — it can't replace a binary it doesn't own. That
	# is not fatal to us: a managed uv is already kept current by its manager, so
	# swallow the failure and proceed straight to installing the CLI.
	uv self update 2>/dev/null || true
	uv tool install -U "$PACKAGE" --python "$PYTHON_VERSION"
fi

advise_system_deps || true

echo ""
echo "For help and support, see the AssemblyAI CLI repository"
echo "https://github.com/AssemblyAI/cli"
echo ""
echo "Read the docs at https://www.assemblyai.com/docs"
echo ""
echo "The AssemblyAI CLI is installed!"
echo "Run 'assembly login' to sign in, then 'assembly transcribe --sample' to try it"
