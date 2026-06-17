#!/bin/bash

set -e # Exit on any error

# Canonical installer for the AssemblyAI CLI (`assembly`).
# Installs the app as a uv tool, bootstrapping uv first if it is missing.

PACKAGE="git+https://github.com/AssemblyAI/cli.git"
PYTHON_VERSION="3.13"

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
	uv self update
	uv tool install -U "$PACKAGE" --python "$PYTHON_VERSION"
fi

echo ""
echo "For help and support, see the AssemblyAI CLI repository"
echo "https://github.com/AssemblyAI/cli"
echo ""
echo "Read the docs at https://www.assemblyai.com/docs"
echo ""
echo "The AssemblyAI CLI is installed!"
echo "Run 'assembly login' to sign in, then 'assembly transcribe --sample' to try it"
