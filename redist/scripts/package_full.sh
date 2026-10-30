#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${CYXWIZ_PACKAGING_PYTHON:-python3}" "$SCRIPT_DIR/package_release.py" full "$@"
