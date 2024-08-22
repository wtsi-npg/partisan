#!/bin/bash

set -e

export PYENV_ROOT="/app/.pyenv"

# Put PYENV first to ensure we use the pyenv-installed Python
export PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:/bin:/usr/bin:/usr/local/bin"
export PYTHONUNBUFFERED=1
export PYTHONPATH=""

exec "$@"
