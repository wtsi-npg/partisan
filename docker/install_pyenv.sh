#!/bin/bash

set -ex

PYENV_RELEASE_VERSION=${PYENV_RELEASE_VERSION:="2.4.16"}
export PYENV_GIT_TAG="v${PYENV_RELEASE_VERSION}"

PYENV_ROOT=${PYENV_ROOT:-"$HOME/.pyenv"}
export PATH="$PYENV_ROOT/bin:$PATH"

PYENV_SHA256="a1ad63c22842dce498b441551e2f83ede3e3b6ebb33f62013607bba424683191"
curl -sSL -O https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer
sha256sum ./pyenv-installer | grep "$PYENV_SHA256"
/bin/bash ./pyenv-installer
rm ./pyenv-installer
