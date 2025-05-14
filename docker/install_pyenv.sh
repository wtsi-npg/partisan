#!/bin/bash

set -ex

PYENV_RELEASE_VERSION=${PYENV_RELEASE_VERSION:="2.4.16"}
export PYENV_GIT_TAG="v${PYENV_RELEASE_VERSION}"

PYENV_ROOT=${PYENV_ROOT:-"$HOME/.pyenv"}
export PATH="$PYENV_ROOT/bin:$PATH"

PYENV_SHA256="4b0adf623a6205727163eb98610b6c5e63f23b99183948b874d867cd9b30ef13"
curl -sSL -O https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer
sha256sum ./pyenv-installer | grep "$PYENV_SHA256"
/bin/bash ./pyenv-installer
rm ./pyenv-installer
