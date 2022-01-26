#!/bin/bash -il

set -eo pipefail
set -x

trap cleanup EXIT INT TERM

cleanup() {
    local exit_code=$?

    [ -d "$WORK_DIR" ] && rm -rf "$WORK_DIR"
    exit $exit_code
}

make_temp_dir() {
    echo $(mktemp -d /tmp/$(basename -- $0).XXXXXXXXXX)
}

# Conda parameters
OS=${OS:-"Linux"}
ARCH=${ARCH:-"x86_64"}
PYTHON_VERSION=${PYTHON_VERSION:-"39"}
CONDA_VERSION=${CONDA_VERSION:-"4.10.3"}
CONDA_SHA256=${CONDA_SHA256:-"1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f"}
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py${PYTHON_VERSION}_${CONDA_VERSION}-${OS}-${ARCH}.sh"

CONDA_INSTALL_DIR=${CONDA_INSTALL_DIR:-/opt/conda}
WORK_DIR=$(make_temp_dir)
curl -sSL $CONDA_URL > "$WORK_DIR/miniconda.sh"
sha256sum "$WORK_DIR/miniconda.sh" | grep $CONDA_SHA256

/bin/sh "$WORK_DIR/miniconda.sh" -b -p "$CONDA_INSTALL_DIR"

source $CONDA_INSTALL_DIR/etc/profile.d/conda.sh

conda activate

conda config --set auto_update_conda False
conda config --set ssl_verify True
conda config --set show_channel_urls True

conda clean -y --all
