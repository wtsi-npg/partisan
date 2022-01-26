#!/bin/bash

# --login

set -eo pipefail
set -x

source /opt/conda/etc/profile.d/conda.sh

conda activate irods

echo "Waiting for iRODS to become ready ..."
while true
do
    echo irods | iinit | grep -v USER_SOCK_CONNECT_ERR && break
    sleep 5
done
echo "iRODS is ready"

ienv
ilsresc
ils

exec "$@"
