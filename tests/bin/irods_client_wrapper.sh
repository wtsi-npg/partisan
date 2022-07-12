#!/bin/bash

# This script may be used as a universal wrapper for iRODS clients hosted in a
# running container. E.g. a symlink of "ils" to this script will exec the "ils"
# client in the container. The container must already be running.

set -eo pipefail

IRODS_ENVIRONMENT_FILE=${IRODS_ENVIRONMENT_FILE:-"$HOME/.irods/irods_environment.json"}

# The default container name matches that created by the start_client_container.sh
# script
DOCKER_CONTAINER=${DOCKER_CONTAINER:-"irods-clients"}
CLIENT_USER_ID=${CLIENT_USER_ID:-$(id -u)}
CLIENT_USER=${CLIENT_USER:-"$USER"}
CLIENT_USER_HOME=${CLIENT_USER_HOME:-"$HOME"}

client=$(basename "$0")

# Provide a TTY for those applications that need it
tty_arg=""
if [ "$client" = "iinit" ]; then
    tty_arg="-t"
fi

docker exec -i $tty_arg \
       -u "$CLIENT_USER_ID" \
       -w "$PWD" \
       -e CLIENT_USER_ID="$CLIENT_USER_ID" \
       -e CLIENT_USER="$CLIENT_USER" \
       -e CLIENT_USER_HOME="$CLIENT_USER_HOME" \
       -e IRODS_ENVIRONMENT_FILE="$IRODS_ENVIRONMENT_FILE" \
       "$DOCKER_CONTAINER" "/opt/conda/envs/irods/bin/$client" "$@"
