# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021, 2022, 2023, 2024, 2025 Genome Research Ltd. All
# rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# @author Keith James <kdj@sanger.ac.uk>

import json
import os
import re
import shlex
import subprocess
from io import StringIO
from pathlib import Path, PurePath

from structlog import get_logger

from partisan.exception import RodsError

log = get_logger()


def mkgroup(name: str):
    cmd = ["iadmin", "mkgroup", name]
    _run(cmd)


def rmgroup(name: str):
    cmd = ["iadmin", "rmgroup", name]
    _run(cmd)


def mkuser(name: str):
    cmd = ["iadmin", "mkuser", name, "rodsuser"]
    _run(cmd)


def rmuser(name: str):
    cmd = ["iadmin", "rmuser", name]
    _run(cmd)


def group_exists(name: str) -> bool:
    info = iuserinfo(name)
    for line in info.splitlines():
        log.debug("Checking line", line=line)
        if re.match(r"type:\s+rodsgroup", line):
            log.debug("Group check", exists=True, name=name)
            return True

    log.debug("Group check", exists=False, name=name)
    return False


def user_exists(name: str) -> bool:
    info = iuserinfo(name)
    for line in info.splitlines():
        log.debug("Checking line", line=line)
        if re.match(r"type:\s+(rodsuser|groupadmin|rodsadmin)", line):
            log.debug("User check", exists=True, name=name)
            return True

    log.debug("User check", exists=False, name=name)
    return False


def iuserinfo(name: str = None) -> str:
    cmd = ["iuserinfo"]
    if name is not None:
        cmd.append(name)
    log.debug("Running command", cmd=cmd)

    completed = subprocess.run(cmd, capture_output=True)
    if completed.returncode == 0:
        return completed.stdout.decode("utf-8").strip()

    raise RodsError(completed.stderr.decode("utf-8").strip())


def imkdir(remote_path: PurePath | str, make_parents=True):
    cmd = ["imkdir"]
    if make_parents:
        cmd.append("-p")

    cmd.append(remote_path)
    _run(cmd)


def iinit():
    password = os.environ.get("IRODS_PASSWORD")
    if password is None or password == "":
        log.info(
            "Not authenticating with iRODS; no password specified by the "
            "IRODS_PASSWORD environment variable. Assuming the user is already "
            "authenticated."
        )
        return

    env_val = os.environ.get("IRODS_ENVIRONMENT_FILE")
    if env_val is None or env_val == "":
        log.info(
            "No iRODS environment file specified by the IRODS_ENVIRONMENT_FILE "
            "environment variable; using the default"
        )
        env_path = Path("~/.irods/irods_environment.json").expanduser().as_posix()
    else:
        env_path = Path(env_val).resolve().as_posix()

    log.info("Using iRODS environment file", env_path=env_path)

    with open(env_path) as f:
        env = json.load(f)
        if "irods_authentication_file" not in env:
            log.info(
                "No iRODS authentication file specified in the environment file; "
                "using the default"
            )
            auth_path = Path("~/.irods/.irodsA").expanduser().as_posix()
        else:
            auth_path = Path(env["irods_authentication_file"]).as_posix()

    if Path(auth_path).exists():
        log.info("Updating the existing iRODS auth file", auth_path=auth_path)
    else:
        log.info("Creating a new iRODS auth file", auth_path=auth_path)

    password = shlex.quote(password)
    cmd = ["/bin/sh", "-c", f"echo {password} | iinit"]
    _run(cmd)


def iget(
    remote_path: PurePath | str,
    local_path: PurePath | str,
    force=False,
    verify_checksum=True,
    recurse=False,
):
    cmd = ["iget"]
    if force:
        cmd.append("-f")
    if verify_checksum:
        cmd.append("-K")
    if recurse:
        cmd.append("-r")

    cmd.append(remote_path)
    cmd.append(local_path)
    _run(cmd)


def iput(
    local_path: PurePath | str,
    remote_path: PurePath | str,
    force=False,
    verify_checksum=True,
    recurse=False,
):
    cmd = ["iput"]
    if force:
        cmd.append("-f")
    if verify_checksum:
        cmd.append("-K")
    if recurse:
        cmd.append("-r")

    cmd.append(local_path)
    cmd.append(remote_path)
    _run(cmd)


def irm(remote_path: PurePath | str, force=False, recurse=False):
    cmd = ["irm"]
    if force:
        cmd.append("-f")
    if recurse:
        cmd.append("-r")

    cmd.append(remote_path)

    # Workaround for iRODS 4.2.7 which insists on raising errors with when the target is
    # absent, even when -f is used. This should be silent for a missing target.
    try:
        _run(cmd)
    except RodsError as re:
        if force:
            log.error(re.message, code=re.code)
        else:
            raise


def itrim(remote_path: PurePath | str, replica_num: int, min_replicas=2):
    cmd = ["itrim", "-n", f"{replica_num}", "-N", f"{min_replicas}", remote_path]
    _run(cmd)


def icp(
    from_path: PurePath | str,
    to_path: PurePath | str,
    force=False,
    verify_checksum=True,
    recurse=False,
):
    cmd = ["icp"]
    if force:
        cmd.append("-f")
    if verify_checksum:
        cmd.append("-K")
    if recurse:
        cmd.append("-r")

    cmd.append(from_path)
    cmd.append(to_path)
    _run(cmd)


def iquest(*args) -> str:
    """Run a non-paged iquest command with the specified arguments and return the
    result as a string. If the command returned no results, return an empty string."""
    cmd = ["iquest", "--no-page", *args]
    log.debug("Running command", cmd=cmd)

    completed = subprocess.run(cmd, capture_output=True)
    if completed.returncode == 0:
        lines = completed.stdout.decode("utf-8").strip().splitlines()
        # Remove logging that iquest can mix with its output
        if lines and lines[0].startswith("Zone is"):
            lines.pop(0)
        return "\n".join(lines)

    # As of iRODS 4.2.12, iquest behaves from like commands such as grep and
    # exits with a special error code of 1 to indicate a successful query that
    # returned an empty result.
    if completed.returncode == 1:
        return ""

    # For earlier versions of iRODS
    raise RodsError(completed.stderr.decode("utf-8").strip())


def has_specific_sql(alias) -> bool:
    """Return True if iRODS has a specific query installed under the alias."""
    existing = iquest("--sql", "ls")

    with StringIO(existing) as reader:
        return any(line.strip() == alias for line in reader)


def have_admin() -> bool:
    """Return true if the current user has iRODS admin capability."""
    cmd = ["iadmin", "lu"]
    try:
        _run(cmd)
        return True
    except RodsError:
        return False


def add_specific_sql(alias, sql):
    """Add a specific query under the alias, if the alias is not already used."""
    if not has_specific_sql(alias):
        cmd = ["iadmin", "asq", sql, alias]
        _run(cmd)


def remove_specific_sql(alias):
    """Remove any specific query under the alias, if present."""
    if has_specific_sql(alias):
        cmd = ["iadmin", "rsq", alias]
        _run(cmd)


def _run(cmd: list[str]):
    log.debug("Running command", cmd=cmd)

    completed = subprocess.run(cmd, capture_output=True)
    if completed.returncode == 0:
        return

    raise RodsError(completed.stderr.decode("utf-8").strip())
