# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021, 2022, 2023 Genome Research Ltd. All rights
# reserved.
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

import subprocess
from io import StringIO
from pathlib import PurePath
from typing import List, Union

from structlog import get_logger

from partisan.exception import RodsError

log = get_logger()


def mkgroup(name: str):
    cmd = ["iadmin", "mkgroup", name]
    _run(cmd)


def rmgroup(name: str):
    cmd = ["iadmin", "rmgroup", name]
    _run(cmd)


def iuserinfo(name: str = None) -> str:
    cmd = ["iuserinfo"]
    if name is not None:
        cmd.append(name)
    log.debug("Running command", cmd=cmd)

    completed = subprocess.run(cmd, capture_output=True)
    if completed.returncode == 0:
        return completed.stdout.decode("utf-8").strip()

    raise RodsError(completed.stderr.decode("utf-8").strip())


def imkdir(remote_path: Union[PurePath, str], make_parents=True):
    cmd = ["imkdir"]
    if make_parents:
        cmd.append("-p")

    cmd.append(remote_path)
    _run(cmd)


def iget(
    remote_path: Union[PurePath, str],
    local_path: Union[PurePath, str],
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
    local_path: Union[PurePath, str],
    remote_path: Union[PurePath, str],
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


def irm(remote_path: Union[PurePath, str], force=False, recurse=False):
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


def itrim(remote_path: Union[PurePath, str], replica_num: int, min_replicas=2):
    cmd = ["itrim", "-n", f"{replica_num}", "-N", f"{min_replicas}", remote_path]
    _run(cmd)


def icp(
    from_path: Union[PurePath, str],
    to_path: Union[PurePath, str],
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
        return completed.stdout.decode("utf-8").strip()

    # As of iRODS 4.2.12, iquest behaves from like commands such as grep and
    # exits with a special error code of 1 to indicate a successful query that
    # returned an empty result.
    if completed.returncode == 1:
        return ""

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


def _run(cmd: List[str]):
    log.debug("Running command", cmd=cmd)

    completed = subprocess.run(cmd, capture_output=True)
    if completed.returncode == 0:
        return

    raise RodsError(completed.stderr.decode("utf-8").strip())
