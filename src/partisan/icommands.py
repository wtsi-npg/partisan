# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021 Genome Research Ltd. All rights reserved.
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
from pathlib import PurePath
from typing import List, Union

from partisan.exception import RodsError
from partisan.irods import log


def mkgroup(name: str):
    cmd = ["iadmin", "mkgroup", name]
    _run(cmd)


def rmgroup(name: str):
    cmd = ["iadmin", "rmgroup", name]
    _run(cmd)


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
    _run(cmd)


def have_admin() -> bool:
    """Returns true if the current user has iRODS admin capability."""
    cmd = ["iadmin", "lu"]
    try:
        _run(cmd)
        return True
    except RodsError:
        return False


def _run(cmd: List[str]):
    log.debug("Running command", cmd=cmd)

    completed = subprocess.run(cmd, capture_output=True)
    if completed.returncode == 0:
        return

    raise RodsError(completed.stderr.decode("utf-8").strip(), 0)
