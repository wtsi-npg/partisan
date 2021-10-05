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

# From the pytest docs:
#
# "The conftest.py file serves as a means of providing fixtures for an entire
# directory. Fixtures defined in a conftest.py can be used by any test in that
# package without needing to import them (pytest will automatically discover
# them)."

from pathlib import PurePath

import pytest

from partisan.icommands import have_admin, imkdir, iput, irm, mkgroup, rmgroup
from partisan.irods import AVU, Collection, DataObject

tests_have_admin = pytest.mark.skipif(
    not have_admin(), reason="tests do not have iRODS admin access"
)

TEST_GROUPS = ["ss_study_01", "ss_study_02", "ss_study_03"]


def add_test_groups():
    if have_admin():
        for g in TEST_GROUPS:
            mkgroup(g)


def remove_test_groups():
    if have_admin():
        for g in TEST_GROUPS:
            rmgroup(g)


def add_rods_path(root_path: PurePath, tmp_path: PurePath) -> PurePath:
    parts = PurePath(*tmp_path.parts[1:])
    rods_path = root_path / parts
    imkdir(rods_path, make_parents=True)

    return rods_path


@pytest.fixture(scope="function")
def simple_collection(tmp_path):
    """A fixture providing an empty collection"""
    root_path = PurePath("/testZone/home/irods/test")
    coll_path = add_rods_path(root_path, tmp_path)

    try:
        yield coll_path
    finally:
        irm(root_path, force=True, recurse=True)


@pytest.fixture(scope="function")
def annotated_collection(simple_collection):
    """A fixture providing an annotated, empty collection"""

    coll = Collection(simple_collection)
    coll.add_metadata(
        AVU("attr1", "value1"), AVU("attr2", "value2"), AVU("attr3", "value3")
    )

    try:
        yield simple_collection
    finally:
        irm(simple_collection, force=True, recurse=True)


@pytest.fixture(scope="function")
def simple_data_object(tmp_path):
    """A fixture providing a collection containing a single data object containing
    UTF-8 data."""
    root_path = PurePath("/testZone/home/irods/test")
    rods_path = add_rods_path(root_path, tmp_path)

    obj_path = rods_path / "lorem.txt"
    iput("./tests/data/simple/data_object/lorem.txt", obj_path)

    try:
        yield obj_path
    finally:
        irm(root_path, force=True, recurse=True)


@pytest.fixture(scope="function")
def annotated_data_object(simple_data_object):
    """A fixture providing a collection containing a single, annotated data object
    containing UTF-8 data."""

    obj = DataObject(simple_data_object)
    obj.add_metadata(
        AVU("attr1", "value1"), AVU("attr2", "value2"), AVU("attr3", "value3")
    )

    try:
        yield simple_data_object
    finally:
        irm(simple_data_object, force=True, recurse=True)


@pytest.fixture(scope="function")
def ont_gridion(tmp_path):
    """A fixture providing a set of files based on output from an ONT GridION
    instrument. This dataset provides an example of file and directory naming
    conventions. The file contents are dummy values."""
    root_path = PurePath("/testZone/home/irods/test")
    rods_path = add_rods_path(root_path, tmp_path)

    iput("./tests/data/ont/gridion", rods_path, recurse=True)
    expt_root = rods_path / "gridion"

    try:
        add_test_groups()

        yield expt_root
    finally:
        irm(root_path, force=True, recurse=True)
        remove_test_groups()
