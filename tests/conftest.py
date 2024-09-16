# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021, 2023, 2024 Genome Research Ltd. All rights reserved.
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
#
# From the pytest docs:
#
# "The conftest.py file serves as a means of providing fixtures for an entire
# directory. Fixtures defined in a conftest.py can be used by any test in that
# package without needing to import them (pytest will automatically discover
# them)."

from pathlib import PurePath

import pytest

from partisan.icommands import (
    add_specific_sql,
    have_admin,
    imkdir,
    iput,
    iquest,
    irm,
    mkgroup,
    remove_specific_sql,
    rmgroup,
)
from partisan.irods import AVU, Collection, DataObject

tests_have_admin = pytest.mark.skipif(
    not have_admin(), reason="tests do not have iRODS admin access"
)

# The following iRODS groups manage permissions for data belonging to the corresponding
# study.
STUDY_GROUPS = ["ss_study_01", "ss_study_02", "ss_study_03"]
# The following iRODS groups manage permissions for human contamination identified in
# data belonging to the corresponding study.
HUMAN_STUDY_GROUPS = [g + "_human" for g in STUDY_GROUPS]

TEST_GROUPS = STUDY_GROUPS + HUMAN_STUDY_GROUPS

TEST_SQL_STALE_REPLICATE = "setObjectReplStale"
TEST_SQL_INVALID_CHECKSUM = "setObjectChecksumInvalid"


def add_test_groups():
    if have_admin():
        for g in TEST_GROUPS:
            mkgroup(g)


def remove_test_groups():
    if have_admin():
        for g in TEST_GROUPS:
            rmgroup(g)


def add_sql_test_utilities():
    if have_admin():
        add_specific_sql(
            TEST_SQL_STALE_REPLICATE,
            "UPDATE r_data_main dm SET DATA_IS_DIRTY = 0 FROM r_coll_main cm "
            "WHERE dm.coll_id = cm.coll_id "
            "AND cm.COLL_NAME = ? "
            "AND dm.DATA_NAME = ? "
            "AND dm.DATA_REPL_NUM = ?",
        )
        add_specific_sql(
            TEST_SQL_INVALID_CHECKSUM,
            "UPDATE r_data_main dm SET DATA_CHECKSUM = 0 FROM r_coll_main cm "
            "WHERE dm.coll_id = cm.coll_id "
            "AND cm.COLL_NAME = ? "
            "AND dm.DATA_NAME = ? "
            "AND dm.DATA_REPL_NUM = ?",
        )


def remove_sql_test_utilities():
    if have_admin():
        remove_specific_sql(TEST_SQL_STALE_REPLICATE)
        remove_specific_sql(TEST_SQL_INVALID_CHECKSUM)


def add_rods_path(root_path: PurePath, tmp_path: PurePath) -> PurePath:
    parts = PurePath(*tmp_path.parts[1:])
    rods_path = root_path / parts
    imkdir(rods_path, make_parents=True)

    return rods_path


def set_replicate_invalid(obj: DataObject, replicate_num: int):
    iquest(
        "--sql",
        TEST_SQL_STALE_REPLICATE,
        obj.path.as_posix(),
        obj.name,
        str(replicate_num),
    )


def set_checksum_invalid(obj: DataObject, replicate_num: int):
    iquest(
        "--sql",
        TEST_SQL_INVALID_CHECKSUM,
        obj.path.as_posix(),
        obj.name,
        str(replicate_num),
    )


@pytest.fixture(scope="session")
def sql_test_utilities():
    try:
        add_sql_test_utilities()
        yield
    finally:
        remove_sql_test_utilities()


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
def full_collection(tmp_path):
    """A fixture providing a collection with some contents"""
    root_path = PurePath("/testZone/home/irods/test")
    rods_path = add_rods_path(root_path, tmp_path)

    iput("./tests/data/recursive/", rods_path, recurse=True)
    coll_path = rods_path / "recursive"

    try:
        add_test_groups()

        yield coll_path
    finally:
        remove_test_groups()
        irm(root_path, force=True, recurse=True)


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
def empty_data_object(tmp_path):
    """A fixture providing a collection containing a single data object containing
    no data."""
    root_path = PurePath("/testZone/home/irods/test")
    rods_path = add_rods_path(root_path, tmp_path)

    obj_path = rods_path / "empty.txt"
    iput("./tests/data/simple/data_object/empty.txt", obj_path)

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
def invalid_replica_data_object(tmp_path, sql_test_utilities):
    """A fixture providing a data object with one of its two replicas marked invalid."""
    root_path = PurePath("/testZone/home/irods/test")
    rods_path = add_rods_path(root_path, tmp_path)

    obj_path = rods_path / "invalid_replica.txt"
    iput("./tests/data/simple/data_object/lorem.txt", obj_path)
    set_replicate_invalid(DataObject(obj_path), 1)

    try:
        yield obj_path
    finally:
        irm(root_path, force=True, recurse=True)


@pytest.fixture(scope="function")
def invalid_checksum_data_object(tmp_path, sql_test_utilities):
    """A fixture providing a data object with one of its two replica's checksum
    changed."""
    root_path = PurePath("/testZone/home/irods/test")
    rods_path = add_rods_path(root_path, tmp_path)

    obj_path = rods_path / "invalid_checksum.txt"
    iput("./tests/data/simple/data_object/lorem.txt", obj_path)
    set_checksum_invalid(DataObject(obj_path), 1)

    try:
        yield obj_path
    finally:
        irm(root_path, force=True, recurse=True)


@pytest.fixture(scope="function")
def special_paths(tmp_path):
    """A fixture providing a collection of challengingly named paths which contain spaces
    and/or quotes."""
    root_path = PurePath("/testZone/home/irods/test")
    rods_path = add_rods_path(root_path, tmp_path)

    iput("./tests/data/special", rods_path, recurse=True)
    expt_root = rods_path / "special"

    try:
        yield expt_root
    finally:
        irm(root_path, force=True, recurse=True)


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
