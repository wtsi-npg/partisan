# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021, 2023, 2024, 2025 Genome Research Ltd. All
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

import hashlib
import json
import os.path
import re
from datetime import datetime, timezone
from pathlib import Path, PurePath

import pytest
from pytest import mark as m

from partisan import irods
from partisan.exception import BatonError, RodsError
from partisan.irods import (
    AC,
    AVU,
    Baton,
    Collection,
    DataObject,
    Permission,
    Timestamp,
    User,
    current_user,
    make_rods_item,
    query_metadata,
    rods_path_type,
    rods_user,
    rods_users,
    server_version,
)

from src.partisan.irods import _calculate_file_checksum


@m.describe("Server")
class TestServer:
    @m.context("When queried for its version")
    @m.it("Returns the server version")
    def test_server_version(self):
        version = server_version()
        assert len(version) == 3
        assert version >= (4, 2, 0)
        assert version <= (6, 0, 0)


@m.describe("User")
class TestUser:
    @m.describe("Comparison")
    def test_compare_user_equal(self):
        name, id1, id2 = "user1", "1111", "2222"
        user_type = "rodsuser"
        this_zone, other_zone = "testZone", "otherZone"

        assert User(name, id1, user_type, this_zone) == User(
            name, id1, user_type, this_zone
        )

        assert User(name, id1, user_type, this_zone) != User(
            name, id1, user_type, other_zone
        )

        assert User(name, id1, user_type, this_zone) != User(
            name, id2, user_type, this_zone
        )

    @m.context("When a user is queried")
    @m.it("Is returned")
    def test_rods_user(self):
        user1 = rods_user("irods")
        assert user1.name == "irods"
        assert user1.type == "rodsadmin"
        assert user1.is_rodsadmin()
        assert not user1.is_group()
        assert not user1.is_rodsuser()

        user2 = rods_user("public")
        assert user2.name == "public"
        assert user2.type == "rodsgroup"
        assert user2.is_group()
        assert not user2.is_rodsuser()
        assert not user2.is_rodsadmin()

    @m.context("When the current user is queried")
    @m.it("Is returned")
    def test_current_user(self):
        user = current_user()
        assert user.name == "irods"
        assert user.type == "rodsadmin"
        assert user.zone == "testZone"

    @m.context("When a non-existent user is queried, such as one on another zone")
    @m.it("Returns None")
    def test_non_existent_user(self):
        user = rods_user("no_such_user")
        assert user is None

    @m.context("When a list of users is queried")
    @m.it("Is returned")
    def test_rods_users(self):
        # This gives different results on iRODS servers <4.3.0 and >=4.3.0
        if server_version() < (4, 3, 0):
            users = ["rodsadmin", "public", "irods"]
            groups = ["rodsadmin", "public"]
        else:
            users = ["public", "irods"]
            groups = ["public"]

        assert [user.name for user in rods_users()] == users
        assert [user.name for user in rods_users(user_type="rodsgroup")] == groups
        assert [user.name for user in rods_users(user_type="rodsadmin")] == ["irods"]
        assert [user.name for user in rods_users(user_type="rodsuser")] == []

        with pytest.raises(ValueError, match="Invalid user type"):
            rods_users(user_type="invalid type")


@m.describe("AC")
class TestAC:
    @m.describe("Comparison")
    def test_compare_acs_equal(self):
        assert AC("aaa", Permission.OWN, zone="x") == AC(
            "aaa", Permission.OWN, zone="x"
        )

        assert AC("aaa", Permission.OWN, zone="x") != AC(
            "aaa", Permission.READ, zone="x"
        )

        assert AC("aaa", Permission.OWN, zone="x") != AC(
            "bbb", Permission.OWN, zone="x"
        )

    def test_compare_acs_lt(self):
        assert AC("aaa", Permission.OWN) < AC("bbb", Permission.OWN)
        assert AC("aaa", Permission.OWN) < AC("aaa", Permission.READ)

        # Zoned AC sorts lowest
        assert AC("aaa", Permission.OWN, zone="x") < AC("aaa", Permission.OWN)
        assert AC("aaa", Permission.READ, zone="x") < AC(
            "aaa", Permission.OWN, zone="y"
        )

    def test_compare_acs_sort(self):
        acl = [
            AC("zzz", Permission.OWN, zone="x"),
            AC("aaa", Permission.WRITE, zone="x"),
            AC("aaa", Permission.READ, zone="x"),
            AC("zyy", Permission.READ, zone="x"),
            AC("zyy", Permission.OWN, zone="x"),
        ]
        acl.sort()
        assert acl == [
            AC("aaa", Permission.READ, zone="x"),
            AC("aaa", Permission.WRITE, zone="x"),
            AC("zyy", Permission.OWN, zone="x"),
            AC("zyy", Permission.READ, zone="x"),
            AC("zzz", Permission.OWN, zone="x"),
        ]

        acl = [
            AC("zyy", Permission.OWN),
            AC("aaa", Permission.WRITE),
            AC("zyy", Permission.READ, zone="x"),
            AC("aaa", Permission.READ),
            AC("zyy", Permission.OWN, zone="x"),
        ]
        acl.sort()
        assert acl == [
            AC("zyy", Permission.OWN, zone="x"),
            AC("zyy", Permission.READ, zone="x"),
            AC("aaa", Permission.READ),
            AC("aaa", Permission.WRITE),
            AC("zyy", Permission.OWN),
        ]


@m.describe("AVU")
class TestAVU:
    @m.describe("Namespaces")
    def test_create_with_namespaced_attribute(self):
        assert AVU("a", 1).namespace == ""
        assert AVU("a", 1).attribute == "a"
        assert AVU("a", 1).without_namespace == "a"

        assert AVU("a", 1, namespace="x").namespace == "x"
        assert AVU("a", 1, namespace="x").attribute == "x:a"
        assert AVU("a", 1, namespace="x").without_namespace == "a"

        assert AVU("x:a", 1).namespace == "x"
        assert AVU("x:a", 1).attribute == "x:a"
        assert AVU("x:a", 1).without_namespace == "a"

        with pytest.raises(ValueError, match="may not be entirely whitespace"):
            AVU(" ", 1)

        with pytest.raises(ValueError, match="may not be entirely whitespace"):
            AVU("a", 1, namespace=" ")

        with pytest.raises(ValueError, match="namespace contained a separator"):
            AVU("a", 1, namespace="x:")

        with pytest.raises(ValueError, match="did not match the declared namespace"):
            AVU("y:a", 1, namespace="x")

        # We can handle attributes with colons
        assert AVU("x::a", 1).namespace == "x"
        assert AVU("x::a", 1).attribute == "x::a"
        assert AVU("x::a", 1).without_namespace == ":a"

        assert AVU("x:a:", 1).namespace == "x"
        assert AVU("x:a:", 1).attribute == "x:a:"
        assert AVU("x:a:", 1).without_namespace == "a:"

        assert AVU("x:a:b", 1).namespace == "x"
        assert AVU("x:a:b", 1).attribute == "x:a:b"
        assert AVU("x:a:b", 1).without_namespace == "a:b"

        # We can handle attributes with multiple colons
        assert AVU("x:a::", 1).namespace == "x"
        assert AVU("x:a::", 1).attribute == "x:a::"
        assert AVU("x:a::", 1).without_namespace == "a::"

        assert AVU("x::a:", 1).namespace == "x"
        assert AVU("x::a:", 1).attribute == "x::a:"
        assert AVU("x::a:", 1).without_namespace == ":a:"

        assert AVU("x:::a", 1).namespace == "x"
        assert AVU("x:::a", 1).attribute == "x:::a"
        assert AVU("x:::a", 1).without_namespace == "::a"

        # We can handle iRODS' own AVUs that it adds sometimes
        assert AVU("irods::a", 1).namespace == AVU.IRODS_NAMESPACE
        assert AVU("irods::a", 1).attribute == "irods::a"
        assert AVU("irods::a", 1).without_namespace == "a"

        # Even if they should have extra colons
        assert AVU("irods:::a", 1).namespace == AVU.IRODS_NAMESPACE
        assert AVU("irods:::a", 1).attribute == "irods:::a"
        assert AVU("irods:::a", 1).without_namespace == ":a"

        with pytest.raises(ValueError, match="did not match the declared namespace"):
            AVU("irods::a", 1, namespace="x")

    @m.describe("Comparison")
    def test_compare_avus_equal(self):
        assert AVU("a", 1) == AVU("a", 1)
        assert AVU("a", 1, "mm") == AVU("a", 1, "mm")

        assert AVU("a", 1) != AVU("a", 1, "mm")

        assert AVU("a", 1).with_namespace("x") == AVU("a", 1).with_namespace("x")

        assert AVU("a", 1).with_namespace("x") != AVU("a", 1).with_namespace("y")

    def test_compare_avus_lt(self):
        assert AVU("a", 1) < AVU("b", 1)
        assert AVU("a", 1) < AVU("a", 2)

        assert AVU("a", 1, "mm") < AVU("a", 1)
        assert AVU("a", 1, "mm") < AVU("a", 2, "mm")
        assert AVU("a", 1, "cm") < AVU("a", 1, "mm")

        assert AVU("a", 1).with_namespace("x") < AVU("a", 1)
        assert AVU("z", 99).with_namespace("x") < AVU("a", 1)

        assert AVU("a", 1).with_namespace("x") < AVU("a", 1).with_namespace("y")

    def test_compare_avus_sort(self):
        x = [AVU("z", 1), AVU("y", 1), AVU("x", 1)]
        x.sort()
        assert x == [AVU("x", 1), AVU("y", 1), AVU("z", 1)]

        y = [AVU("x", 2), AVU("x", 3), AVU("x", 1)]
        y.sort()
        assert y == [AVU("x", 1), AVU("x", 2), AVU("x", 3)]

    def test_compare_avus_sort_ns(self):
        x = [AVU("z", 1).with_namespace("a"), AVU("y", 1), AVU("x", 1)]
        x.sort()

        assert x == [AVU("z", 1).with_namespace("a"), AVU("x", 1), AVU("y", 1)]

    def test_compare_avus_sort_units(self):
        x = [AVU("x", 1, "mm"), AVU("x", 1, "cm"), AVU("x", 1, "km")]
        x.sort()

        assert x == [AVU("x", 1, "cm"), AVU("x", 1, "km"), AVU("x", 1, "mm")]


@m.describe("RodsPath")
class TestRodsPath:
    @m.describe("Support for iRODS path inspection")
    @m.context("When a collection path exists")
    @m.it("Is identified as a collection")
    def test_collection_path_type(self, simple_collection):
        assert rods_path_type(simple_collection) == Collection
        assert make_rods_item(simple_collection) == Collection(simple_collection)
        assert Collection(simple_collection).rods_type == Collection

    @m.context("When a data object path exists")
    @m.it("Is identified as a data object")
    def test_data_object_path_type(self, simple_data_object):
        assert rods_path_type(simple_data_object) == DataObject
        assert make_rods_item(simple_data_object) == DataObject(simple_data_object)
        assert DataObject(simple_data_object).rods_type == DataObject

    @m.it("Can report its ancestors")
    def test_collection_ancestors(self, full_collection):
        assert Collection("/testZone").ancestors() == [Collection("/")]
        assert Collection("/testZone/level0").ancestors() == [
            Collection("/testZone"),
            Collection("/"),
        ]
        assert Collection("/testZone/level0/level1").ancestors() == [
            Collection("/testZone/level0"),
            Collection("/testZone"),
            Collection("/"),
        ]
        assert DataObject("/testZone/level0/level1/leaf.txt").ancestors() == [
            Collection("/testZone/level0/level1"),
            Collection("/testZone/level0"),
            Collection("/testZone"),
            Collection("/"),
        ]

    @m.it("Returns an empty list for root collection ancestors")
    def test_ancestor_metadata_root(self):
        root = Collection("/testZone")
        assert root.ancestor_metadata() == []

    @m.it("Returns combined metadata from all ancestors")
    def test_ancestor_metadata_nested(self, full_collection):
        root_avu = AVU("level", "root")
        desc_avu = AVU("level", "descendant")

        root = Collection(full_collection)
        root.add_metadata(root_avu)

        for item in root.contents(recurse=True):
            assert item.ancestor_metadata() == root.metadata(), "Shared root metadata"

        for item in root.contents(recurse=True):
            item.add_metadata(desc_avu)

        for child in root.contents(recurse=False):
            assert child.ancestor_metadata() == root.metadata(), "Shared root metadata"

            for desc in child.contents(recurse=True):
                assert desc.ancestor_metadata() == sorted(
                    [
                        *root.metadata(),
                        desc_avu,
                    ]
                ), "Shared descendant metadata"

        for item in root.contents(recurse=True):
            if item.rods_type == Collection:
                item.add_metadata(AVU("name", item.path.name))

        for item in root.contents(recurse=True):
            if item.rods_type == DataObject and item.name in [
                "leaf1.txt",
                "leaf2.txt",
            ]:
                assert item.ancestor_metadata() == sorted(
                    [
                        root_avu,
                        desc_avu,
                        AVU("name", "level1"),
                        AVU("name", "level2"),
                    ]
                )


@m.describe("Collection")
class TestCollection:
    @m.describe("Support for str path")
    @m.context("When a Collection is made from a str path")
    @m.it("Can be created")
    def test_make_collection_str(self, simple_collection):
        p = PurePath(simple_collection)
        coll = Collection(p.as_posix())

        assert coll.exists()
        assert coll.path == p

    @m.describe("Support for pathlib.Path")
    @m.context("When a Collection is made from a pathlib.Path")
    @m.it("Can be created")
    def test_make_collection_pathlib(self, simple_collection):
        p = PurePath(simple_collection)
        coll = Collection(p)

        assert coll.exists()
        assert coll.path == p

    @m.describe("Disallow data object paths")
    @m.context("When a Collection is made from a data object path")
    @m.it("Raises an error if checking is enabled")
    def test_make_collection_data_object_path(self, simple_data_object):
        p = PurePath(simple_data_object)
        Collection(p, check_type=False).exists()

        with pytest.raises(BatonError, match="Invalid iRODS path"):
            Collection(p, check_type=True).exists()

    @m.describe("Testing existence")
    @m.context("When a Collection exists")
    @m.it("Can be detected")
    def test_collection_exists(self, simple_collection):
        assert Collection(PurePath(simple_collection)).exists()
        assert not Collection("/no/such/collection").exists()

    @m.it("Can be listed (non-recursively)")
    def test_list_collection(self, simple_collection):
        coll = Collection(simple_collection)
        assert coll.list() == Collection(simple_collection)

        coll = Collection("/no/such/collection")
        with pytest.raises(RodsError, match="does not exist"):
            coll.list()

    @m.it("Can have its contents listed")
    def test_list_collection_contents(self, ont_gridion):
        p = PurePath(
            ont_gridion,
            "66",
            "DN585561I_A1",
            "20190904_1514_GA20000_FAL01979_43578c8f",
        )

        contents = Collection(p).contents()
        all_paths = [c.path.as_posix() for c in contents]
        common_root = PurePath(os.path.commonpath(all_paths))

        def relative(path):
            return PurePath(path).relative_to(common_root).as_posix()

        colls = filter(lambda x: isinstance(x, Collection), contents)
        assert [relative(c) for c in colls] == [
            "fast5_fail",
            "fast5_pass",
            "fastq_fail",
            "fastq_pass",
        ]

        objs = filter(lambda x: isinstance(x, DataObject), contents)
        assert [relative(o) for o in objs] == [
            "GXB02004_20190904_151413_FAL01979_gridion_sequencing_run_DN585561I_A1_sequencing_summary.txt",
            "duty_time.csv",
            "final_report.txt.gz",
            "final_summary.txt",
            "report.md",
            "report.pdf",
            "throughput.csv",
        ]

    @m.it("Can have its contents listed recursively")
    def test_list_collection_contents_recurse(self, ont_gridion):
        p = PurePath(
            ont_gridion,
            "66",
            "DN585561I_A1",
            "20190904_1514_GA20000_FAL01979_43578c8f",
        )
        contents = Collection(p).contents(recurse=True)

        all_paths = [c.path.as_posix() for c in contents]
        common_root = PurePath(os.path.commonpath(all_paths))

        def relative(path):
            return PurePath(path).relative_to(common_root).as_posix()

        colls = filter(lambda x: isinstance(x, Collection), contents)
        assert [relative(c) for c in colls] == [
            "fast5_fail",
            "fast5_pass",
            "fastq_fail",
            "fastq_pass",
        ]

        objs = filter(lambda x: isinstance(x, DataObject), contents)
        assert [relative(o) for o in objs] == [
            "GXB02004_20190904_151413_FAL01979_gridion_sequencing_run_DN585561I_A1_sequencing_summary.txt",
            "duty_time.csv",
            "final_report.txt.gz",
            "final_summary.txt",
            "report.md",
            "report.pdf",
            "throughput.csv",
            "fast5_fail/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_1.fast5",
            "fast5_fail/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_2.fast5",
            "fast5_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_0.fast5",
            "fast5_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_1.fast5",
            "fast5_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_2.fast5",
            "fast5_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_3.fast5",
            "fastq_fail/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_1.fastq",
            "fastq_fail/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_2.fastq",
            "fastq_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_1.fastq",
            "fastq_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_2.fastq",
            "fastq_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_3.fastq",
            "fastq_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_4.fastq",
        ]

    @m.it("Can have its contents listed using a generator")
    def test_iter_contents(self, ont_gridion):
        p = PurePath(
            ont_gridion,
            "66",
            "DN585561I_A1",
            "20190904_1514_GA20000_FAL01979_43578c8f",
        )

        iter_contents = Collection(p).iter_contents()

        expected_list = [
            "fast5_fail",
            "fast5_pass",
            "fastq_fail",
            "fastq_pass",
            "GXB02004_20190904_151413_FAL01979_gridion_sequencing_run_DN585561I_A1_sequencing_summary.txt",
            "duty_time.csv",
            "final_report.txt.gz",
            "final_summary.txt",
            "report.md",
            "report.pdf",
            "throughput.csv",
            "fast5_fail/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_1.fast5",
            "fast5_fail/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_2.fast5",
            "fast5_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_0.fast5",
            "fast5_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_1.fast5",
            "fast5_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_2.fast5",
            "fast5_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_3.fast5",
            "fastq_fail/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_1.fastq",
            "fastq_fail/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_2.fastq",
            "fastq_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_1.fastq",
            "fastq_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_2.fastq",
            "fastq_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_3.fastq",
            "fastq_pass/FAL01979_9cd2a77baacfe99d6b16f3dad2c36ecf5a6283c3_4.fastq",
        ]
        for actual, expected in zip(iter_contents, expected_list):
            assert str(actual).endswith(expected)

    @m.it("Has a creation timestamp")
    def test_creation_timestamp(self, simple_collection):
        coll = Collection(simple_collection)
        assert isinstance(coll.created(), datetime)

    @m.it("Has a modification timestamp")
    def test_modification_timestamp(self, simple_collection):
        coll = Collection(simple_collection)
        assert isinstance(coll.modified(), datetime)

    @m.it("Has a timestamp equal the modification timestamp")
    def test_timestamp(self, simple_collection):
        coll = Collection(simple_collection)
        assert coll.timestamp() == coll.modified()

    @m.it("Can have metadata added")
    def test_meta_add_collection(self, simple_collection):
        coll = Collection(simple_collection)
        assert coll.metadata() == []

        avu1 = AVU("abcde", "12345")
        avu2 = AVU("vwxyz", "567890")

        assert coll.add_metadata(avu1, avu2) == 2
        assert avu1 in coll.metadata()
        assert avu2 in coll.metadata()

        assert (
            coll.add_metadata(avu1, avu2) == 0
        ), "adding collection metadata is idempotent"

    @m.it("Can have metadata removed")
    def test_meta_rem_collection(self, simple_collection):
        coll = Collection(simple_collection)
        assert coll.metadata() == []

        avu1 = AVU("abcde", "12345")
        avu2 = AVU("vwxyz", "567890")
        coll.add_metadata(avu1, avu2)

        assert coll.remove_metadata(avu1, avu2) == 2
        assert avu1 not in coll.metadata()
        assert avu2 not in coll.metadata()
        assert (
            coll.remove_metadata(avu1, avu2) == 0
        ), "removing collection metadata is idempotent"

    @m.it("Can be searched for an AVU with an unique attribute")
    def test_avu_collection(self, simple_collection):
        coll = Collection(simple_collection)
        avu = AVU("abcde", "12345")

        with pytest.raises(ValueError, match="did not contain any AVU with attribute"):
            coll.avu("abcde")
        coll.add_metadata(avu)

        assert coll.avu("abcde") == avu

        coll.add_metadata(AVU("abcde", "67890"))
        with pytest.raises(
            ValueError, match="contained more than one AVU with attribute"
        ):
            coll.avu("abcde")

    @m.it("Can be found by its metadata")
    def test_meta_query_collection(self, simple_collection):
        coll = Collection(simple_collection)

        avu = AVU("abcde", "12345")
        coll.add_metadata(avu)
        assert coll.metadata() == [avu]

        found = query_metadata(avu, collection=True, zone=coll)
        assert found == [Collection(simple_collection)]

    @m.it("Can be found by timestamp")
    def test_timestamp_query_collection(self, simple_collection):
        coll = Collection(simple_collection)

        avu = AVU("abcde", "12345")
        coll.add_metadata(avu)
        assert coll.metadata() == [avu]

        le_created = Timestamp(coll.created(), Timestamp.Event.CREATED, operator="n<=")

        found = query_metadata(avu, timestamps=[le_created], collection=True, zone=coll)
        assert found == [Collection(simple_collection)]

        gt_created = Timestamp(coll.created(), Timestamp.Event.CREATED, operator="n>")
        found = query_metadata(avu, timestamps=[gt_created], collection=True, zone=coll)
        assert found == []

    @m.context("When a Collection does not exist")
    @m.it("Can be created de novo")
    def test_create_collection(self, simple_collection):
        coll = Collection(simple_collection / "new-sub")
        assert not coll.exists()
        coll.create(parents=False, exist_ok=False)
        assert coll.exists()

    @m.it("Can be created with parents on demand")
    def test_create_collection_parents(self, simple_collection):
        coll = Collection(simple_collection / "new-sub" / "new-sub-sub")
        assert not coll.exists()
        with pytest.raises(RodsError, match="create collection"):
            coll.create(parents=False, exist_ok=False)

        coll.create(parents=True, exist_ok=False)
        assert coll.exists()

    @m.it("Will ignore existing collections on demand")
    def test_create_collection_existing(self, simple_collection):
        coll = Collection(simple_collection)
        assert coll.exists()
        coll.create(parents=False, exist_ok=True)

        with pytest.raises(RodsError, match="create collection"):
            coll.create(parents=False, exist_ok=False)

    @m.it("Can have access controls added, non-recursively")
    def test_add_ac_collection(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)

        assert coll.acl() == [irods_own]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own]

        assert (
            coll.add_permissions(irods_own) == 0
        ), "Nothing is added when new ACL == all old ACL"
        assert coll.acl() == [irods_own]

        assert coll.add_permissions(public_read) == 1
        assert coll.acl() == [
            irods_own,
            public_read,
        ], "Access control added non-recursively"

        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own], "Collection content ACL unchanged"

    @m.it("Can have its permissions listed")
    def test_list_ac_collection(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)
        assert coll.add_permissions(public_read) == 1

        assert coll.acl() == [irods_own, public_read]
        assert coll.acl(user_type="rodsadmin") == [irods_own]
        assert coll.acl(user_type="rodsgroup") == [public_read]

    @m.it("Can have access controls added, recursively")
    def test_add_ac_collection_recurse(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)

        assert coll.acl() == [irods_own]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own]

        assert (
            coll.add_permissions(irods_own, recurse=True) == 0
        ), "Nothing is added when new ACL == all old ACL, recursively"
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own]

        tree = [
            "recurse",
            "level1/",
            "level1/level2/",
            "level1/level2/leaf1.txt",
            "level1/level2/leaf2.txt",
        ]
        assert coll.add_permissions(public_read, recurse=True) == len(
            tree
        ), "Access control added recursively"
        for item in coll.contents(recurse=True):
            assert item.acl() == [
                irods_own,
                public_read,
            ], "Collection content ACL updated"

    @m.it("Can have access controls added, recursively, with a filter")
    def test_add_ac_collection_recurse_filter(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)

        assert coll.acl() == [irods_own]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own]

        assert (
            coll.add_permissions(
                irods_own,
                recurse=True,
                filter_fn=lambda x: x.rods_type == DataObject and x.name == "leaf1.txt",
            )
            == 0
        ), "Nothing is added when new ACL == all old ACL, recursively"
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own]

        tree = [
            "recurse",
            "level1/",
            "level1/level2/",
            "level1/level2/leaf2.txt",
        ]
        assert coll.add_permissions(
            public_read,
            recurse=True,
            filter_fn=lambda x: x.rods_type == DataObject and x.name == "leaf1.txt",
        ) == len(tree), "Access control added recursively"

        for item in coll.contents(recurse=True):
            expected = (
                [irods_own]
                if item.rods_type == DataObject and item.name == "leaf1.txt"
                else [irods_own, public_read]
            )
            assert item.acl() == expected, "Collection content ACL updated"

    @m.it("Can have access controls removed, non-recursively")
    def test_rem_ac_collection(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)

        assert coll.acl() == [irods_own]
        assert (
            coll.remove_permissions(public_read) == 0
        ), "Nothing is removed when the access control does not exist"

        coll.add_permissions(public_read, recurse=True)
        assert coll.acl() == [irods_own, public_read]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own, public_read]

        assert coll.remove_permissions(public_read) == 1
        assert coll.acl() == [irods_own], "Access control removed non-recursively"
        for item in coll.contents(recurse=True):
            assert item.acl() == [
                irods_own,
                public_read,
            ], "Collection content ACL unchanged"

    @m.it("Can have access controls removed, recursively")
    def test_rem_ac_collection_recurse(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)

        coll.add_permissions(public_read, recurse=True)
        assert coll.acl() == [irods_own, public_read]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own, public_read]

        tree = [
            "recurse",
            "level1/",
            "level1/level2/",
            "level1/level2/leaf1.txt",
            "level1/level2/leaf2.txt",
        ]
        assert coll.remove_permissions(public_read, recurse=True) == len(
            tree
        ), "Access control removed recursively"
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own], "Collection content ACL updated"

    @m.it("Can have access controls removed, recursively, with a filter")
    def test_rem_ac_collection_recurse_filter(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)

        coll.add_permissions(public_read, recurse=True)
        assert coll.acl() == [irods_own, public_read]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own, public_read]

        tree = [
            "recurse",
            "level1/",
            "level1/level2/",
            "level1/level2/leaf2.txt",
        ]
        assert coll.remove_permissions(
            public_read,
            recurse=True,
            filter_fn=lambda x: x.rods_type == DataObject and x.name == "leaf1.txt",
        ) == len(tree), "Access control removed recursively"

        for item in coll.contents(recurse=True):
            expected = (
                [irods_own, public_read]
                if item.rods_type == DataObject and item.name == "leaf1.txt"
                else [irods_own]
            )
            assert item.acl() == expected, "Collection content ACL updated"

    @m.it("Can have access controls superseded, non-recursively")
    def test_super_ac_collection(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)
        study_01_read = AC("ss_study_01", Permission.READ, zone=zone)
        study_02_read = AC("ss_study_02", Permission.READ, zone=zone)

        coll.add_permissions(study_01_read, study_02_read, recurse=True)
        assert coll.acl() == [irods_own, study_01_read, study_02_read]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own, study_01_read, study_02_read]

        num_removed, num_added = coll.supersede_permissions(irods_own, public_read)
        assert num_removed == 2  # study access
        assert num_added == 1  # public access
        assert coll.acl() == [
            irods_own,
            public_read,
        ], "Access superseded removed non-recursively"
        for item in coll.contents(recurse=True):
            assert item.acl() == [
                irods_own,
                study_01_read,
                study_02_read,
            ], "Collection content ACL unchanged"

    @m.it("Can have access controls superseded, recursively")
    def test_super_ac_collection_recur(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)
        study_01_read = AC("ss_study_01", Permission.READ, zone=zone)
        study_02_read = AC("ss_study_02", Permission.READ, zone=zone)

        coll.add_permissions(study_01_read, study_02_read, recurse=True)
        assert coll.acl() == [irods_own, study_01_read, study_02_read]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own, study_01_read, study_02_read]

        new_acl = [irods_own, public_read]
        num_removed, num_added = coll.supersede_permissions(*new_acl, recurse=True)
        assert num_removed == 2 * 5  # study access
        assert num_added == 1 * 5  # public access
        assert coll.acl() == new_acl, "Access superseded removed recursively"
        for item in coll.contents(recurse=True):
            assert item.acl() == new_acl, "Collection content ACL updated"

    @m.it("Can have access controls superseded, recursively, with a filter")
    def test_super_ac_collection_recur_filter(self, full_collection):
        zone = "testZone"
        coll = Collection(full_collection)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)
        study_01_read = AC("ss_study_01", Permission.READ, zone=zone)
        study_02_read = AC("ss_study_02", Permission.READ, zone=zone)

        coll.add_permissions(study_01_read, study_02_read, recurse=True)
        assert coll.acl() == [irods_own, study_01_read, study_02_read]
        for item in coll.contents(recurse=True):
            assert item.acl() == [irods_own, study_01_read, study_02_read]

        new_acl = [irods_own, public_read]
        num_removed, num_added = coll.supersede_permissions(
            *new_acl,
            recurse=True,
            filter_fn=lambda x: x.rods_type == DataObject and x.name == "leaf1.txt",
        )
        assert num_removed == 2 * (5 - 1)  # study access
        assert num_added == 1 * (5 - 1)  # public access
        assert coll.acl() == new_acl, "Access superseded removed recursively"

        for item in coll.contents(recurse=True):
            expected = (
                [irods_own, study_01_read, study_02_read]
                if item.rods_type == DataObject and item.name == "leaf1.txt"
                else new_acl
            )
            assert item.acl() == expected, "Collection content ACL updated"

    @m.context("When a Collection is put from a local path that is not a directory")
    @m.it("Raises an error")
    def test_put_collection_no_path(self, tmp_path, simple_collection):
        coll = Collection(simple_collection / "sub")

        with pytest.raises(FileNotFoundError):
            next(coll.put(Path("./tests/data/no/such/path").absolute()))

        tmpfile = tmp_path / "test.txt"
        tmpfile.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            next(coll.put(tmpfile))

    @m.context(
        "When a Collection is put from a local path that is not a directory, with exception yielding"
    )
    @m.it("Raises an error")
    def test_put_collection_no_path_yield_exception(self, tmp_path, simple_collection):
        coll = Collection(simple_collection / "sub")

        items = [
            item
            for item in coll.put(
                Path(
                    "./tests/data/no/such/path",
                ).absolute(),
                yield_exceptions=True,
            )
        ]

        assert len(items) == 1
        assert isinstance(items[0], FileNotFoundError)

    @m.context("When a Collection does not exist and is put non-recursively")
    @m.it("Is created, with its immediate contents")
    def test_put_collection(self, simple_collection):
        coll = Collection(simple_collection / "sub")
        assert not coll.exists()

        local_path = Path("./tests/data/simple/collection").absolute()

        items = [item for item in coll.put(local_path, recurse=False)]
        for item in items:
            assert item.exists()

        assert items[0] == coll
        assert items[0].contents() == [Collection(coll.path / "child")]

        assert items[1] == Collection(coll.path / "child")
        assert len(items) == 2

    @m.context(
        "When a collection has a top level data object and is put non-recursively"
    )
    @m.it("Is created, with its immediate contents")
    def test_put_collection_top_level_data_object_non_recursive(
        self, simple_collection
    ):
        dest = simple_collection / "sub"
        coll = Collection(dest)
        assert not coll.exists()

        local_path = Path("./tests/data/top_level")

        items = [item for item in coll.put(local_path, recurse=False)]
        for item in items:
            assert item.exists()

        assert Collection(dest).contents(recurse=True) == [
            Collection(dest / "sub"),
            DataObject(dest / "top_level.txt"),
        ]

    @m.context("When a Collection does not exist and is put recursively")
    @m.it("Is created, with descendants and their contents")
    def test_put_collection_recur(self, simple_collection):
        coll = Collection(simple_collection / "sub")
        assert not coll.exists()

        local_path = Path("./tests/data/simple").absolute()
        items = {
            item for item in coll.put(local_path, recurse=True, verify_checksum=True)
        }
        for item in items:
            assert item.exists()

        sub1 = Collection(coll.path / "collection")
        sub2 = Collection(coll.path / "data_object")
        child = Collection(sub1.path / "child")
        utf8 = DataObject(sub2.path / "utf-8.txt")
        empty = DataObject(sub2.path / "empty.txt")
        lorem = DataObject(sub2.path / "lorem.txt")
        ignore = DataObject(child.path / ".gitignore")

        assert items == {coll, sub1, sub2, child, ignore, empty, lorem, utf8}
        assert coll.contents() == [sub1, sub2]
        assert sub1.contents() == [child]
        assert child.contents() == [ignore]
        assert sub2.contents() == [empty, lorem, utf8]

        assert empty.size() == 0
        assert empty.checksum() == "d41d8cd98f00b204e9800998ecf8427e"
        assert lorem.size() == 555
        assert lorem.checksum() == "39a4aa291ca849d601e4e5b8ed627a04"
        assert utf8.size() == 2522
        assert utf8.checksum() == "500cec3fbb274064e2a25fa17a69638a"
        assert ignore.size() == 0
        assert ignore.checksum() == "d41d8cd98f00b204e9800998ecf8427e"

    @m.context("When a Collection does not exist and is put recursively")
    @m.it("Collection paths can be pruned by providing a filter predicate")
    def test_put_collection_filter_coll(self, simple_collection):
        coll = Collection(simple_collection / "sub")
        assert not coll.exists()

        local_path = Path("./tests/data/simple").absolute()
        items = {
            item
            for item in coll.put(
                local_path,
                recurse=True,
                verify_checksum=True,
                filter_fn=lambda p: p.name == "data_object",  # Prune this path
            )
        }
        for item in items:
            assert item.exists()

        sub1 = Collection(coll.path / "collection")
        child = Collection(sub1.path / "child")
        ignore = DataObject(child.path / ".gitignore")

        assert items == {coll, sub1, child, ignore}
        assert coll.contents() == [sub1]
        assert sub1.contents() == [child]
        assert child.contents() == [ignore]

    @m.context("When a Collection does not exist and is put recursively")
    @m.it("Data object paths can be skipped by providing a filter predicate")
    def test_put_collection_filter_obj(self, simple_collection):
        coll = Collection(simple_collection / "sub")
        assert not coll.exists()

        local_path = Path("./tests/data/simple").absolute()
        items = {
            item
            for item in coll.put(
                local_path,
                recurse=True,
                verify_checksum=True,
                filter_fn=(
                    lambda p: os.path.isfile(p) and os.path.getsize(p) == 0
                ),  # Skip empty files
            )
        }
        for item in items:
            assert item.exists()

        sub1 = Collection(coll.path / "collection")
        sub2 = Collection(coll.path / "data_object")
        child = Collection(sub1.path / "child")
        lorem = DataObject(sub2.path / "lorem.txt")
        utf8 = DataObject(sub2.path / "utf-8.txt")

        assert items == {coll, sub1, sub2, child, lorem, utf8}

        assert coll.exists()
        assert sub2.contents() == [lorem, utf8]

    @m.context("When a Collection does not exist and is put recursively")
    @m.it("Errors cause an exception to be raised by default, stopping the generator")
    def test_put_collection_raise_except(self, simple_collection):
        coll = Collection(simple_collection / "sub")
        assert not coll.exists()

        def raise_error_filter(path):
            if path.name == "empty.txt":
                raise ValueError("This is an artificial error")
            return False

        local_path = Path("./tests/data/simple").absolute()
        items = set()

        with pytest.raises(ValueError, match="This is an artificial error"):
            gen = coll.put(
                local_path,
                recurse=True,
                verify_checksum=True,
                filter_fn=raise_error_filter,
            )
            try:
                for item in gen:
                    items.add(item)
            finally:
                gen.close()

        # The generator stops at the error on "empty.txt", but will yield earlier items
        sub1 = Collection(coll.path / "collection")
        sub2 = Collection(coll.path / "data_object")
        child = Collection(sub1.path / "child")
        ignore = DataObject(child.path / ".gitignore")

        assert items == {coll, sub1, sub2, child, ignore}
        for item in items:
            assert item.exists()

    @m.context("When a Collection does not exist and is put recursively")
    @m.it("Errors cause an exceptions to be yielded, when requested")
    def test_put_collection_yield_except(self, simple_collection):
        coll = Collection(simple_collection / "sub")
        assert not coll.exists()

        local_path = Path("./tests/data/simple").absolute()

        present = [
            item
            for item in coll.put(local_path, recurse=True, yield_exceptions=False)
            if item.exists()
        ]
        assert len(present) == 8

        items, exceptions = [], []
        gen = coll.put(
            local_path,
            recurse=True,
            force=False,  # Raises errors for data objects because items are already present
            yield_exceptions=True,
        )
        try:
            for item in gen:
                match item:
                    case Exception():
                        exceptions.append(item)
                    case _:
                        items.append(item)
        finally:
            gen.close()

        assert len(items) == 4
        assert len(exceptions) == 4
        for e in exceptions:
            assert isinstance(e, FileExistsError)
            assert re.search("object already exists", str(e), re.IGNORECASE)

    @m.context("When a Collection does not exist and is put recursively")
    @m.it("Can fill in around existing data objects from a previous attempt")
    def test_put_collection_fill(self, simple_collection):
        coll = Collection(simple_collection / "sub")
        assert not coll.exists()

        local_path = Path("./tests/data/simple").absolute()
        items = {
            item
            for item in coll.put(
                local_path,
                recurse=True,
                verify_checksum=True,
                filter_fn=(lambda p: p.name == "lorem.txt"),  # Omit this
            )
        }

        for item in items:
            assert item.exists()
        assert len(items) == 7
        assert "lorem.txt" not in [
            item.name for item in items if isinstance(item, DataObject)
        ]

        items = {
            item
            for item in coll.put(
                local_path, fill=True, force=False, recurse=True, verify_checksum=True
            )
        }

        assert len(items) == 8
        assert "lorem.txt" in [
            item.name for item in items if isinstance(item, DataObject)
        ]

    @m.context("When a Collection exists and is got non-recursively")
    @m.it("Is downloaded, with its immediate contents")
    def test_get_collection(self, tmp_path, full_collection):
        coll = Collection(full_collection)
        assert coll.exists()

        items = list(coll.get(tmp_path, recurse=False, verify_checksum=True))

        local_paths = _check_local_paths(tmp_path, coll, items)
        assert local_paths == [tmp_path]

    @m.context("When a Collection exists and is got recursively")
    @m.it("Is downloaded, with its recursive contents")
    def test_get_collection_recur(self, tmp_path, full_collection):
        coll = Collection(full_collection)
        assert coll.exists()

        items = list(coll.get(tmp_path, recurse=True, verify_checksum=True))

        local_paths = _check_local_paths(tmp_path, coll, items)
        assert sorted(local_paths) == [
            tmp_path / p
            for p in [
                ".",
                "level1",
                "level1/level2",
                "level1/level2/leaf1.txt",
                "level1/level2/leaf2.txt",
            ]
        ]

    @m.context("When a Collection exists and is got recursively")
    @m.it("Collection paths can be pruned by providing a filter predicate")
    def test_get_collection_filter_coll(self, tmp_path, full_collection):
        coll = Collection(full_collection)
        assert coll.exists()

        items = list(
            coll.get(
                tmp_path,
                recurse=True,
                verify_checksum=True,
                filter_fn=lambda c: c.path.name == "level2",
            )
        )

        local_paths = _check_local_paths(tmp_path, coll, items)
        assert sorted(local_paths) == [tmp_path / p for p in [".", "level1"]]

    @m.context("When a Collection exists and is got recursively")
    @m.it("Data object paths can be skipped by providing a filter predicate")
    def test_get_collection_filter_obj(self, tmp_path, full_collection):
        coll = Collection(full_collection)
        assert coll.exists()

        items = list(
            coll.get(
                tmp_path,
                recurse=True,
                verify_checksum=True,
                filter_fn=lambda x: isinstance(x, DataObject) and x.name == "leaf1.txt",
            )
        )

        local_paths = _check_local_paths(tmp_path, coll, items)
        assert sorted(local_paths) == [
            tmp_path / p
            for p in [
                ".",
                "level1",
                "level1/level2",
                "level1/level2/leaf2.txt",
            ]
        ]

    @m.context("When a Collection exists and is got recursively")
    @m.it("Errors cause an exception to be raised by default, stopping the generator")
    def test_get_collection_raise_except(self, tmp_path, full_collection):
        coll = Collection(full_collection)
        assert coll.exists()

        def raise_error_filter(x):
            if isinstance(x, DataObject) and x.name == "leaf1.txt":
                raise ValueError("This is an artificial error")
            return False

        items = []
        with pytest.raises(ValueError, match="This is an artificial error"):
            gen = coll.get(
                tmp_path,
                recurse=True,
                verify_checksum=True,
                filter_fn=raise_error_filter,
            )
            try:
                for item in gen:
                    items.append(item)
            finally:
                gen.close()

        local_paths = _check_local_paths(tmp_path, coll, items)

        assert sorted(local_paths) == [
            tmp_path / p for p in [".", "level1", "level1/level2"]
        ]

    @m.context("When a Collection exists and is got recursively")
    @m.it("Errors cause an exceptions to be yielded, when requested")
    def test_get_collection_yield_except(self, tmp_path, full_collection):
        coll = Collection(full_collection)
        assert coll.exists()

        present = list(coll.get(tmp_path, recurse=True, verify_checksum=True))
        assert len(present) == 5

        items, exceptions = [], []
        gen = coll.get(
            tmp_path,
            recurse=True,
            force=False,  # Raises errors because local files are already present
            yield_exceptions=True,
        )
        try:
            for item in gen:
                match item:
                    case Exception():
                        exceptions.append(item)
                    case _:
                        items.append(item)
        finally:
            gen.close()

        assert len(items) == 3
        assert len(exceptions) == 2
        for e in exceptions:
            assert isinstance(e, FileExistsError)
            assert re.search("file already exists", str(e), re.IGNORECASE)

    @m.context("When a Collection exists and is got recursively")
    @m.it("Can fill in around existing data content from a previous attempt")
    def test_get_collection_fill(self, tmp_path, full_collection):
        coll = Collection(full_collection)
        assert coll.exists()

        items = list(
            coll.get(
                tmp_path,
                recurse=True,
                verify_checksum=True,
                filter_fn=lambda x: isinstance(x, DataObject) and x.name == "leaf1.txt",
            )
        )

        local_paths = _check_local_paths(tmp_path, coll, items)
        assert sorted(local_paths) == [
            tmp_path / p
            for p in [
                ".",
                "level1",
                "level1/level2",
                "level1/level2/leaf2.txt",
            ]
        ]

        items = list(
            coll.get(
                tmp_path, recurse=True, verify_checksum=True, force=False, fill=True
            )
        )

        local_paths = _check_local_paths(tmp_path, coll, items)
        assert sorted(local_paths) == [
            tmp_path / p
            for p in [
                ".",
                "level1",
                "level1/level2",
                "level1/level2/leaf1.txt",
                "level1/level2/leaf2.txt",
            ]
        ]


@m.describe("DataObject")
class TestDataObject:
    @m.context("When a DataObject is made from a str path")
    @m.it("Can be created")
    def test_make_data_object_str(self, simple_data_object):
        obj = DataObject(simple_data_object.as_posix())

        assert obj.exists()
        assert obj.path == simple_data_object.parent
        assert obj.name == simple_data_object.name

    @m.context("When a DataObject is made from a pathlib.Path")
    @m.it("Can be created")
    def test_make_data_object_pathlib(self, simple_data_object):
        obj = DataObject(simple_data_object)

        assert obj.exists()
        assert obj.path == simple_data_object.parent
        assert obj.name == simple_data_object.name

    @m.describe("Disallow collection paths")
    @m.context("When a DataObject is made from a collection path")
    @m.it("Raises an error if checking is enabled and a checked method called")
    def test_make_data_object_collection_path(self, simple_collection):
        p = PurePath(simple_collection)
        DataObject(p, check_type=False).exists()

        with pytest.raises(BatonError, match="Invalid iRODS path"):
            DataObject(p).exists()

    @m.describe("Putting new data objects")
    @m.context("When a DataObject does not exist")
    @m.it("Can be put from a local file without checksum creation")
    def test_data_object_put_no_checksum(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        assert not obj.exists()

        local_path = Path("./tests/data/simple/data_object/lorem.txt").absolute()
        obj.put(local_path, calculate_checksum=False, verify_checksum=False)
        assert obj.exists()
        assert obj.size() == 555
        assert obj.checksum() is None

    @m.it("Can be put from a local file with checksum creation")
    def test_data_object_put_checksum_no_verify(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        assert not obj.exists()

        local_path = Path("./tests/data/simple/data_object/lorem.txt").absolute()
        obj.put(local_path, calculate_checksum=True, verify_checksum=False)
        assert obj.exists()
        assert obj.size() == 555
        assert obj.checksum() == "39a4aa291ca849d601e4e5b8ed627a04"

    @m.it("Can be put from a local file with checksum calculated on the fly")
    def test_data_object_put_checksum_supplied(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        assert not obj.exists()

        local_path = Path("./tests/data/simple/data_object/lorem.txt").absolute()
        checksum = "39a4aa291ca849d601e4e5b8ed627a04"

        obj.put(local_path, calculate_checksum=True, compare_checksums=True)
        assert obj.exists()
        assert obj.checksum() == checksum

    @m.it("Can be put from a local file with a supplied local checksum string")
    def test_data_object_put_checksum_supplied(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        assert not obj.exists()

        local_path = Path("./tests/data/simple/data_object/lorem.txt").absolute()
        checksum = "39a4aa291ca849d601e4e5b8ed627a04"
        obj.put(
            local_path,
            calculate_checksum=True,
            compare_checksums=True,
            local_checksum=checksum,
        )
        assert obj.exists()
        assert obj.checksum() == checksum

    @m.it("Can be put from a local file with a supplied local checksum callable")
    def test_data_object_put_callable_supplied(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        assert not obj.exists()

        local_path = Path("./tests/data/simple/data_object/lorem.txt").absolute()
        checksum = "39a4aa291ca849d601e4e5b8ed627a04"
        obj.put(
            local_path,
            calculate_checksum=True,
            compare_checksums=True,
            local_checksum=lambda _: checksum,
        )
        assert obj.exists()
        assert obj.checksum() == checksum

    @m.it("Raises an error if a supplied local checksum callable does not match")
    def test_data_object_put_callable_supplied(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        assert not obj.exists()

        local_path = Path("./tests/data/simple/data_object/lorem.txt").absolute()
        with pytest.raises(ValueError, match="mismatch"):
            obj.put(
                local_path,
                calculate_checksum=True,
                compare_checksums=True,
                local_checksum=lambda _: "a bad checksum",
            )

    @m.it("Raises an error if a supplied local checksum string does not match")
    def test_data_object_put_checksum_supplied_mismatch(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        assert not obj.exists()

        local_path = Path("./tests/data/simple/data_object/lorem.txt").absolute()
        with pytest.raises(ValueError, match="mismatch"):
            obj.put(local_path, compare_checksums=True, local_checksum="a bad checksum")

    @m.describe("Operations on an existing DataObject")
    @m.context("When a DataObject exists")
    @m.it("Can be detected")
    def test_detect_data_object(self, simple_data_object):
        assert DataObject(simple_data_object).exists()
        assert not DataObject("/no/such/object.txt").exists()

    @m.it("Can be listed")
    def test_list_data_object(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert obj.list() == DataObject(simple_data_object)

        obj = DataObject("/no/such/data_object.txt")
        with pytest.raises(RodsError, match="does not exist"):
            obj.list()

    @m.it("Can be got from iRODS to a file")
    def test_get_data_object(self, tmp_path, simple_data_object):
        obj = DataObject(simple_data_object)

        local_path = tmp_path / simple_data_object.name
        size = obj.get(local_path, verify_checksum=True)
        assert size == 555

        md5 = hashlib.md5(open(local_path, "rb").read()).hexdigest()
        assert md5 == "39a4aa291ca849d601e4e5b8ed627a04"

    @m.it("Can be read from iRODS")
    def test_read_data_object(self, tmp_path, simple_data_object):
        obj = DataObject(simple_data_object)
        contents = obj.read()
        assert (
            hashlib.md5(contents.encode()).hexdigest()
            == "39a4aa291ca849d601e4e5b8ed627a04"
        )

    @m.it("Has a size")
    def test_data_object_size(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert obj.size() == 555
        assert len(obj.read()) == 555

    @m.it("Can have its checksum and size consistency verified")
    def test_verify_data_object_consistency(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        obj.put(
            Path("./tests/data/simple/data_object/lorem.txt"),
            calculate_checksum=False,
            verify_checksum=False,
        )
        assert obj.size() == 555
        assert obj.checksum() is None
        assert obj.is_consistent_size()
        chk = obj.checksum(calculate_checksum=True)
        assert obj.checksum() == chk
        assert chk == "39a4aa291ca849d601e4e5b8ed627a04"
        assert obj.is_consistent_size()

        empty = DataObject(simple_collection / "empty.txt")
        empty.put(
            Path("./tests/data/simple/data_object/empty.txt"),
            calculate_checksum=False,
            verify_checksum=False,
        )
        assert empty.size() == 0
        assert empty.checksum() is None
        assert empty.is_consistent_size()
        chk = empty.checksum(calculate_checksum=True)
        assert empty.checksum() == chk
        assert chk == "d41d8cd98f00b204e9800998ecf8427e"  # Checksum of an empty file
        assert empty.is_consistent_size()

    @m.it("Has a checksum")
    def test_get_checksum(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert obj.checksum() == "39a4aa291ca849d601e4e5b8ed627a04"

    @m.it("Can have its checksum verified as good")
    @pytest.mark.skipif(
        server_version() <= (4, 2, 10),
        reason=f"requires iRODS server >4.2.10; version is {server_version()}",
    )
    def test_verify_checksum_good(self, simple_data_object):
        obj = DataObject(simple_data_object)

        # Note that in iRODS >= 4.2.10, this always passes, even if the remote file
        # is the wrong size or has a mismatching checksum, because of this iRODS bug:
        # https://github.com/irods/irods/issues/5843
        assert obj.checksum(verify_checksum=True)

    @m.it("Can have its checksum verified as bad")
    @pytest.mark.skipif(
        server_version() <= (4, 2, 10),
        reason=f"requires iRODS server >4.2.10; version is {server_version()}",
    )
    def test_verify_checksum_bad(self, invalid_checksum_data_object):
        obj = DataObject(invalid_checksum_data_object)

        with pytest.raises(RodsError, match="checksum") as e:
            obj.checksum(verify_checksum=True)
        assert e.value.code == -407000  # CHECK_VERIFICATION_RESULTS

    @m.it("Fails checksum verification if it has no checksum")
    @pytest.mark.skipif(
        server_version() <= (4, 2, 10),
        reason=f"requires iRODS server >4.2.10; version is {server_version()}",
    )
    def test_verify_checksum_missing(self, simple_collection):
        obj = DataObject(simple_collection / "new.txt")
        obj.put(
            Path("./tests/data/simple/data_object/lorem.txt"),
            calculate_checksum=False,
            verify_checksum=False,
        )

        assert obj.size() == 555
        assert obj.checksum() is None
        assert obj.is_consistent_size()
        with pytest.raises(RodsError, match="checksum") as e:
            obj.checksum(verify_checksum=True)
        assert e.value.code == -407000  # CHECK_VERIFICATION_RESULTS

    @m.it("Has replicas")
    def test_replicas(self, simple_data_object):
        obj = DataObject(simple_data_object)

        assert len(obj.replicas()) == 2
        for r in obj.replicas():
            assert r.checksum == "39a4aa291ca849d601e4e5b8ed627a04"
            assert r.valid

    @m.it("Can have its invalid replicas detected")
    def test_invalid_replica(self, invalid_replica_data_object):
        obj = DataObject(invalid_replica_data_object)
        r0, r1 = obj.replicas()
        assert r0.valid
        assert not r1.valid

    @m.it("Has a creation timestamp equal to the earliest replica creation time")
    def test_creation_timestamp(self, simple_data_object):
        obj = DataObject(simple_data_object)

        assert obj.created() == min([o.created for o in obj.replicas()])

    @m.it(
        "Has a modification timestamp equal to the earliest replica modification time"
    )
    def test_modification_timestamp(self, simple_data_object):
        obj = DataObject(simple_data_object)

        assert obj.modified() == min([o.modified for o in obj.replicas()])

    @m.it("Has a timestamp equal to the earliest replica modification time")
    def test_timestamp(self, simple_data_object):
        obj = DataObject(simple_data_object)

        assert obj.timestamp() == min([o.modified for o in obj.replicas()])

    @m.it("Can be overwritten")
    def test_overwrite_data_object(self, tmp_path, simple_data_object):
        obj = DataObject(simple_data_object)

        local_path = tmp_path / simple_data_object.name
        with open(local_path, "w") as f:
            f.write("test\n")

        obj.put(local_path, calculate_checksum=False, verify_checksum=True)
        assert obj.exists()
        assert obj.checksum() == "d8e8fca2dc0f896fd7cb4cb0031ba249"

    @m.it("Can report its metadata")
    def test_has_meta(self, simple_data_object):
        obj = DataObject(simple_data_object)
        avus = [AVU("a", 1), AVU("b", 2), AVU("c", 3)]

        assert not obj.has_metadata(*avus)
        obj.add_metadata(*avus)
        assert obj.has_metadata(*avus)

        for avu in avus:
            assert obj.has_metadata(avu)
        assert obj.has_metadata(*avus[:2])
        assert obj.has_metadata(*avus[1:])
        assert not obj.has_metadata(*avus, AVU("z", 9))

    @m.it("Can report its metadata attributes")
    def test_has_meta_attrs(self, simple_data_object):
        obj = DataObject(simple_data_object)
        avus = [AVU("a", 1), AVU("a", 2), AVU("b", 2), AVU("b", 3), AVU("c", 3)]

        assert not obj.has_metadata_attrs("a", "b", "c")
        obj.add_metadata(*avus)
        assert obj.has_metadata_attrs("a", "b", "c")

        attrs = ["a", "b", "c"]
        for attr in attrs:
            assert obj.has_metadata_attrs(attr)
        assert obj.has_metadata_attrs(*attrs)
        assert obj.has_metadata_attrs(*attrs[:2])
        assert obj.has_metadata_attrs(*attrs[1:])
        assert not obj.has_metadata_attrs(*attrs, "z")

    @m.it("Can add have metadata added")
    def test_add_meta_data_object(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert obj.metadata() == []

        avu1 = AVU("abcde", "12345")
        avu2 = AVU("vwxyz", "567890")

        obj.add_metadata(avu1, avu2)
        assert avu1 in obj.metadata()
        assert avu2 in obj.metadata()

        assert (
            obj.add_metadata(avu1, avu2) == 0
        ), "adding data object metadata is idempotent"

    @m.it("Can have metadata removed")
    def test_rem_meta_data_object(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert obj.metadata() == []

        avu1 = AVU("abcde", "12345")
        avu2 = AVU("vwxyz", "567890")
        obj.add_metadata(avu1, avu2)

        assert obj.remove_metadata(avu1, avu2) == 2
        assert avu1 not in obj.metadata()
        assert avu2 not in obj.metadata()
        assert (
            obj.remove_metadata(avu1, avu2) == 0
        ), "removing data object metadata is idempotent"

    @m.it("Can be searched for an AVU with an unique attribute")
    def test_avu_collection(self, simple_data_object):
        obj = DataObject(simple_data_object)
        avu = AVU("abcde", "12345")

        with pytest.raises(ValueError, match="did not contain any AVU with attribute"):
            obj.avu("abcde")
        obj.add_metadata(avu)

        assert obj.avu("abcde") == avu

        obj.add_metadata(AVU("abcde", "67890"))
        with pytest.raises(
            ValueError, match="contained more than one AVU with attribute"
        ):
            obj.avu("abcde")

    @m.it("Can have metadata superseded")
    def test_supersede_meta_data_object(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert obj.metadata() == []

        avu1 = AVU("abcde", "12345")
        avu2 = AVU("vwxyz", "567890")
        obj.add_metadata(avu1, avu2)

        assert obj.supersede_metadata(avu1, avu2) == (
            0,
            0,
        ), "nothing is replaced when new all AVUs == all old AVUs"
        assert obj.metadata() == [avu1, avu2]

        assert obj.supersede_metadata(avu1) == (
            0,
            0,
        ), "nothing is replaced when one new AVU is in the AVUs"
        assert obj.metadata() == [avu1, avu2]

        avu3 = AVU("abcde", "88888")
        obj.add_metadata(avu3)

        # Replace avu1, avu3 with avu4, avu5 (leaving avu2 in place)
        avu4 = AVU("abcde", "99999")
        avu5 = AVU("abcde", "00000")
        date = datetime.now(timezone.utc)
        assert obj.supersede_metadata(avu4, avu5, history=True, history_date=date) == (
            2,
            3,
        ), "AVUs sharing an attribute with a new AVU are replaced"

        date = irods.format_timestamp(date)
        history = AVU("abcde_history", f"[{date}] {avu1.value},{avu3.value}")
        expected = [avu2, avu4, avu5, history]
        expected.sort()
        assert obj.metadata() == expected

    @m.it("Can be found by its metadata")
    def test_query_meta_data_object(self, simple_data_object):
        obj = DataObject(simple_data_object)

        avu = AVU("abcde", "12345")
        obj.add_metadata(avu)
        assert obj.metadata() == [avu]

        found = query_metadata(avu, data_object=True, zone=obj.path)
        assert found == [DataObject(simple_data_object)]

    @m.it("Can have access controls added")
    def test_add_ac_data_object(self, simple_data_object):
        zone = "testZone"
        user = "irods"
        obj = DataObject(simple_data_object)
        irods_own = AC(user, Permission.OWN, zone=zone)
        assert obj.acl() == [irods_own]

        assert (
            obj.add_permissions(irods_own) == 0
        ), "nothing is replaced when new ACL == all old ACL"
        assert obj.acl() == [irods_own]

        public_read = AC("public", Permission.READ, zone=zone)
        assert obj.add_permissions(public_read) == 1
        assert obj.acl() == [irods_own, public_read]

    @m.it("Can have its permissions listed")
    def test_list_ac_data_object(self, simple_data_object):
        zone = "testZone"
        obj = DataObject(simple_data_object)
        irods_own = AC("irods", Permission.OWN, zone=zone)
        public_read = AC("public", Permission.READ, zone=zone)
        assert obj.add_permissions(public_read) == 1

        assert obj.acl() == [irods_own, public_read]
        assert obj.acl(user_type="rodsadmin") == [irods_own]
        assert obj.acl(user_type="rodsgroup") == [public_read]

    @m.it("Can have access controls removed")
    def test_rem_ac_data_object(self, simple_data_object):
        zone = "testZone"
        user = "irods"
        obj = DataObject(simple_data_object)
        irods_own = AC(user, Permission.OWN, zone=zone)
        assert obj.acl() == [irods_own]

        public_read = AC("public", Permission.READ, zone=zone)
        assert (
            obj.remove_permissions(public_read) == 0
        ), "nothing is removed when the access control does not exist"
        assert obj.acl() == [irods_own]

        assert obj.add_permissions(public_read) == 1
        assert obj.acl() == [irods_own, public_read]

        assert obj.remove_permissions(public_read) == 1
        assert obj.acl() == [irods_own]


@m.describe("Replica management")
class TestReplicaManagement:
    @m.describe("Data objects with valid replicas")
    @m.context("When trimming would not violate the minimum replica count")
    @m.it("Trims valid replicas")
    def test_trim_valid_replica(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert len(obj.replicas()) == 2

        nv, ni = obj.trim_replicas(min_replicas=1, valid=False, invalid=False)  # noop
        assert nv == ni == 0
        assert len(obj.replicas()) == 2

        nv, ni = obj.trim_replicas(min_replicas=1, valid=True, invalid=False)
        assert nv == 1
        assert ni == 0
        assert len(obj.replicas()) == 1

    @m.context("When trimming would violate the minimum replica count")
    @m.it("Does not trim valid replicas")
    def test_trim_valid_replica_min(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert len(obj.replicas()) == 2

        nv, ni = obj.trim_replicas(min_replicas=2, valid=True, invalid=False)
        assert nv == ni == 0
        assert len(obj.replicas()) == 2

    @m.context("When there are no invalid replicas")
    @m.it("Trims no valid replicas")
    def test_trim_valid_replica_none(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert len(obj.replicas()) == 2

        nv, ni = obj.trim_replicas(min_replicas=2, valid=False, invalid=True)
        assert nv == ni == 0
        assert len(obj.replicas()) == 2

        nv, ni = obj.trim_replicas(min_replicas=1, valid=False, invalid=True)
        assert len(obj.replicas()) == 2
        assert nv == ni == 0

    @m.context("When trimming would remove all replicas")
    @m.it("Raises an error")
    def test_trim_valid_replica_zero(self, simple_data_object, sql_test_utilities):
        obj = DataObject(simple_data_object)
        assert len(obj.replicas()) == 2

        with pytest.raises(RodsError, match="trim error"):
            obj.trim_replicas(min_replicas=0, valid=True, invalid=False)

    @pytest.mark.skipif(
        server_version() >= (4, 3, 2),
        reason=f"fixed in iRODS 4.3.2; version is {server_version()}",
    )
    @m.describe("Data objects with invalid replicas")
    @m.context("When trimming would violate the minimum replica count")
    @m.it("Still trims invalid replicas")
    def test_trim_invalid_replica(
        self, invalid_replica_data_object, sql_test_utilities
    ):
        obj = DataObject(invalid_replica_data_object)
        assert len(obj.replicas()) == 2

        nv, ni = obj.trim_replicas(min_replicas=2, valid=False, invalid=True)
        # I'd prefer not, but this is iRODS' behaviour. This test is just to emphasise
        # this and detect if iRODS changes.
        assert nv == 0
        assert ni == 1
        assert len(obj.replicas()) == 1
        assert obj.replicas()[0].number == 0  # The fixture has replica 1 as invalid


@m.describe("Query Metadata")
class TestQueryMetadata:
    @m.describe("Query Collection namespace")
    @m.context("When a Collection has metadata")
    @m.it("Can be queried by that metadata, only returning collections")
    def test_query_meta_collection(self, annotated_collection, annotated_data_object):
        assert Collection.query_metadata(AVU("no_such_attr1", "no_such_value1")) == []
        assert Collection.query_metadata(AVU("attr1", "value1")) == [
            Collection(annotated_collection)
        ]
        assert Collection.query_metadata(
            AVU("attr1", "value1"),
            AVU("attr2", "value2"),
        ) == [Collection(annotated_collection)]
        assert Collection.query_metadata(
            AVU("attr1", "value1"),
            AVU("attr2", "value2"),
            AVU("attr3", "value3"),
        ) == [Collection(annotated_collection)]
        assert Collection.query_metadata(AVU("attr1", "value%", operator="like")) == [
            Collection(annotated_collection)
        ]

    @m.describe("Query DataObject namespace")
    @m.context("When a DataObject has metadata")
    @m.it("Can be queried by that metadata, only returning data objects")
    def test_query_meta_data_object(self, annotated_collection, annotated_data_object):
        assert DataObject.query_metadata(AVU("no_such_attr1", "no_such_value1")) == []
        assert DataObject.query_metadata(AVU("attr1", "value1")) == [
            DataObject(annotated_data_object)
        ]
        assert DataObject.query_metadata(
            AVU("attr1", "value1"),
            AVU("attr2", "value2"),
        ) == [DataObject(annotated_data_object)]
        assert DataObject.query_metadata(
            AVU("attr1", "value1"),
            AVU("attr2", "value2"),
            AVU("attr3", "value3"),
        ) == [DataObject(annotated_data_object)]
        assert [DataObject(annotated_data_object)] == DataObject.query_metadata(
            AVU("attr1", "value%", operator="like")
        )

    @m.describe("Query both DataObject and Collection namespaces")
    @m.context("When a DataObject and Collections have metadata")
    @m.it("Can be queried by that metadata, returning everything")
    def test_query_meta_all(self, annotated_collection, annotated_data_object):
        assert query_metadata(AVU("no_such_attr1", "no_such_value1")) == []
        assert query_metadata(AVU("attr1", "value1")) == [
            Collection(annotated_collection),
            DataObject(annotated_data_object),
        ]
        assert query_metadata(AVU("attr1", "value1"), AVU("attr2", "value2")) == [
            Collection(annotated_collection),
            DataObject(annotated_data_object),
        ]
        assert query_metadata(
            AVU("attr1", "value1"),
            AVU("attr2", "value2"),
            AVU("attr3", "value3"),
        ) == [
            Collection(annotated_collection),
            DataObject(annotated_data_object),
        ]
        assert query_metadata(AVU("attr1", "value%", operator="like")) == [
            Collection(annotated_collection),
            DataObject(annotated_data_object),
        ]

    @m.context("When a collection is supplied to filter metadata queries")
    @m.it("Filters results by that collection")
    def test_query_meta_collection_filter(self, simple_collection):
        coll = Collection(simple_collection)
        sub1 = Collection(simple_collection / "sub1").create()
        sub2 = Collection(simple_collection / "sub2").create()

        avu = AVU("attr1", "value1")
        sub1.add_metadata(avu)

        assert coll.query_metadata(avu, zone=coll) == [sub1]  # sub1 is a child of coll
        assert sub1.query_metadata(avu, zone=sub1) == [sub1]
        assert sub1.query_metadata(avu, zone=sub2) == []  # sub2 is not a child of sub1


@m.describe("Test special paths (quotes, spaces)")
class TestSpecialPath:
    @m.describe("iRODS paths")
    @m.context("When a Collection has spaces in its path")
    @m.it("Behaves normally")
    def test_collection_space(self, special_paths):
        p = PurePath(special_paths, "a a")
        coll = Collection(p)
        assert coll.exists()
        assert coll.path == p
        assert coll.path.name == "a a"

    @m.context("When a Collection has quotes in its path")
    @m.it("Behaves normally")
    def test_collection_quote(self, special_paths):
        p = PurePath(special_paths, 'b"b')
        coll = Collection(p)
        assert coll.exists()
        assert coll.path == p
        assert coll.path.name == 'b"b'

    @m.context("When a DataObject has spaces in its path")
    @m.it("Behaves normally")
    def test_data_object_space(self, special_paths):
        p = PurePath(special_paths, "y y.txt")
        obj = DataObject(p)
        assert obj.exists()
        assert obj.name == "y y.txt"

    @m.context("When a DataObject has quotes in its path")
    @m.it("Behaves normally")
    def test_data_object_quote(self, special_paths):
        p = PurePath(special_paths, 'z".txt')
        obj = DataObject(p)
        assert obj.exists()
        assert obj.name == 'z".txt'

    @m.context("When a Collection has quotes and spaced in the paths of its contents")
    @m.it("Behaves normally")
    def test_collection_contents(self, special_paths):
        expected = [
            PurePath(special_paths, p).as_posix()
            for p in [
                "a a",
                'b"b',
                "x.txt",
                "y y.txt",
                'z".txt',
                "a a/x.txt",
                "a a/y y.txt",
                'a a/z".txt',
                'b"b/x.txt',
                'b"b/y y.txt',
                'b"b/z".txt',
            ]
        ]

        coll = Collection(special_paths)
        observed = [str(x) for x in coll.contents(recurse=True)]
        assert observed == expected


@m.describe("JSON serialization")
class TestJSON:
    @m.it("Can serialize a collection to JSON")
    def test_collection_json_serialize(self, simple_collection):
        coll = Collection(simple_collection)
        coll.add_metadata(AVU("a", 1), AVU("b", 2), AVU("c", 3))

        kwargs = {"indent": None, "sort_keys": True}

        assert coll.to_json(**kwargs) == json.dumps(
            {
                Baton.COLL: coll.path.as_posix(),
                Baton.AVUS: [
                    {Baton.ATTRIBUTE: "a", Baton.VALUE: "1"},
                    {Baton.ATTRIBUTE: "b", Baton.VALUE: "2"},
                    {Baton.ATTRIBUTE: "c", Baton.VALUE: "3"},
                ],
                Baton.ACCESS: [
                    {Baton.OWNER: "irods", Baton.ZONE: "testZone", Baton.LEVEL: "own"}
                ],
            },
            **kwargs,
        )

    @m.it("Can deserialize a collection from JSON")
    def test_collection_json_deserialize(self, simple_collection):
        coll1 = Collection(simple_collection)
        coll1.add_metadata(AVU("a", 1), AVU("b", 2), AVU("c", 3))

        json_str = coll1.to_json(indent=None, sort_keys=True)

        coll2 = Collection.from_json(json_str)
        assert coll2.path == coll1.path
        assert coll2.metadata() == coll1.metadata()
        assert coll2.acl() == coll1.acl()

    @m.it("Can serialize a data object to JSON")
    def test_data_object_json_serialize(
        self, irods_groups, irods_users, simple_data_object
    ):
        obj = DataObject(simple_data_object)
        assert obj.connected()

        metadata = [AVU("a", 1), AVU("b", 2), AVU("c", 3)]
        acl = [
            AC("user1", Permission.OWN, zone="testZone"),
            AC("user2", Permission.WRITE, zone="testZone"),
            AC("user3", Permission.WRITE, zone="testZone"),
            AC("public", Permission.READ, zone="testZone"),
        ]

        obj.add_metadata(*metadata)
        obj.add_permissions(*acl)

        kwargs = {"indent": None, "sort_keys": True}

        assert obj.to_json(**kwargs) == json.dumps(
            {
                Baton.COLL: obj.path.as_posix(),
                Baton.OBJ: obj.name,
                Baton.AVUS: [
                    {Baton.ATTRIBUTE: "a", Baton.VALUE: "1"},
                    {Baton.ATTRIBUTE: "b", Baton.VALUE: "2"},
                    {Baton.ATTRIBUTE: "c", Baton.VALUE: "3"},
                ],
                Baton.ACCESS: [
                    {Baton.OWNER: "irods", Baton.ZONE: "testZone", Baton.LEVEL: "own"},
                    {
                        Baton.OWNER: "public",
                        Baton.ZONE: "testZone",
                        Baton.LEVEL: "read",
                    },
                    {Baton.OWNER: "user1", Baton.ZONE: "testZone", Baton.LEVEL: "own"},
                    {
                        Baton.OWNER: "user2",
                        Baton.ZONE: "testZone",
                        Baton.LEVEL: "write",
                    },
                    {
                        Baton.OWNER: "user3",
                        Baton.ZONE: "testZone",
                        Baton.LEVEL: "write",
                    },
                ],
                Baton.SIZE: 555,
                Baton.CHECKSUM: "39a4aa291ca849d601e4e5b8ed627a04",
            },
            **kwargs,
        )

    @m.it("Can deserialize a data object from JSON")
    def test_data_object_json_deserialize(self, simple_data_object):
        metadata = [AVU("a", 1), AVU("b", 2), AVU("c", 3)]
        acl = [
            AC("hello", Permission.WRITE, zone="testZone"),
            AC("irods", Permission.OWN, zone="testZone"),
            AC("public", Permission.READ, zone="testZone"),
        ]

        obj1 = DataObject(simple_data_object, pool=None)

        assert not obj1.connected()
        with pytest.raises(BatonError, match="operation 'checksum'"):
            obj1.checksum()
        with pytest.raises(BatonError, match="operation 'size'"):
            obj1.size()

        obj1.add_metadata(*metadata)
        obj1.add_permissions(*acl)

        json_str = obj1.to_json(indent=None, sort_keys=True)
        obj2 = DataObject.from_json(json_str)

        assert not obj2.connected()
        with pytest.raises(BatonError, match="operation 'checksum'"):
            obj2.checksum()
        with pytest.raises(BatonError, match="operation 'size'"):
            obj2.size()

        assert obj2.path == obj1.path
        assert obj2.name == obj1.name
        assert obj2.metadata() == metadata
        assert obj2.acl() == acl


@m.describe("File Checksumming")
class TestFileChecksumming:
    @m.context("When calculate checksum of file spanning one chunk")
    @m.it("Returns checksum matching one independently calculated with md5sum")
    def test_calculate_file_checksum_one_chunk(self, tmp_path):
        p = tmp_path / "1K.bin"
        self._create_nul_file(p, 1024) # One chunk, partial

        assert _calculate_file_checksum(p) == "0f343b0931126a20f133d67c2b018a3b"

    @m.context("When calculate checksum of file spanning two chunks, one partial")
    @m.it("Returns checksum matching one independently calculated with md5sum")
    def test_calculate_file_checksum_two_chunks(self, tmp_path):
        p = tmp_path / "1025K.bin"
        self._create_nul_file(p, 1025 * 1024) # Two chunks, one partial

        assert _calculate_file_checksum(p) == "3d283316360c56857e7c212cfecbbd83"

    def _create_nul_file(self, path: Path, size: int):
        with path.open("wb") as f:
            f.write(b"\x00" * size)

def _check_local_paths(local_root, remote_root, remote_items) -> list[Path]:
    local_paths = []
    for r in remote_items:
        p = local_root / Path(str(r)).relative_to(remote_root)
        assert p.exists(), f"Local path '{p}' does not exist"

        if isinstance(r, Collection):
            assert p.is_dir()
        else:
            assert p.is_file()
        local_paths.append(p)

    return local_paths
