# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021, 2023 Genome Research Ltd. All rights reserved.
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
import os.path
from datetime import datetime
from pathlib import PurePath

import pytest
from pytest import mark as m

from partisan import irods
from partisan.exception import BatonError, RodsError
from partisan.irods import (
    AC,
    AVU,
    Collection,
    DataObject,
    Permission,
    User,
    current_user,
    make_rods_item,
    query_metadata,
    rods_path_type,
    rods_user,
)


def irods_version():
    version = os.environ.get("IRODS_VERSION", "4.2.7")
    [major, minor, patch] = [int(elt) for elt in version.split(".")]
    return major, minor, patch


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


@m.describe("AC")
class TestAC:
    @m.describe("Comparison")
    def test_compare_acs_equal(self):
        user = "irods"
        zone = "testZone"

        assert AC(user, Permission.OWN, zone=zone) == AC(
            user, Permission.OWN, zone=zone
        )

        assert AC(user, Permission.OWN, zone=zone) != AC(
            user, Permission.READ, zone=zone
        )

        assert AC(user, Permission.OWN, zone=zone) != AC(
            "public", Permission.OWN, zone=zone
        )

    def test_compare_acs_lt(self):
        user = "irods"
        zone = "testZone"

        assert AC(user, Permission.OWN, zone=zone) < AC(
            "public", Permission.OWN, zone=zone
        )

        assert AC(user, Permission.NULL, zone=zone) < AC(
            user, Permission.OWN, zone=zone
        )

    def test_compare_acs_sort(self):
        zone = "testZone"
        acl = [
            AC("zzz", Permission.OWN, zone=zone),
            AC("aaa", Permission.WRITE, zone=zone),
            AC("aaa", Permission.READ, zone=zone),
            AC("zyy", Permission.READ, zone=zone),
            AC("zyy", Permission.OWN, zone=zone),
        ]
        acl.sort()

        assert acl == [
            AC("aaa", Permission.READ, zone=zone),
            AC("aaa", Permission.WRITE, zone=zone),
            AC("zyy", Permission.OWN, zone=zone),
            AC("zyy", Permission.READ, zone=zone),
            AC("zzz", Permission.OWN, zone=zone),
        ]


@m.describe("AVU")
class TestAVU:
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

    @m.it("Can be found by its metadata")
    def test_meta_query_collection(self, simple_collection):
        coll = Collection(simple_collection)

        avu = AVU("abcde", "12345")
        coll.add_metadata(avu)
        assert coll.metadata() == [avu]

        found = query_metadata(avu, collection=True, zone=coll)
        assert found == [Collection(simple_collection)]

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
        size = obj.get(local_path)
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

    @m.it("Has a checksum")
    def test_get_checksum(self, simple_data_object):
        obj = DataObject(simple_data_object)
        assert obj.checksum() == "39a4aa291ca849d601e4e5b8ed627a04"

    @m.it("Can have its checksum verified as good")
    def test_verify_checksum_good(self, simple_data_object):
        obj = DataObject(simple_data_object)

        # Note that in iRODS >= 4.2.10, this always passes, even if the remote file
        # is the wrong size of has a mismatching checksum, because of this iRODS bug:
        # https://github.com/irods/irods/issues/5843
        assert obj.checksum(verify_checksum=True)

    @m.it("Can have its checksum verified as bad")
    @pytest.mark.skipif(
        irods_version() <= (4, 2, 10), reason="requires iRODS server >4.2.10"
    )
    def test_verify_checksum_bad(self, invalid_checksum_data_object):
        obj = DataObject(invalid_checksum_data_object)

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
        (r0, r1) = obj.replicas()
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

        assert obj.timestamp() == min([o.modified for o in obj.replicas()])

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

    @m.it("Can have metadata replaced")
    def test_repl_meta_data_object(self, simple_data_object):
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
        date = datetime.utcnow()
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
        assert obj.acl() == [AC(user, Permission.OWN, zone=zone)]

        assert (
            obj.add_permissions(AC(user, Permission.OWN, zone=zone)) == 0
        ), "nothing is replaced when new ACL == all old ACL"
        assert obj.acl() == [AC(user, Permission.OWN, zone=zone)]

        assert obj.add_permissions(AC("public", Permission.READ, zone=zone)) == 1
        assert obj.acl() == [
            AC(user, Permission.OWN, zone=zone),
            AC("public", Permission.READ, zone=zone),
        ]

    @m.it("Can have access controls removed")
    def test_rem_ac_data_object(self, simple_data_object):
        zone = "testZone"
        user = "irods"
        obj = DataObject(simple_data_object)
        assert obj.acl() == [AC(user, Permission.OWN, zone=zone)]

        assert (
            obj.remove_permissions(AC("public", Permission.READ, zone=zone)) == 0
        ), "nothing is removed when the access control does not exist"
        assert obj.acl() == [AC(user, Permission.OWN, zone=zone)]

        assert obj.add_permissions(AC("public", Permission.READ, zone=zone)) == 1
        assert obj.acl() == [
            AC(user, Permission.OWN, zone=zone),
            AC("public", Permission.READ, zone=zone),
        ]

        assert obj.remove_permissions(AC("public", Permission.READ, zone=zone)) == 1
        assert obj.acl() == [AC(user, Permission.OWN, zone=zone)]


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
        assert [] == Collection.query_metadata(AVU("no_such_attr1", "no_such_value1"))
        assert [Collection(annotated_collection)] == Collection.query_metadata(
            AVU("attr1", "value1")
        )
        assert [Collection(annotated_collection)] == Collection.query_metadata(
            AVU("attr1", "value1"), AVU("attr2", "value2")
        )
        assert [Collection(annotated_collection)] == Collection.query_metadata(
            AVU("attr1", "value1"), AVU("attr2", "value2"), AVU("attr3", "value3")
        )

    @m.describe("Query DataObject namespace")
    @m.context("When a DataObject has metadata")
    @m.it("Can be queried by that metadata, only returning data objects")
    def test_query_meta_data_object(self, annotated_collection, annotated_data_object):
        assert [] == DataObject.query_metadata(AVU("no_such_attr1", "no_such_value1"))
        assert [DataObject(annotated_data_object)] == DataObject.query_metadata(
            AVU("attr1", "value1")
        )
        assert [DataObject(annotated_data_object)] == DataObject.query_metadata(
            AVU("attr1", "value1"), AVU("attr2", "value2")
        )
        assert [DataObject(annotated_data_object)] == DataObject.query_metadata(
            AVU("attr1", "value1"), AVU("attr2", "value2"), AVU("attr3", "value3")
        )

    @m.describe("Query both DataObject and Collection namespaces")
    @m.context("When a DataObjects and Collections have metadata")
    @m.it("Can be queried by that metadata, only returning everything")
    def test_query_meta_all(self, annotated_collection, annotated_data_object):
        assert [] == query_metadata(AVU("no_such_attr1", "no_such_value1"))
        assert [
            Collection(annotated_collection),
            DataObject(annotated_data_object),
        ] == query_metadata(AVU("attr1", "value1"))
        assert [
            Collection(annotated_collection),
            DataObject(annotated_data_object),
        ] == query_metadata(AVU("attr1", "value1"), AVU("attr2", "value2"))
        assert [
            Collection(annotated_collection),
            DataObject(annotated_data_object),
        ] == query_metadata(
            AVU("attr1", "value1"), AVU("attr2", "value2"), AVU("attr3", "value3")
        )


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
