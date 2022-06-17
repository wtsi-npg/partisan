# -*- coding: utf-8 -*-
#
# Copyright © 2021 Genome Research Ltd. All rights reserved.
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
from pathlib import Path

from partisan import irods
from partisan.irods import (
    AC,
    AVU,
    Collection,
    DataObject,
    Permission,
    client_pool,
)


class TestExamples(object):
    """These are the examples from the README documentation. They use some dummy data
    from the package data directory.

    partisan uses the baton client which must available to Python. Normally
    this is achieved by making sure the `baton-do` executable is on your `PATH`.

    Like the iRODS client icommands, `baton` uses the environment variable
    `IRODS_ENVIRONMENT_FILE` to locate an iRODS configuration file. You should
    ensure that this set before attempting to connect to iRODS.

    The assertions are correct for a test system where the zone is named `testZone`,
    the user is `irods`, the iRODS resource is a replication resource with 2 replicas,
    and the server is configured to use MD5 checksums.
    """

    def test_collection_examples(self, ont_gridion):

        # Given some collections and data objects already in iRODS ...

        # To make an object representing a collection, pass a string or os.PathLike
        # to the Collection constructor. iRODS paths are virtual, existing only in
        # the iRODS catalog database. A Collection is a Python os.PathLike object.
        coll = Collection(ont_gridion)
        assert coll.exists(), "The collection exists"

        # We can examine the collection permissions. Note that you may get a
        # different result on your iRODS, depending on your user and zone.
        assert coll.permissions() == [AC("irods", Permission.OWN, zone="testZone")]

        # We can list the collection, which returns a new instance
        assert coll.list() == coll

        # We can examine the collection's immediate contents.
        assert coll.contents() == [
            Collection(ont_gridion / "66")
        ], "The collection contains one sub-collection"

        # We can examine the collection's contents recursively. If you print the
        # contents, you'll see that collections sort before data objects.
        assert len(coll.contents(recurse=True)) == 26

        # We can examine the collection's metadata.
        assert coll.metadata() == [], "The collection has no metadata"

        # We can examine the collections permissions. Note that you may get a
        # different result on your iRODS, depending on your user and zone.
        assert coll.permissions() == [AC("irods", Permission.OWN, zone="testZone")]

    def test_data_object_examples(self, ont_gridion, tmp_path):
        # To make an object representing a data object, pass a string or os.PathLike
        # to the DataObject constructor. iRODS paths are virtual, existing only in
        # the iRODS catalog database. A DataObject is a Python os.PathLike object.
        obj = DataObject(
            ont_gridion
            / "66"
            / "DN585561I_A1"
            / "20190904_1514_GA20000_FAL01979_43578c8f"
            / "report.md"
        )
        assert obj.exists(), "The data object exists"

        # We can examine the data object permissions. Note that you may get a
        # different result on your iRODS, depending on your user and zone.
        assert obj.permissions() == [AC("irods", Permission.OWN, zone="testZone")]

        # We can compare the remote object's checksum with its expected value.
        assert obj.checksum() == "c462fb84625c26ba15ecdf62e15a9560"

        # We can list the object, which returns a new instance
        assert obj.list() == obj

        # We can examine any replicas of the object. Note that you may get a
        # different result on your iRODS, depending on whether you are using
        # replication, or not.
        assert len(obj.replicas()) == 2
        for r in obj.replicas():
            assert r.checksum == "c462fb84625c26ba15ecdf62e15a9560"
            assert r.valid

        # We can see the expected size, according to the iRODS IES database, in bytes.
        assert obj.size() == 1369

        # We can get the data object to a local file and confirm that the data within
        # is the expected size and matches the expected checksum.

        local_path = Path(tmp_path, obj.name)
        obj.get(local_path)

        with open(local_path, "rb") as f:
            m = hashlib.md5()
            n = 0
            for b in iter(lambda: f.read(1024), b""):
                n += len(b)
                m.update(b)

            assert n == 1369
            assert m.hexdigest() == "c462fb84625c26ba15ecdf62e15a9560"

        # For small text files, we can also read a data object directly into memory.
        assert (
            hashlib.md5(obj.read().encode("utf-8")).hexdigest()
            == "c462fb84625c26ba15ecdf62e15a9560"
        )

        # We can examine the data object's metadata.
        assert obj.metadata() == [], "The data object has no metadata"

        # We can add some metadata to the data object. Let's add two AVUs (Attribute,
        # Value, Unit tuples).
        obj.add_metadata(
            AVU("sample_id", "sample_1"), AVU("experiment_id", "experiment_1")
        )

        # Note that metadata returned by this method are sorted by AVU attribute and
        # value.
        assert obj.metadata() == [
            AVU("experiment_id", "experiment_1"),
            AVU("sample_id", "sample_1"),
        ]

        # Now that there are metadata on the data object, we can find it by query.
        # Here we query on experiment_id alone.
        assert DataObject.query_metadata(AVU("experiment_id", "experiment_1")) == [obj]

        # Note that in iRODS, collection and data object metadata are separate and
        # must be queried independently. In the partisan API, a query using
        # `DataObject.query_metadata()` searches only data object metadata, while
        # `Collection.query_metadata()` searches only collection metadata.
        assert Collection.query_metadata(AVU("experiment_id", "experiment_1")) == []

        # The function `partisan.irods.query_metadata()` is available to search both
        # collection and data object metadata to return combined results.
        assert irods.query_metadata(AVU("experiment_id", "experiment_1")) == [obj]

    def test_pool_examples(self, ont_gridion):

        # partisan uses a small pool (the default is 4) of BatonClient instances to
        # serve requests. This pool is created automatically and is passed to the
        # constructors of Collections and DataObjects by default. If you would like an
        # alternative client management strategy, you can do this by creating your
        # own pool.

        # Let's create a Collection using the default pool.
        coll = Collection(ont_gridion)

        # Now let's create a Collection using a pool which contains only one
        # BatonClient. partisan provides a context manager to help with this. When a
        # pool goes out of scope it will automatically be closed, stopping any
        # BatonClients within it and terminating any baton-do processes they may
        # have started.
        with client_pool(maxsize=1) as pool:
            coll = Collection(ont_gridion, pool=pool)

    def test_timeout_examples(self, ont_gridion):

        # In its API methods that communicate with the iRODS server, partisan
        # provides the two keyword arguments `timeout` and `tries` to control the
        # timeout for the operation (in seconds) and how many attempts should be made
        # to carry it out before raising an exception. The default values for these
        # are `None` (meaning do not time out) and `1` (meaning make a single attempt).

        # Let's check if a collection exists, timing out if a response takes longer
        # than 10 seconds and allowing 3 tries at that (each timing out after 10
        # seconds).
        Collection(ont_gridion).exists(timeout=10, tries=3)

        # A timeout may occur for one of two reasons. Firstly, if no BatonClient is
        # available before the timeout expires and secondly, if the iRODS server takes
        # longer than the timeout to respond with a result. partisan does not
        # distinguish between these circumstances.
