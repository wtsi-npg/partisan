# -*- coding: utf-8 -*-
#
# Copyright © 2025 Genome Research Ltd. All rights reserved.
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

from pytest import mark as m

from partisan.icommands import icp, iget, imkdir, iput, iquest, irm
from partisan.irods import Collection, DataObject


@m.describe("icommand wrappers")
class TestICommands:
    @m.context("When querying a specific zone")
    @m.it("Filters the zone log line from the output")
    def test_iquest_zone_filter(self):
        output = iquest("-z", "testZone", "select COLL_NAME").splitlines()
        assert output
        assert not output[0].startswith("Zone is testZone")

    @m.context("When a query returns no results")
    @m.it("Returns the empty string")
    def test_iquest_no_results(self):
        output = iquest("select COLL_NAME where COLL_NAME = 'no_such_collection'")
        assert output == ""


FILENAMES = [
    "plain.txt",
    "with spaces.txt",
    "single'quote.txt",
    'double"quote.txt',
    r"back\slash.txt",
    "dollar$HOME.txt",
    "backtick`printf expanded`.txt",
    "substitution$(printf expanded).txt",
    "parentheses(1).txt",
    "ampersand&name.txt",
    "semicolon;name.txt",
    "wildcard*?[ab].txt",
    "unicode-café-λ.txt",
]
FILE_IDS = [
    "plain",
    "spaces",
    "single-quote",
    "double-quote",
    "backslash",
    "dollar",
    "backticks",
    "command-substitution",
    "parentheses",
    "ampersand",
    "semicolon",
    "wildcards",
    "unicode",
]


@m.describe("icommand wrappers should preserve paths with special characters")
@m.parametrize("filename", FILENAMES, ids=FILE_IDS)
class TestICommandPaths:
    @m.context("When creating and removing a collection with special characters")
    @m.it("Preserves the literal collection name")
    def test_collection_path(self, simple_collection, filename):
        remote_path = simple_collection / filename

        imkdir(remote_path)
        assert [item.path for item in Collection(simple_collection).contents()] == [
            remote_path
        ]

        # Include a child to exercise recursive removal of the named collection.
        imkdir(remote_path / "child")

        irm(remote_path, force=True, recurse=True)
        assert Collection(simple_collection).contents() == []

    @m.context("When uploading and downloading paths with special characters")
    @m.it("Preserves literal local and remote names and file contents")
    def test_transfer_paths(self, tmp_path, simple_collection, filename):
        upload_dir = tmp_path / "upload"
        download_dir = tmp_path / "download"
        upload_dir.mkdir()
        download_dir.mkdir()

        source = upload_dir / filename
        destination = download_dir / filename
        payload = b"Wrapper argument preservation\n\x00\xff\n"
        source.write_bytes(payload)
        remote_path = simple_collection / filename

        iput(source, remote_path)
        assert Collection(simple_collection).contents() == [
            DataObject(simple_collection / filename)
        ]

        iget(remote_path, destination)
        assert list(download_dir.iterdir()) == [destination]
        assert destination.read_bytes() == payload

        irm(remote_path, force=True)
        assert Collection(simple_collection).contents() == []

    @m.context("When copying remote paths with special characters")
    @m.it("Preserves both remote names and the copied contents")
    def test_copy_paths(self, tmp_path, simple_collection, filename):
        source = tmp_path / "source.txt"
        payload = b"Copied through the container wrapper\n"
        source.write_bytes(payload)
        remote_source = simple_collection / filename
        copied_name = "copy-" + filename
        remote_copy = simple_collection / copied_name
        iput(source, remote_source)

        icp(remote_source, remote_copy)
        assert Collection(simple_collection).contents() == sorted(
            [
                DataObject(simple_collection / copied_name),
                DataObject(simple_collection / filename),
            ]
        )

        destination = tmp_path / "download.txt"

        iget(remote_copy, destination)
        assert destination.read_bytes() == payload

        irm(remote_source, force=True)
        irm(remote_copy, force=True)
        assert Collection(simple_collection).contents() == []
