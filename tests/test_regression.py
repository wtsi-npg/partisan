# -*- coding: utf-8 -*-
#
# Copyright © 2026 Genome Research Ltd. All rights reserved.
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

from partisan.irods import Collection, DataObject


class TestRegression:
    @m.context(
        "When overwriting an existing data object with an updated version "
        "where checksum checking, verification and fill are enabled"
    )
    @m.it("Should overwrite the existing data object with the updated version")
    def test_fill_mod_cmp_checksum(self, tmp_path, simple_collection):
        """
        To replicate the issue:

        1. Upload a file with checksum verification and comparison with fill enabled
        2. Modify the file locally
        3. Repeat the upload to the same location

        A checksum mismatch is raised, whereas the file should be overwritten cleanly.

        See https://github.com/wtsi-npg/npg-irods-python/issues/513
        """

        file1 = tmp_path / "file1.txt"
        with open(file1, "w") as f:
            print("line1", file=f)

        obj = DataObject(simple_collection / "file1.txt").put(
            local_path=file1, compare_checksums=True, verify_checksum=True, fill=True
        )
        assert obj.exists()
        assert obj.checksum() == "1ddab9058a07abc0db2605ab02a61a00"

        # Modify the file locally
        with open(file1, "w+") as f:
            print("line2", file=f)

        obj = DataObject(simple_collection / "file1.txt").put(
            local_path=file1, compare_checksums=True, verify_checksum=True, fill=True
        )

        assert obj.exists()
        assert obj.checksum() == "8475275e7210a7821f93475ff52669d3"
