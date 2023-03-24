# -*- coding: utf-8 -*-
#
# Copyright © 2022 Genome Research Ltd. All rights reserved.
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

from pytest import mark as m

from partisan.metadata import AsValueEnum, DublinCore


@m.describe("Metadata")
class TestMetadata(object):
    @m.describe("AsValue Enum")
    def test_as_value_enum(self):
        class TestEnum(AsValueEnum):
            A = "a"

        assert TestEnum.A.value == TestEnum.A.__repr__()
        assert TestEnum.A.value == str(TestEnum.A)
        assert TestEnum.values() == ["a"]

    @m.describe("When a member of the Dublin Core enum is stringified")
    @m.it("Has the expected namespace")
    def test_dublin_core(self):
        assert DublinCore.namespace == "dcterms"

        for elt in DublinCore:
            assert str(elt) == f"{DublinCore.namespace}:{elt.value}"
