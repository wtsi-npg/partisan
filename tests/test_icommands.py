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

from partisan.icommands import iquest


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
