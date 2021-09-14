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

from pytest import mark as m

from partisan.irods import Baton, Collection
from irods_fixture import (
    baton_session,
    simple_collection,
)

_ = baton_session
_ = simple_collection


@m.describe("Baton")
class TestBatonClient(object):
    @m.context("When created")
    @m.it("Is not running")
    def test_create_baton_client(self):
        client = Baton()
        assert not client.is_running()

    @m.it("Can be started and stopped")
    def test_start_baton_client(self):
        client = Baton()
        client.start()
        assert client.is_running()
        client.stop()
        assert not client.is_running()

    @m.context("When stopped")
    @m.it("Can be re-started")
    def test_restart_baton_client(self, simple_collection):
        client = Baton()
        client.start()
        assert client.is_running()
        client.stop()
        assert not client.is_running()
        # Re-start
        client.start()
        assert client.is_running()
        # Try an operation
        coll = Collection(client, simple_collection)
        assert coll.exists()
        client.stop()
