# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021, 2026 Genome Research Ltd. All rights reserved.
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

import pytest
from pytest import mark as m

from partisan.exception import BatonError
from partisan.irods import Baton, client_version


@m.describe("Baton")
class TestBatonClient:
    @m.context("When queried for its version")
    @m.it("Returns the client version")
    def test_client_version(self):
        version = client_version()
        assert len(version) == 3

    @m.context("When created")
    @m.it("Is not running")
    def test_create_baton_client(self):
        c = Baton()
        assert not c.is_running()

    @m.it("Can be started and stopped")
    def test_start_baton_client(self):
        c = Baton()
        c.start()
        assert c.is_running()
        c.stop()
        assert not c.is_running()

    @m.context("When stopped")
    @m.it("Can be re-started")
    def test_restart_baton_client(self):
        c = Baton()
        c.start()
        assert c.is_running()
        c.stop()
        assert not c.is_running()

        # Re-start
        c.start()
        assert c.is_running()
        c.stop()
        assert not c.is_running()

    @m.context("When running")
    @m.it("Can handle a sequence of requests on one connection")
    def test_multiple_requests(self, simple_collection):
        c = Baton()
        c.start()
        assert c.is_running()
        pid = c.pid()

        arg = {Baton.COLL: simple_collection.as_posix()}
        assert c.list(arg) == [arg]
        assert c.list(arg) == [arg]
        assert c.list(arg) == [arg]

        assert c.is_running()
        assert c.pid() == pid
        c.stop()
        assert not c.is_running()

    @m.context("When send fails in the worker thread")
    @m.it("Propagates the send exception")
    def test_propagates_send_exception(self):
        c = Baton()
        c.start()
        try:
            with pytest.raises(TypeError, match="not JSON serializable"):
                c.list({Baton.COLL: object()}, timeout=1, tries=1)
            assert c.is_running()
            assert c.is_healthy()
        finally:
            c.stop()

    @m.context("When a fatal baton error occurs")
    @m.it("Marks the client unhealthy and stops it")
    def test_fatal_error_behaviour(self):
        class FatalBaton(Baton):
            def __init__(self):
                super().__init__()
                self.stop_calls = 0

            def is_running(self):
                return True

            def stop(self):
                self.stop_calls += 1

            def _send(self, envelope):
                raise BatonError("Fatal send failure")

        c = FatalBaton()

        with pytest.raises(BatonError, match="Fatal send failure"):
            c._execute(Baton.LIST, {}, {}, timeout=1, tries=1)

        assert not c.is_healthy()
        assert c.stop_calls == 1
