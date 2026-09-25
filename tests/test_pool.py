# -*- coding: utf-8 -*-
#
# Copyright © 2021, 2026 Genome Research Ltd. All rights reserved.
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

from queue import Empty

import pytest
from pytest import mark as m

from partisan.irods import Baton, client, client_pool


@m.describe("BatonPool")
class TestBatonPool(object):
    @m.context("When created")
    @m.it("Is open")
    def test_pool_is_open(self):
        with client_pool() as p:
            assert p.is_open()

    @m.context("When the pool is open")
    @m.it("Yields up to maxsize running clients")
    def test_get_clients(self):
        with client_pool(maxsize=2) as p:
            with client(p) as c1:
                assert c1.is_running()
                with client(p) as c2:
                    assert c2.is_running()

    @m.context("After getting maxsize clients")
    @m.it("Getting another client times out")
    def test_get_clients_timeout(self):
        with client_pool(maxsize=1) as p:
            with client(p) as c1:
                assert c1.is_running()
                with pytest.raises(Empty):
                    with client(p, timeout=1) as _:
                        pass

    @m.context("When the pool is closed")
    @m.it("Stops its clients")
    def test_pool_close(self):
        pool_size = 2
        clients = list()
        with client_pool(maxsize=pool_size) as p:
            for _ in range(pool_size):
                with client(p) as c:
                    clients.append(c)
                    assert c.is_running()

        assert len(clients) == pool_size
        for c in clients:
            assert not c.is_running()

    @m.context("When putting an unhealthy client into the pool")
    @m.it("Replaces it with a new Baton instance")
    def test_pool_put_replaces_unhealthy(self):
        with client_pool(maxsize=1) as p:
            _ = p._queue.get_nowait()

            bad = Baton()
            bad._mark_unhealthy()
            p.put(bad)
            replacement = p._queue.get_nowait()

            assert isinstance(replacement, Baton)
            assert replacement is not bad
            assert replacement.is_healthy()

    @m.context("When putting a stopped but healthy client into the pool")
    @m.it("Returns it to the pool for lazy restart on next get")
    def test_pool_put_keeps_stopped_healthy(self):
        with client_pool(maxsize=1) as p:
            _ = p._queue.get_nowait()

            stopped = Baton()
            assert stopped.is_healthy()
            assert not stopped.is_running()

            p.put(stopped)
            returned = p._queue.get_nowait()
            assert returned is stopped
