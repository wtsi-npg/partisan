# -*- coding: utf-8 -*-
#
# Copyright © 2025, Genome Research Ltd. All rights reserved.
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

import json
from datetime import datetime, timezone, timedelta

from pytest import mark as m

from partisan.irods import BatonJSONEncoder, Timestamp, Baton


@m.describe("BatonJSONEncoder")
class TestBatonJSONEncoder:
    @m.context("When serialising timestamp objects")
    @m.it("Preserves timezone information during UTC conversion for created timestamps")
    def test_timestamp_serialisation_created_with_timezone(self):
        # Timezone-aware datetime (EST timezone, UTC-5)
        est_timezone = timezone(timedelta(hours=-5))
        est_time = datetime(2023, 12, 25, 15, 30, 45, tzinfo=est_timezone)

        ts = Timestamp(est_time, Timestamp.Event.CREATED)
        encoded = BatonJSONEncoder().default(ts)

        # Time difference preserved
        assert encoded[Baton.CREATED] == "2023-12-25T20:30:45Z"

    @m.context("When serialising timestamp objects")
    @m.it("Handles timezone-naive datetimes without conversion")
    def test_timestamp_serialisation_naive_datetime(self):
        # Naive datetime (no timezone info)
        naive_time = datetime(2023, 3, 10, 12, 0, 0)

        ts = Timestamp(naive_time, Timestamp.Event.CREATED)
        encoded = BatonJSONEncoder().default(ts)

        # No conversion since it's naive
        assert encoded[Baton.CREATED] == "2023-03-10T12:00:00Z"

    @m.context("When serialising timestamp objects")
    @m.it("Handles UTC timezone correctly")
    def test_timestamp_serialisation_utc_timezone(self):
        # UTC datetime
        utc_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        ts = Timestamp(utc_time, Timestamp.Event.CREATED)
        encoded = BatonJSONEncoder().default(ts)

        # Already UTC, no conversion needed
        assert encoded[Baton.CREATED] == "2023-01-01T00:00:00Z"

    @m.context("When using the encoder in JSON serialisation")
    @m.it("Works correctly with json.dumps")
    def test_full_json_serialisation_with_timestamp(self):
        # Create a timezone-aware timestamp
        cet_timezone = timezone(timedelta(hours=1))
        cet_time = datetime(2023, 7, 4, 16, 45, 0, tzinfo=cet_timezone)
        ts = Timestamp(cet_time, Timestamp.Event.CREATED)

        json_str = json.dumps(ts, cls=BatonJSONEncoder)
        parsed_value = json.loads(json_str)

        # Expected UTC time (16:45:00 CET = 15:45:00 UTC)
        assert parsed_value[Baton.CREATED] == "2023-07-04T15:45:00Z"
