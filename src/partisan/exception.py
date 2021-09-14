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


class RodsError(Exception):
    """Exception wrapping an error raised by the iRODS server."""

    def __init__(self, *args):
        super().__init__(*args)
        self.message = args[0] if len(args) > 0 else ""
        self.code = args[1] if len(args) > 1 else -1

    def __repr__(self):
        return str(self.code)

    def __str__(self):
        return f"<RodsError: {self.message} code: {self.code}>"


class BatonError(Exception):
    """The base class of all exceptions originating in this package."""

    pass


class BatonArgumentError(BatonError):
    """Exception raised when invalid arguments are passed to a baton.py-do operation."""

    pass


class InvalidJSONError(BatonError):
    """Exception raised when a baton.py-do JSON document is invalid."""

    pass


class InvalidEnvelopeError(InvalidJSONError):
    """Exception raised when the baton.py-do JSON document envelope has invalid
    structure, such as missing mandatory properties."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.message = args[0] if len(args) > 0 else ""
        self.envelope = kwargs.get("envelope")

    def __str__(self):
        return f"<InvalidEnvelopeError: {self.message}>"
