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

from enum import Enum, EnumMeta, unique

"""Metadata controlled vocabularies used in iRODS."""


def with_namespace(ns: str):
    """Returns a metaclass that may be used to add a property 'namespace' to
    an Enum. The metaclass is a subclass of EnumMeta in order to permit
    this.

    e.g.
       class MyEnum(Enum, metaclass=with_namespace("mynamespace")):
          X = "x"
          Y = "y"

       MyEnum.namespace == "mynamespace" is True


    This allows the namespace of the Enum to be accessed in the same
    syntactic style as the Enum members.
    """

    class WithNamespace(EnumMeta):
        """Metaclass adding a 'namespace' property an Enum"""

        @property
        def namespace(self):
            return ns

    return WithNamespace


class AsValueEnum(Enum):
    """ "An Enum whose member representation is equal to their value attribute."""

    def __repr__(self):
        return self.value


@unique
class DublinCore(AsValueEnum, metaclass=with_namespace("dcterms")):
    """Dublin Core metadata. See
    https://dublincore.org/specifications/dublin-core/dcmi-terms/"""

    AUDIENCE = "audience"
    CREATED = "created"
    CREATOR = "creator"
    IDENTIFIER = "identifier"
    MODIFIED = "modified"
    PUBLISHER = "publisher"
    TITLE = "title"
