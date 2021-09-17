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

from __future__ import annotations  # Will not be needed in Python 3.10

import json
import subprocess
from abc import abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from datetime import datetime
from enum import Enum, unique
from functools import total_ordering
from os import PathLike
from pathlib import Path, PurePath
from typing import Any, Dict, List, Tuple, Union

from structlog import get_logger

from partisan.exception import (
    BatonError,
    InvalidEnvelopeError,
    InvalidJSONError,
    RodsError,
)

log = get_logger(__package__)

"""This module provides a basic API for accessing iRODS using the native
iRODS client 'baton' (https://github.com/wtsi-npg/baton).
"""


@unique
class Permission(Enum):
    """The kinds of data access permission available to iRODS users."""

    NULL = "null"
    OWN = ("own",)
    READ = ("read",)
    WRITE = ("write",)


@total_ordering
class AC(object):
    """AC is an iRODS access control.

    ACs may be sorted, where they will sorted lexically, first by
    zone, then by user, and finally by permission.
    """

    SEPARATOR = "#"

    def __init__(self, user: str, perm: Permission, zone=None):
        if user is None:
            raise ValueError("user may not be None")

        if user.find(AC.SEPARATOR) >= 0:
            raise ValueError(
                f"User '{user}' should not contain a zone suffix. Please use the "
                "zone= keyword argument to set a zone"
            )

        if zone:
            if zone.find(AC.SEPARATOR) >= 0:
                raise ValueError(f"Zone '{zone}' contained '{AC.SEPARATOR}'")
        self.user = user
        self.zone = zone
        self.perm = perm

    def __hash__(self):
        return hash(self.user) + hash(self.zone) + hash(self.perm)

    def __eq__(self, other):
        return (
            isinstance(other, AC)
            and self.user == other.user
            and self.zone == other.zone
            and self.perm == other.perm
        )

    def __lt__(self, other):
        if self.zone is not None and other.zone is None:
            return True

        if self.zone is None and other.zone is not None:
            return True

        if self.zone is not None and other.zone is not None:
            if self.zone < other.zone:
                return True

        if self.zone == other.zone:
            if self.user < other.user:
                return True

            if self.user == other.user:
                return self.perm.name < other.perm.name

        return False

    def __repr__(self):
        zone = AC.SEPARATOR + self.zone if self.zone else ""
        perm = self.perm.name.lower()

        return f"{self.user}{zone}:{perm}"


@total_ordering
class AVU(object):
    """AVU is an iRODS attribute, value, units tuple.

    AVUs may be sorted, where they will sorted lexically, first by
    namespace (if present), then by attribute, then by value and finally by
    units (if present).
    """

    SEPARATOR = ":"
    """The attribute namespace separator"""

    HISTORY_SUFFIX = "_history"
    """The attribute history suffix"""

    def __init__(self, attribute: str, value: Any, units=None, namespace=None):
        if namespace:
            if namespace.find(AVU.SEPARATOR) >= 0:
                raise ValueError(
                    f"AVU namespace '{namespace}' contained '{AVU.SEPARATOR}'"
                )
        if attribute is None:
            raise ValueError("AVU attribute may not be None")
        if value is None:
            raise ValueError("AVU value may not be None")

        self._namespace = namespace
        self._attribute = str(attribute)
        self._value = str(value)
        self._units = units

    @classmethod
    def collate(cls, *avus) -> Dict[str : List[AVU]]:
        """Collates AVUs by attribute (including namespace, if any) and
        returns a dict mapping the attribute to a list of AVUs with that
        attribute.

        Returns: Dict[str: List[AVU]]
        """
        collated = defaultdict(lambda: list())

        for avu in avus:
            collated[avu.attribute].append(avu)

        return collated

    @classmethod
    def history(cls, *avus, history_date=None) -> AVU:
        """Returns a history AVU describing the argument AVUs. A history AVU is
        sometimes added to an iRODS path to describe AVUs that were once
        present, but have been removed. Adding a history AVU can act as a poor
        man's audit trail and it used because iRODS does not have native
        history support.

        Args:
            avus: AVUs removed, which must share the same attribute
            and namespace (if any).
            history_date: A datetime to be embedded as part of the history
            AVU value.

        Returns: AVU
        """
        if history_date is None:
            history_date = datetime.utcnow()
        date = history_date.isoformat(timespec="seconds")

        # Check that the AVUs have the same namespace and attribute and that
        # none are history attributes (we don't do meta-history!)
        namespaces = set()
        attributes = set()
        values = set()
        for avu in avus:
            if avu.is_history():
                raise ValueError(f"Cannot create a history of a history AVU: {avu}")
            namespaces.add(avu.namespace)
            attributes.add(avu.without_namespace)
            values.add(avu.value)

        if len(namespaces) > 1:
            raise ValueError(
                "Cannot create a history for AVUs with a mixture of "
                f"namespaces: {namespaces}"
            )
        if len(attributes) > 1:
            raise ValueError(
                "Cannot create a history for AVUs with a mixture of "
                f"attributes: {attributes}"
            )

        history_namespace = namespaces.pop()
        history_attribute = attributes.pop() + AVU.HISTORY_SUFFIX
        history_value = "[{}] {}".format(date, ",".join(sorted(values)))

        return AVU(history_attribute, history_value, namespace=history_namespace)

    @property
    def namespace(self):
        """The attribute namespace. If the attribute has no namespace, the empty
        string."""
        return self._namespace

    @property
    def without_namespace(self):
        """The attribute without namespace."""
        return self._attribute

    @property
    def attribute(self):
        """The attribute, including namespace, if any. The namespace an attribute are
        separated by AVU.SEPARATOR."""
        if self._namespace:
            return f"{self._namespace}{AVU.SEPARATOR}{self._attribute}"
        else:
            return self.without_namespace

    @property
    def value(self):
        """The value associates with the attribute."""
        return self._value

    @property
    def units(self):
        """The units associated with the attribute. Units may be None."""
        return self._units

    def with_namespace(self, namespace: str):
        """make a new copy of this AVU with the specified namespace.

        Args:
            namespace: The new namespace.

        Returns: str
        """
        return AVU(self._attribute, self._value, self._units, namespace=namespace)

    def is_history(self) -> bool:
        """Return true if this is a history AVU."""
        return self._attribute.endswith(AVU.HISTORY_SUFFIX)

    def __hash__(self):
        return hash(self.attribute) + hash(self.value) + hash(self.units)

    def __eq__(self, other):
        if not isinstance(other, AVU):
            return False

        return (
            self.attribute == other.attribute
            and self.value == other.value
            and self.units == other.units
        )

    def __lt__(self, other):
        if self.namespace is not None and other.namespace is None:
            return True
        if self.namespace is None and other.namespace is not None:
            return False

        if self.namespace is not None and other.namespace is not None:
            if self.namespace < other.namespace:
                return True

        if self.namespace == other.namespace:
            if self.attribute < other.attribute:
                return True

            if self.attribute == other.attribute:
                if self.value < other.value:
                    return True

                if self.value == other.value:
                    if self.units is not None and other.units is None:
                        return True
                    if self.units is None and other.units is not None:
                        return False
                    if self.units is None and other.units is None:
                        return False

                    return self.units < other.units

        return False

    def __repr__(self):
        units = " " + self.units if self._units else ""
        return f"{self.attribute}={self.value}{units}"

    def __str__(self):
        units = " " + self.units if self._units else ""
        return f"<AVU '{self.attribute}' = '{self.value}'{units}>"


class RodsItem(PathLike):
    """A base class for iRODS path entities."""

    def __init__(self, client: Baton, path: Union[PurePath, str]):
        self.client = client
        self.path = PurePath(path)

    def exists(self) -> bool:
        """Return true if the item exists in iRODS."""
        try:
            self._list()
        except RodsError as re:
            if re.code == -310000:  # iRODS error code for path not found
                return False
        return True

    def meta_add(self, *avus: Union[AVU, Tuple[AVU]]) -> int:
        """Add AVUs to the item's metadata, if they are not already present.
        Return the number of AVUs added.

        Args:
            *avus: AVUs to add.

        Returns: int
        """
        current = self.metadata()
        to_add = sorted(set(avus).difference(current))

        if to_add:
            log.debug("Adding AVUs", path=self.path, avus=to_add)
            item = self._to_dict()
            item[Baton.AVUS] = to_add
            self.client.meta_add(item)

        return len(to_add)

    def meta_remove(self, *avus: Union[AVU, Tuple[AVU]]) -> int:
        """Remove AVUs from the item's metadata, if they are present.
        Return the number of AVUs removed.

        Args:
            *avus: AVUs to remove.

        Returns: int
        """
        current = self.metadata()
        to_remove = sorted(set(current).intersection(avus))

        if to_remove:
            log.debug("Removing AVUs", path=self.path, avus=to_remove)
            item = self._to_dict()
            item[Baton.AVUS] = to_remove
            self.client.meta_rem(item)

        return len(to_remove)

    def meta_supersede(
        self, *avus: Union[AVU, Tuple[AVU]], history=False, history_date=None
    ) -> Tuple[int, int]:
        """Remove AVUs from the item's metadata that share an attribute with
         any of the argument AVUs and add the argument AVUs to the item's
         metadata. Return the numbers of AVUs added and removed, including any
         history AVUs created.

         Args:
             avus: AVUs to add in place of existing AVUs sharing those
             attributes.
             history: Create history AVUs describing any AVUs removed when
             superseding. See AVU.history.
             history_date: A datetime to be embedded as part of the history
             AVU values.

        Returns: Tuple[int, int]
        """
        if history_date is None:
            history_date = datetime.utcnow()

        current = self.metadata()
        log.debug("Superseding AVUs", path=self.path, old=current, new=avus)

        rem_attrs = set(map(lambda avu: avu.attribute, avus))
        to_remove = set(filter(lambda a: a.attribute in rem_attrs, current))

        # If the argument AVUs have some of the AVUs to remove amongst them,
        # we don't want to remove them from the item, just to add them back.
        to_remove.difference_update(avus)
        to_remove = sorted(to_remove)
        if to_remove:
            log.debug("Removing AVUs", path=self.path, avus=to_remove)
            item = self._to_dict()
            item[Baton.AVUS] = to_remove
            self.client.meta_rem(item)

        to_add = sorted(set(avus).difference(current))
        if history:
            hist = []
            for avus in AVU.collate(*to_remove).values():
                hist.append(AVU.history(*avus, history_date=history_date))
            to_add += hist

        if to_add:
            log.debug("Adding AVUs", path=self.path, avus=to_add)
            item = self._to_dict()
            item[Baton.AVUS] = to_add
            self.client.meta_add(item)

        return len(to_remove), len(to_add)

    def ac_add(self, *acs: Union[AC, Tuple[AC]], recurse=False) -> int:
        """Add access controls to the item. Return the number of access
        controls added. If some of the argument access controls are already
        present, those arguments will be ignored.

        Args:
            acs: Access controls.
            recurse: Recursively add access control.

        Returns: int
        """
        current = self.acl()
        to_add = sorted(set(acs).difference(current))
        if to_add:
            log.debug("Adding to ACL", path=self.path, ac=to_add)
            item = self._to_dict()
            item[Baton.ACCESS] = to_add
            self.client.ac_set(item, recurse=recurse)

        return len(to_add)

    def ac_rem(self, *acs: Union[AC, Tuple[AC]], recurse=False) -> int:
        """Remove access controls from the item. Return the number of access
        controls removed. If some of the argument access controls are not
        present, those arguments will be ignored.

        Args:
            acs: Access controls.
            recurse: Recursively add access control.

        Returns: int
        """
        current = self.acl()
        to_remove = sorted(set(current).intersection(acs))
        if to_remove:
            log.debug("Removing from ACL", path=self.path, ac=to_remove)

            # In iRODS we "remove" permissions by setting them to NULL
            for ac in to_remove:
                ac.perm = Permission.NULL

            item = self._to_dict()
            item[Baton.ACCESS] = to_remove
            self.client.ac_set(item, recurse=recurse)

        return len(to_remove)

    def ac_supersede(
        self, *acs: Union[AC, Tuple[AC]], recurse=False
    ) -> Tuple[int, int]:
        """Remove all access controls from the item, replacing them with the
        specified access controls. Return the numbers of access controls
        removed and added.

        Args:
            acs: Access controls.
            recurse: Recursively supersede access controls.

        Returns: Tuple[int, int]
        """
        current = self.acl()
        log.debug("Superseding ACL", path=self.path, old=current, new=acs)

        to_remove = sorted(set(current).difference(acs))
        if to_remove:
            log.debug("Removing from ACL", path=self.path, ac=to_remove)

            # In iRODS we "remove" permissions by setting them to NULL
            for ac in to_remove:
                ac.perm = Permission.NULL

            item = self._to_dict()
            item[Baton.ACCESS] = to_remove
            self.client.ac_set(item, recurse=recurse)

        to_add = sorted(set(acs).difference(current))
        if to_add:
            log.debug("Adding to ACL", path=self.path, ac=to_add)
            item = self._to_dict()
            item[Baton.ACCESS] = to_add
            self.client.ac_set(item, recurse=recurse)

        return len(to_remove), len(to_add)

    def metadata(self) -> List[AVU]:
        """Return the item's metadata.

        Returns: List[AVU]
        """
        item = self._list(avu=True).pop()
        if Baton.AVUS not in item.keys():
            raise BatonError(f"{Baton.AVUS} key missing from {item}")

        return sorted(item[Baton.AVUS])

    def acl(self) -> List[AC]:
        """Return the item's Access Control List (ACL).

        Returns: List[AC]"""
        item = self._list(acl=True).pop()
        if Baton.ACCESS not in item.keys():
            raise BatonError(f"{Baton.ACCESS} key missing from {item}")

        return sorted(item[Baton.ACCESS])

    @abstractmethod
    def get(self, local_path: Union[Path, str], **kwargs):
        """Get the item from iRODS."""
        pass

    @abstractmethod
    def put(self, local_path: Union[Path, str], **kwargs):
        pass

    @abstractmethod
    def _to_dict(self):
        pass

    @abstractmethod
    def _list(self, **kwargs):
        pass


class DataObject(RodsItem):
    """An iRODS data object.

    DataObject is a PathLike for the iRODS path it represents.
    """

    def __init__(self, client: Baton, remote_path: Union[PurePath, str]):
        super().__init__(client, PurePath(remote_path).parent)
        self.name = PurePath(remote_path).name

    def list(self) -> DataObject:
        """Return a new DataObject representing this one.

        Returns: DataObject
        """
        item = self._list().pop()
        if Baton.OBJ not in item.keys():
            raise BatonError(f"{Baton.OBJ} key missing from {item}")

        return make_rods_item(self.client, item)

    def checksum(
        self,
        calculate_checksum=False,
        recalculate_checksum=False,
        verify_checksum=False,
    ) -> str:
        """Get the checksum of the data object. If no checksum has been calculated on
        the remote side, return None.

        Keyword Args:
            calculate_checksum: Calculate remote checksums for all replicates. If
            checksums exist, this is s no-op.
            recalculate_checksum: Force recalculation of remote checksums for all
            replicates.
            verify_checksum: Verify the local checksum against the remote checksum.
            Verification implies checksum calculation.

        Returns: str
        """

        item = self._to_dict()
        return self.client.checksum(
            item,
            calculate_checksum=calculate_checksum,
            recalculate_checksum=recalculate_checksum,
            verify_checksum=verify_checksum,
        )

    def get(self, local_path: Union[Path, str], verify_checksum=True) -> int:
        """Get the data object from iRODS"""
        item = self._to_dict()
        return self.client.get(item, Path(local_path), verify_checksum=verify_checksum)

    def put(
        self,
        local_path: Union[Path, str],
        calculate_checksum=False,
        verify_checksum=True,
        force=True,
    ):
        """Put the data object into iRODS.

        Args:
            local_path: The local path of a file to put into iRODS at the path
            specified by this data object.
            calculate_checksum: Calculate remote checksums for all replicates. If
            checksums exist, this is s no-op.
            verify_checksum: Verify the local checksum against the remote checksum.
            Verification implies checksum calculation.
            force: Overwrite any data object already present in iRODS.
        """
        item = self._to_dict()
        self.client.put(
            item,
            Path(local_path),
            calculate_checksum=calculate_checksum,
            verify_checksum=verify_checksum,
            force=force,
        )

    def read(self) -> str:
        """Get the data object from iRODS."""
        item = self._to_dict()
        return self.client.read(item)

    def _list(self, **kwargs) -> List[dict]:
        item = self._to_dict()
        return self.client.list(item, **kwargs)

    def _to_dict(self) -> Dict:
        return {Baton.COLL: self.path, Baton.OBJ: self.name}

    def __eq__(self, other):
        if not isinstance(other, DataObject):
            return False

        return self.path == other.path and self.name == other.name

    def __fspath__(self):
        return self.__repr__()

    def __repr__(self):
        return PurePath(self.path, self.name).as_posix()


class Collection(RodsItem):
    """An iRODS collection.

    Collection is a PathLike for the iRODS path it represents.
    """

    def __init__(self, client: Baton, path: Union[PurePath, str]):
        super().__init__(client, path)

    def contents(
        self, acl=False, avu=False, recurse=False
    ) -> List[Union[DataObject, Collection]]:
        """Return list of the Collection contents.

        Keyword Args:
          acl: Include ACL information.
          avu: Include AVU (metadata) information.
          recurse: Recurse into sub-collections.

        Returns: List[Union[DataObject, Collection]]
        """
        items = self._list(acl=acl, avu=avu, contents=True, recurse=recurse)

        return [make_rods_item(self.client, item) for item in items]

    def list(self, acl=False, avu=False) -> Collection:
        """Return a new Collection representing this one.

        Keyword Args:
          acl: Include ACL information.
          avu: Include AVU (metadata) information.

        Returns: Collection
        """
        items = self._list(acl=acl, avu=avu)
        # Gets a single item
        return make_rods_item(self.client, items.pop())

    def get(self, local_path: Union[Path, str], **kwargs):
        raise NotImplementedError()

    def put(self, local_path: Union[Path, str], recurse=True):
        raise NotImplementedError()

    def _list(self, **kwargs) -> List[dict]:
        return self.client.list({Baton.COLL: self.path}, **kwargs)

    def _to_dict(self):
        return {Baton.COLL: self.path}

    def __eq__(self, other):
        if not isinstance(other, Collection):
            return False

        return self.path == other.path

    def __fspath__(self):
        return self.__repr__()

    def __repr__(self):
        return self.path.as_posix()


def make_rods_item(client: Baton, item: Dict) -> Union[DataObject, Collection]:
    """Create a new Collection or DataObject as appropriate for a dictionary
    returned by a Baton.

    Returns: Union[DataObject, Collection]
    """
    if Baton.COLL not in item.keys():
        raise BatonError(f"{Baton.COLL} key missing from {item}")

    if Baton.OBJ in item.keys():
        return DataObject(client, PurePath(item[Baton.COLL], item[Baton.OBJ]))
    return Collection(client, PurePath(item[Baton.COLL]))


class Baton(AbstractContextManager):
    """A wrapper around the baton-do client program, used for interacting with
    iRODS."""

    CLIENT = "baton-do"

    AVUS = "avus"
    ATTRIBUTE = "attribute"
    VALUE = "value"
    UNITS = "units"

    COLL = "collection"
    OBJ = "data_object"
    ZONE = "zone"

    ACCESS = "access"
    OWNER = "owner"
    LEVEL = "level"

    CHMOD = "chmod"
    LIST = "list"
    GET = "get"
    PUT = "put"
    CHECKSUM = "checksum"
    METAQUERY = "metaquery"

    METAMOD = "metamod"
    ADD = "add"
    REM = "rem"

    OP = "operation"
    ARGS = "arguments"
    TARGET = "target"

    RESULT = "result"
    SINGLE = "single"
    MULTIPLE = "multiple"
    CONTENTS = "contents"
    DATA = "data"

    ERR = "error"
    MSG = "message"
    CODE = "code"

    def __init__(self):
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def is_running(self) -> bool:
        """Returns true if the client is running."""
        return self.proc and self.proc.poll() is None

    def start(self):
        """Starts the client if it is not already running."""
        if self.is_running():
            log.warning(
                "Tried to start a Baton instance that is already running",
                pid=self.proc.pid,
            )
            return

        self.proc = subprocess.Popen(
            [Baton.CLIENT, "--unbuffered"],
            bufsize=0,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log.debug(f"Started {Baton.CLIENT} process", pid=self.proc.pid)

    def stop(self):
        """Stops the client if it is running."""
        if not self.is_running():
            log.warning("Tried to start a Baton instance that is not running")
            return

        self.proc.stdin.close()
        try:
            log.debug(f"Terminating {Baton.CLIENT} process", pid=self.proc.pid)
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.error(
                f"Failed to terminate {Baton.CLIENT} process; killing",
                pid=self.proc.pid,
            )
            self.proc.kill()
        self.proc = None

    def list(
        self,
        item: Dict,
        acl=False,
        avu=False,
        contents=False,
        recurse=False,
        size=False,
        timestamp=False,
    ) -> List[Dict]:
        if recurse:
            raise NotImplementedError("recurse")

        result = self._execute(
            Baton.LIST,
            {
                "acl": acl,
                "avu": avu,
                "contents": contents,
                "size": size,
                "timestamp": timestamp,
            },
            item,
        )
        if contents:
            result = result[Baton.CONTENTS]
        else:
            result = [result]

        return result

    def checksum(
        self,
        item,
        calculate_checksum=False,
        recalculate_checksum=False,
        verify_checksum=False,
    ) -> str:
        result = self._execute(
            Baton.CHECKSUM,
            {
                "calculate": calculate_checksum,
                "recalculate": recalculate_checksum,
                "verify": verify_checksum,
            },
            item,
        )
        checksum = result[Baton.CHECKSUM]
        return checksum

    def meta_add(self, item: Dict):
        self._execute(Baton.METAMOD, {Baton.OP: Baton.ADD}, item)

    def meta_rem(self, item: Dict):
        self._execute(Baton.METAMOD, {Baton.OP: Baton.REM}, item)

    def meta_query(
        self, avus: List[AVU], zone=None, collection=False, data_object=False
    ) -> List[Union[DataObject, Collection]]:
        args = {}
        if collection:
            args["collection"] = True
        if data_object:
            args["object"] = True

        item = {Baton.AVUS: avus}
        if zone:
            item[Baton.COLL] = self._zone_hint_to_path(zone)

        result = self._execute(Baton.METAQUERY, args, item)
        items = [make_rods_item(self, item) for item in result]
        items.sort()

        return items

    def ac_set(self, item: Dict, recurse=False):
        self._execute(Baton.CHMOD, {"recurse": recurse}, item)

    def get(
        self, item: Dict, local_path: Path, verify_checksum=True, force=True
    ) -> int:
        # TODO: Note that baton does not use rcDataObjGet to get data. It streams the
        #  file while calculating the MD5 and overwrites existing files without
        #  warning. Therefore it is similar to using verify_checksum=True,
        #  force=True. Maybe add an rcDataObjGet mode to benefit from parallel get?

        # Let's be sure users are aware if they try to change these at the moment.
        if not verify_checksum:
            raise BatonError(
                f"{Baton.CLIENT} does not support get without checksum verification"
            )
        if not force:
            raise BatonError(
                f"{Baton.CLIENT} does not support get without forced overwriting"
            )

        item["directory"] = local_path.parent
        item["file"] = local_path.name

        self._execute(
            Baton.GET, {"save": True, "verify": verify_checksum, "force": force}, item
        )
        return local_path.stat().st_size

    def read(self, item: Dict) -> str:
        result = self._execute(Baton.GET, {}, item)
        if Baton.DATA not in result:
            raise InvalidJSONError(
                f"Invalid result '{result}': data property was missing"
            )
        return result[Baton.DATA]

    def put(
        self,
        item: Dict,
        local_path: Path,
        calculate_checksum=True,
        verify_checksum=True,
        force=True,
    ):
        item["directory"] = local_path.parent
        item["file"] = local_path.name

        self._execute(
            Baton.PUT,
            {
                "checksum": calculate_checksum,
                "verify": verify_checksum,
                "force": force,
            },
            item,
        )

    def _execute(self, operation: str, args: Dict, item: Dict) -> Dict:
        if not self.is_running():
            log.debug(f"{Baton.CLIENT} is not running ... starting")
            self.start()
            if not self.is_running():
                raise BatonError(f"{Baton.CLIENT} failed to start")

        response = self._send(self._wrap(operation, args, item))
        return self._unwrap(response)

    @staticmethod
    def _wrap(operation: str, args: Dict, item: Dict) -> Dict:
        return {
            Baton.OP: operation,
            Baton.ARGS: args,
            Baton.TARGET: item,
        }

    @staticmethod
    def _unwrap(envelope: Dict) -> Dict:
        # If there is an error report from the iRODS server in the envelope, we need to
        # raise a RodsError
        if Baton.ERR in envelope:
            err = envelope[Baton.ERR]
            if Baton.CODE not in err:
                raise InvalidEnvelopeError("Error code was missing", envelope=envelope)
            if Baton.MSG not in err:
                raise InvalidEnvelopeError(
                    "Error message was missing", envelope=envelope
                )

            raise RodsError(err[Baton.MSG], err[Baton.CODE])

        if Baton.RESULT not in envelope:
            raise InvalidEnvelopeError(
                "Operation result property was missing", envelope=envelope
            )

        if Baton.SINGLE in envelope[Baton.RESULT]:
            return envelope[Baton.RESULT][Baton.SINGLE]

        if Baton.MULTIPLE in envelope[Baton.RESULT]:
            return envelope[Baton.RESULT][Baton.MULTIPLE]

        raise InvalidEnvelopeError(
            "Operation result value was empty", envelope=envelope
        )

    def _send(self, envelope: Dict) -> Dict:
        encoded = json.dumps(envelope, cls=BatonJSONEncoder)
        log.debug("Sending", msg=encoded)

        msg = bytes(encoded, "utf-8")
        self.proc.stdin.write(msg)
        self.proc.stdin.flush()

        resp = self.proc.stdout.readline()

        log.debug("Received", msg=resp)

        return json.loads(resp, object_hook=as_baton)

    @staticmethod
    def _zone_hint_to_path(zone) -> str:
        z = str(zone)
        if z.startswith("/"):
            return z

        return "/" + z


class BatonJSONEncoder(json.JSONEncoder):
    """Encoder for baton JSON."""

    def default(self, o: Any) -> Any:
        if isinstance(o, AVU):
            enc = {Baton.ATTRIBUTE: o.attribute, Baton.VALUE: o.value}
            if o.units:
                enc[Baton.UNITS] = o.units
            return enc

        if isinstance(o, Permission):
            return o.name.lower()

        if isinstance(o, AC):
            return {
                Baton.OWNER: o.user,
                Baton.ZONE: o.zone,
                Baton.LEVEL: o.perm,
            }

        if isinstance(o, PurePath):
            return o.as_posix()


def as_baton(d: Dict) -> Any:
    """Object hook for decoding baton JSON."""

    # Match an AVU sub-document
    if Baton.ATTRIBUTE in d:
        attr = str(d[Baton.ATTRIBUTE])
        value = d[Baton.VALUE]
        units = d.get(Baton.UNITS, None)

        if attr.find(AVU.SEPARATOR) >= 0:  # Has namespace
            (ns, _, bare_attr) = attr.partition(AVU.SEPARATOR)

            # This accepts an attribute with a namespace that is the empty
            # string i.e. ":foo" or is whitespace i.e. " :foo" and discards
            # the namespace.
            if not ns.strip():
                ns = None

            return AVU(bare_attr, value, units, namespace=ns)

        return AVU(attr, value, units)

    # Match an access permission sub-document
    if Baton.OWNER in d and Baton.LEVEL in d:
        user = d[Baton.OWNER]
        zone = d[Baton.ZONE]
        level = d[Baton.LEVEL]

        return AC(user, Permission[level.upper()], zone=zone)

    return d
