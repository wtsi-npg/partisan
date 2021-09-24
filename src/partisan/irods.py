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

import atexit
import json
import subprocess
import threading
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from enum import Enum, unique
from functools import total_ordering
from os import PathLike
from pathlib import Path, PurePath
from queue import LifoQueue, Queue
from threading import Thread
from typing import Annotated, Any, Dict, List, Tuple, Union

from structlog import get_logger

from partisan.exception import (
    BatonError,
    BatonTimeoutError,
    InvalidEnvelopeError,
    InvalidJSONError,
    RodsError,
)

log = get_logger(__package__)

"""This module provides a basic API for accessing iRODS using the native
iRODS client 'baton' (https://github.com/wtsi-npg/baton).
"""


class Baton:
    """A wrapper around the baton-do client program, used for interacting with
    iRODS.
    """

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
        self._proc = None
        self._pid = None

    def __str__(self):
        return f"<Baton {Baton.CLIENT}, running: {self.is_running()}, PID: {self._pid}>"

    def is_running(self) -> bool:
        """Returns true if the client is running."""
        return self._proc and self._proc.poll() is None

    def pid(self):
        """Returns the PID of the baton-do client process."""
        return self._pid

    def start(self):
        """Starts the client if it is not already running."""
        if self.is_running():
            log.warning(
                "Tried to start a Baton instance that is already running",
                pid=self._proc.pid,
            )
            return

        self._proc = subprocess.Popen(
            [Baton.CLIENT, "--unbuffered", "--no-error", "--silent"],
            bufsize=0,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._pid = self._proc.pid
        log.debug(f"Started {Baton.CLIENT} process", pid=self._pid)

        def stderr_reader(err):
            for line in iter(err.readline, b""):
                log.error(
                    f"{Baton.CLIENT} STDERR",
                    pid=self._pid,
                    msg=line.decode("utf-8").rstrip(),
                )

        t = Thread(target=stderr_reader, args=(self._proc.stderr,))
        t.daemon = True
        t.start()

    def stop(self):
        """Stops the client if it is running."""
        if not self.is_running():
            return

        self._proc.stdin.close()
        try:
            log.debug(f"Terminating {Baton.CLIENT} process", pid=self._pid)
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.error(
                f"Failed to terminate {Baton.CLIENT} process; killing",
                pid=self._pid,
            )
            self._proc.kill()
        self._proc = None

    def list(
        self,
        item: Dict,
        acl=False,
        avu=False,
        contents=False,
        recurse=False,
        size=False,
        timestamp=False,
        timeout=None,
        tries=1,
    ) -> List[Dict]:
        """Lists i.e. reports on items in iRODS.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
            this must be suitable input for baton-do.
            acl: Include ACL information in the result
            avu: Include AVU information in the result
            contents: Include contents in the result (for a collection item)
            recurse: Recurse into collections (for a collection item)
            size: Include size information in the result (for a data object)
            timestamp: Include timestamp information in the result (for a data object)
            timeout: Operation timeout
            tries: Number of times to try the operation
        """
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
            timeout=timeout,
            tries=tries,
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
        timeout=None,
        tries=1,
    ) -> str:
        result = self._execute(
            Baton.CHECKSUM,
            {
                "calculate": calculate_checksum,
                "recalculate": recalculate_checksum,
                "verify": verify_checksum,
            },
            item,
            timeout=timeout,
            tries=tries,
        )
        checksum = result[Baton.CHECKSUM]
        return checksum

    def meta_add(self, item: Dict, timeout=None, tries=1):
        self._execute(
            Baton.METAMOD, {Baton.OP: Baton.ADD}, item, timeout=timeout, tries=tries
        )

    def meta_rem(self, item: Dict, timeout=None, tries=1):
        self._execute(
            Baton.METAMOD, {Baton.OP: Baton.REM}, item, timeout=timeout, tries=tries
        )

    def meta_query(
        self,
        avus: List[AVU],
        zone=None,
        collection=False,
        data_object=False,
        timeout=None,
        tries=1,
    ) -> Dict:
        args = {}
        if collection:
            args["collection"] = True
        if data_object:
            args["object"] = True

        item = {Baton.AVUS: avus}
        if zone:
            item[Baton.COLL] = self._zone_hint_to_path(zone)

        return self._execute(Baton.METAQUERY, args, item, timeout=timeout, tries=tries)

    def ac_set(self, item: Dict, recurse=False, timeout=None, tries=1):
        self._execute(
            Baton.CHMOD, {"recurse": recurse}, item, timeout=timeout, tries=tries
        )

    def get(
        self,
        item: Dict,
        local_path: Path,
        verify_checksum=True,
        force=True,
        timeout=None,
        tries=1,
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
            Baton.GET,
            {"save": True, "verify": verify_checksum, "force": force},
            item,
            timeout=timeout,
            tries=tries,
        )
        return local_path.stat().st_size

    def read(self, item: Dict, timeout=None, tries=1) -> str:
        result = self._execute(Baton.GET, {}, item, timeout=timeout, tries=tries)
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
        timeout=None,
        tries=1,
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
            timeout=timeout,
            tries=tries,
        )

    def _execute(
        self, operation: str, args: Dict, item: Dict, timeout=None, tries=1
    ) -> Dict:
        if not self.is_running():
            log.debug(f"{Baton.CLIENT} is not running ... starting")
            self.start()
            if not self.is_running():
                raise BatonError(f"{Baton.CLIENT} failed to start")

        wrapped = self._wrap(operation, args, item)

        # The most common failure mode we encounter with clients that use the iRODS C
        # API is where the server stops responding to API calls on the current
        # connection. In order to timeout these bad operations, round-trips to the
        # server are run in their own thread which provides API for managing the
        # timeout behaviour.
        #
        # Not all long-duration API calls are bad, to timeouts must be set by
        # operation type. A "put" operation of a multi-GiB file may legitimately take
        # hours, a metadata change may not.
        lifo = LifoQueue(maxsize=1)

        t = Thread(target=lambda q, w: q.put(self._send(w)), args=(lifo, wrapped))
        t.start()

        for i in range(tries):
            t.join(timeout=timeout)
            if not t.is_alive():
                break
            log.warning(f"Timed out sending", client=self, tryno=i, doc=wrapped)

            # Still alive after all the tries?
        if t.is_alive():
            self.stop()
            raise BatonTimeoutError(
                "Exhausted all timeouts, stopping client", client=self, tryno=tries
            )

        # By setting a timeout here (0.1 second is arbitrary) we will raise an Empty
        # exception. This shouldn't happen because timeouts are dealt with above.
        response = lifo.get(timeout=0.1)
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

        # We are not using Popen.communicate here because that terminates the process
        self._proc.stdin.write(msg)
        self._proc.stdin.flush()
        resp = self._proc.stdout.readline()
        log.debug("Received", msg=resp)

        if self._proc.returncode is not None:
            raise BatonError(
                f"{Baton.CLIENT} PID: {self._proc.pid} terminated unexpectedly "
                f"with return code {self._proc.returncode}"
            )

        return json.loads(resp, object_hook=as_baton)

    @staticmethod
    def _zone_hint_to_path(zone) -> str:
        z = str(zone)
        if z.startswith("/"):
            return z

        return "/" + z


class BatonPool:
    """A pool of Baton clients."""

    def __init__(self, maxsize=4):
        self._queue = Queue(maxsize=maxsize)  # Queue is threadsafe anyway
        self._mutex = threading.RLock()  # For managing open/close state

        with self._mutex:
            self._open = False

            for _ in range(maxsize):
                c = Baton()
                log.debug(f"Adding a new client to the pool: {c}")
                self._queue.put(c)
            self._open = True

    def __repr__(self):
        return (
            f"<BatonPool maxsize: {self._queue.maxsize}, "
            f"qsize: {self._queue.qsize()} open: {self.is_open()}>"
        )

    def is_open(self):
        """Return True if the pool is open to get clients."""
        with self._mutex:
            return self._open

    def close(self):
        """Close the pool and stop all the clients."""
        with self._mutex:
            log.debug("Closing the client pool")
            self._open = False
            while not self._queue.empty():
                c: Baton = self._queue.get_nowait()
                c.stop()

    def get(self, timeout=None) -> Baton:
        """Get a client from the pool. if a timeout is supplied, waiting up to the
        timeout.

        Keyword Args:
            timeout: Timeout to get a client, in seconds. Raises queue.Empty if the
            operation times out.

        Returns: Baton
        """
        if not self.is_open():
            raise BatonError("Attempted to get a client from a closed pool")

        c: Baton = self._queue.get(timeout=timeout)
        log.debug(f"Getting a client from the pool: {c}")

        if not c.is_running():
            c.start()
        return c

    def put(self, c: Baton, timeout=None):
        """Put a client back into the pool. if a timeout is supplied, waiting up to the
        timeout.

        Keyword Args:
        timeout: Timeout to put a client, in seconds. Raises queue.Full if the
        operation times out.
        """
        log.debug(f"Returning a client to the pool: {c}")
        self._queue.put(c, timeout=timeout)


@contextmanager
def client_pool(maxsize=4) -> BatonPool:
    """Yields a pool of clients that will be closed automatically when the pool goes
    out of scope.

    Keyword Args:
        maxsize: The maximum number of active clients in the pool.

    Yields: BatonPool
    """
    pool = BatonPool(maxsize=maxsize)
    try:
        yield pool
    finally:
        pool.close()


@contextmanager
def client(pool: BatonPool, timeout=None) -> Baton:
    """Yields a client from a pool, returning it to the pool automatically when the
    client goes out of scope.

    Args:
        pool: The pool from which to get the client.
    Keyword Args:
        timeout: Timeout for both getting the client and putting it back, in seconds.
        Raises queue.Empty or queue.Full if the get or put operations
        respectively, time out.

    Returns: Baton
    """
    c = pool.get(timeout=timeout)
    try:
        yield c
    finally:
        pool.put(c, timeout=timeout)


def _default_pool_init() -> BatonPool:
    pool = BatonPool(maxsize=4)
    atexit.register(pool.close)
    return pool


default_pool: Annotated[BatonPool, "The default client pool"] = _default_pool_init()


def meta_query(
    avus: List[AVU],
    zone=None,
    collection=False,
    data_object=False,
    pool=default_pool,
    timeout=None,
    tries=1,
) -> List[Union[DataObject, Collection]]:
    with client(pool) as c:
        result = c.meta_query(
            avus,
            zone=zone,
            collection=collection,
            data_object=data_object,
            timeout=timeout,
            tries=tries,
        )
        items = [_make_rods_item(item, pool=pool) for item in result]
        items.sort()

        return items


@unique
class Permission(Enum):
    """The kinds of data access permission available to iRODS users."""

    NULL = "null"
    OWN = "own"
    READ = "read"
    WRITE = "write"


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

        Args:
            avus: AVUs to collate.

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
        Keyword Args:
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

    def __init__(self, path: Union[PurePath, str], pool=default_pool):
        self.path = PurePath(path)
        self.pool = pool

    def exists(self, timeout=None, tries=1) -> bool:
        """Return true if the item exists in iRODS."""
        try:
            self._list(timeout=timeout, tries=tries)
        except RodsError as re:
            if re.code == -310000:  # iRODS error code for path not found
                return False
        return True

    def meta_add(self, *avus: Union[AVU, Tuple[AVU]], timeout=None, tries=1) -> int:
        """Add AVUs to the item's metadata, if they are not already present.
        Return the number of AVUs added.

        Args:
            *avus: AVUs to add.
        Keyword Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        current = self.metadata()
        to_add = sorted(set(avus).difference(current))

        if to_add:
            log.debug("Adding AVUs", path=self.path, avus=to_add)
            item = self._to_dict()
            item[Baton.AVUS] = to_add
            with client(self.pool) as c:
                c.meta_add(item, timeout=timeout, tries=tries)

        return len(to_add)

    def meta_remove(self, *avus: Union[AVU, Tuple[AVU]], timeout=None, tries=1) -> int:
        """Remove AVUs from the item's metadata, if they are present.
        Return the number of AVUs removed.

        Args:
            *avus: AVUs to remove.
        Keyword Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        current = self.metadata()
        to_remove = sorted(set(current).intersection(avus))

        if to_remove:
            log.debug("Removing AVUs", path=self.path, avus=to_remove)
            item = self._to_dict()
            item[Baton.AVUS] = to_remove
            with client(self.pool) as c:
                c.meta_rem(item, timeout=timeout, tries=tries)

        return len(to_remove)

    def meta_supersede(
        self,
        *avus: Union[AVU, Tuple[AVU]],
        history=False,
        history_date=None,
        timeout=None,
        tries=1,
    ) -> Tuple[int, int]:
        """Remove AVUs from the item's metadata that share an attribute with
         any of the argument AVUs and add the argument AVUs to the item's
         metadata. Return the numbers of AVUs added and removed, including any
         history AVUs created.

         Args:
             avus: AVUs to add in place of existing AVUs sharing those
             attributes.
         Keyword Args:
             history: Create history AVUs describing any AVUs removed when
             superseding. See AVU.history.
             history_date: A datetime to be embedded as part of the history
             AVU values.
             timeout: Operation timeout in seconds.
             tries: Number of times to try the operation.

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
            with client(self.pool) as c:
                c.meta_rem(item, timeout=timeout, tries=tries)

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
            with client(self.pool) as c:
                c.meta_add(item, timeout=timeout, tries=tries)

        return len(to_remove), len(to_add)

    def ac_add(
        self, *acs: Union[AC, Tuple[AC]], recurse=False, timeout=None, tries=1
    ) -> int:
        """Add access controls to the item. Return the number of access
        controls added. If some of the argument access controls are already
        present, those arguments will be ignored.

        Args:
            acs: Access controls.
        Keyword Args:
            recurse: Recursively add access control.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        current = self.acl()
        to_add = sorted(set(acs).difference(current))
        if to_add:
            log.debug("Adding to ACL", path=self.path, ac=to_add)
            item = self._to_dict()
            item[Baton.ACCESS] = to_add
            with client(self.pool) as c:
                c.ac_set(item, recurse=recurse, timeout=timeout, tries=tries)

        return len(to_add)

    def ac_rem(
        self, *acs: Union[AC, Tuple[AC]], recurse=False, timeout=None, tries=1
    ) -> int:
        """Remove access controls from the item. Return the number of access
        controls removed. If some of the argument access controls are not
        present, those arguments will be ignored.

        Args:
            acs: Access controls.
        Keyword Args
            recurse: Recursively add access control.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

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
            with client(self.pool) as c:
                c.ac_set(item, recurse=recurse, timeout=timeout, tries=tries)

        return len(to_remove)

    def ac_supersede(
        self, *acs: Union[AC, Tuple[AC]], recurse=False, timeout=None, tries=1
    ) -> Tuple[int, int]:
        """Remove all access controls from the item, replacing them with the
        specified access controls. Return the numbers of access controls
        removed and added.

        Args:
            acs: Access controls.
        Keyword Args:
            recurse: Recursively supersede access controls.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

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
            with client(self.pool) as c:
                c.ac_set(item, recurse=recurse, timeout=timeout, tries=tries)

        to_add = sorted(set(acs).difference(current))
        if to_add:
            log.debug("Adding to ACL", path=self.path, ac=to_add)
            item = self._to_dict()
            item[Baton.ACCESS] = to_add
            with client(self.pool) as c:
                c.ac_set(item, recurse=recurse, timeout=timeout, tries=tries)

        return len(to_remove), len(to_add)

    def metadata(self, timeout=None, tries=1) -> List[AVU]:
        """Return the item's metadata.

        Keyword Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: List[AVU]
        """
        item = self._list(avu=True, timeout=timeout, tries=tries).pop()
        if Baton.AVUS not in item.keys():
            raise BatonError(f"{Baton.AVUS} key missing from {item}")

        return sorted(item[Baton.AVUS])

    def acl(self, timeout=None, tries=1) -> List[AC]:
        """Return the item's Access Control List (ACL).

        Keyword Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: List[AC]"""
        item = self._list(acl=True, timeout=timeout, tries=tries).pop()
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

    def __init__(self, remote_path: Union[PurePath, str], pool=default_pool):
        super().__init__(PurePath(remote_path).parent, pool=pool)
        self.name = PurePath(remote_path).name

    def list(self, timeout=None, tries=1) -> DataObject:
        """Return a new DataObject representing this one.

        Keyword Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: DataObject
        """
        item = self._list(timeout=timeout, tries=tries).pop()
        if Baton.OBJ not in item.keys():
            raise BatonError(f"{Baton.OBJ} key missing from {item}")

        return _make_rods_item(item, pool=self.pool)

    def checksum(
        self,
        calculate_checksum=False,
        recalculate_checksum=False,
        verify_checksum=False,
        timeout=None,
        tries=1,
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
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: str
        """

        item = self._to_dict()
        with client(self.pool) as c:
            return c.checksum(
                item,
                calculate_checksum=calculate_checksum,
                recalculate_checksum=recalculate_checksum,
                verify_checksum=verify_checksum,
                timeout=timeout,
                tries=tries,
            )

    def get(
        self, local_path: Union[Path, str], verify_checksum=True, timeout=None, tries=1
    ) -> int:
        """Get the data object from iRODS"""
        item = self._to_dict()
        with client(self.pool) as c:
            return c.get(
                item,
                Path(local_path),
                verify_checksum=verify_checksum,
                timeout=timeout,
                tries=tries,
            )

    def put(
        self,
        local_path: Union[Path, str],
        calculate_checksum=False,
        verify_checksum=True,
        force=True,
        timeout=None,
        tries=1,
    ):
        """Put the data object into iRODS.

        Args:
            local_path: The local path of a file to put into iRODS at the path
            specified by this data object.
        Keyword Args:
            calculate_checksum: Calculate remote checksums for all replicates. If
            checksums exist, this is s no-op.
            verify_checksum: Verify the local checksum against the remote checksum.
            Verification implies checksum calculation.
            force: Overwrite any data object already present in iRODS.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
        """
        item = self._to_dict()
        with client(self.pool) as c:
            c.put(
                item,
                Path(local_path),
                calculate_checksum=calculate_checksum,
                verify_checksum=verify_checksum,
                force=force,
                timeout=timeout,
                tries=tries,
            )

    def read(self, timeout=None, tries=1) -> str:
        """Get the data object from iRODS."""
        item = self._to_dict()
        with client(self.pool) as c:
            return c.read(item, timeout=timeout, tries=tries)

    def _list(self, **kwargs) -> List[dict]:
        item = self._to_dict()
        with client(self.pool) as c:
            return c.list(item, **kwargs)

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

    def __init__(self, path: Union[PurePath, str], pool=default_pool):
        super().__init__(path, pool=pool)

    def contents(
        self, acl=False, avu=False, recurse=False, timeout=None, tries=1
    ) -> List[Union[DataObject, Collection]]:
        """Return list of the Collection contents.

        Keyword Args:
          acl: Include ACL information.
          avu: Include AVU (metadata) information.
          recurse: Recurse into sub-collections.
          timeout: Operation timeout in seconds.
          tries: Number of times to try the operation.

        Returns: List[Union[DataObject, Collection]]
        """
        items = self._list(
            acl=acl,
            avu=avu,
            contents=True,
            recurse=recurse,
            timeout=timeout,
            tries=tries,
        )
        return [_make_rods_item(item, pool=self.pool) for item in items]

    def list(self, acl=False, avu=False, timeout=None, tries=1) -> Collection:
        """Return a new Collection representing this one.

        Keyword Args:
          acl: Include ACL information.
          avu: Include AVU (metadata) information.

        Returns: Collection
        """
        items = self._list(acl=acl, avu=avu, timeout=timeout, tries=tries)
        # Gets a single item
        return _make_rods_item(items.pop(), pool=self.pool)

    def get(self, local_path: Union[Path, str], **kwargs):
        raise NotImplementedError()

    def put(self, local_path: Union[Path, str], recurse=True, timeout=None, tries=1):
        raise NotImplementedError()

    def _list(self, **kwargs) -> List[dict]:
        with client(self.pool) as c:
            return c.list({Baton.COLL: self.path}, **kwargs)

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


def _make_rods_item(item: Dict, pool: BatonPool) -> Union[DataObject, Collection]:
    """Create a new Collection or DataObject as appropriate for a dictionary
    returned by a Baton.

    Returns: Union[DataObject, Collection]
    """
    if Baton.COLL not in item.keys():
        raise BatonError(f"{Baton.COLL} key missing from {item}")

    if Baton.OBJ in item.keys():
        return DataObject(PurePath(item[Baton.COLL], item[Baton.OBJ]), pool=pool)
    return Collection(PurePath(item[Baton.COLL]), pool=pool)
