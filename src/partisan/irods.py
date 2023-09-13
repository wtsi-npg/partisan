# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021, 2022, 2023 Genome Research Ltd. All rights
# reserved.
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
from functools import total_ordering, wraps
from os import PathLike
from pathlib import Path, PurePath
from queue import LifoQueue, Queue
from threading import Thread
from typing import Annotated, Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import dateutil.parser
from structlog import get_logger

from partisan.exception import (
    BatonError,
    BatonTimeoutError,
    InvalidEnvelopeError,
    InvalidJSONError,
    RodsError,
)
from partisan.icommands import iquest, itrim, iuserinfo

log = get_logger(__name__)

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

    DIR = "directory"
    FILE = "file"
    SIZE = "size"

    REPLICAS = "replicates"  # Replicas is the newer iRODS terminology
    RESOURCE = "resource"
    LOCATION = "location"
    CHECKSUM = "checksum"
    NUMBER = "number"

    TIMESTAMPS = "timestamps"
    CREATED = "created"
    MODIFIED = "modified"

    CHMOD = "chmod"
    LIST = "list"
    MKDIR = "mkdir"
    GET = "get"
    PUT = "put"
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
            """Report anything from client STDERR to the error log. There should be
            virtually no traffic here."""
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
        replicas=False,
        size=False,
        timestamp=False,
        timeout=None,
        tries=1,
    ) -> List[Dict]:
        """Lists i.e. reports on items in iRODS.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be suitable input for baton-do.
            acl: Include ACL information in the result.
            avu: Include AVU information in the result.
            contents: Include contents in the result (for a collection item).
            replicas: Include replica information in the result.
            size: Include size information in the result (for a data object).
            timestamp: Include timestamp information in the result (for a data object).
            timeout: Operation timeout.
            tries: Number of times to try the operation.

        Returns:
            A single Dict when listing a data object or single collection, or multiple
            Dicts when listing a collection's contents.
        """
        result = self._execute(
            Baton.LIST,
            {
                "acl": acl,
                "avu": avu,
                "contents": contents,
                "replicate": replicas,
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
        """Perform remote checksum operations.

        Args:
            item: A dictionary representing the item When serialized as JSON,
                this must be suitable input for baton-do.
            calculate_checksum: Ask iRODS to calculate the checksum, if there is no
                remote checksum currently.
            recalculate_checksum: Ask iRODS to calculate the checksum, even if there
                is a remote checksum currently.
            verify_checksum: Verify the remote checksum against the data.
            timeout: Operation timeout.
            tries: Number of times to try the operation.

        Returns: The checksum.
        """
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

    def add_metadata(self, item: Dict, timeout=None, tries=1):
        """Add metadata to an item in iRODS.

        Args:
            item: A dictionary representing the item When serialized as JSON,
            this must be suitable input for baton-do.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
        self._execute(
            Baton.METAMOD, {Baton.OP: Baton.ADD}, item, timeout=timeout, tries=tries
        )

    def remove_metadata(self, item: Dict, timeout=None, tries=1):
        """Remove metadata from an item in iRODS.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be suitable input for baton-do.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
        self._execute(
            Baton.METAMOD, {Baton.OP: Baton.REM}, item, timeout=timeout, tries=tries
        )

    def query_metadata(
        self,
        avus: List[AVU],
        zone=None,
        collection=False,
        data_object=False,
        timeout=None,
        tries=1,
    ) -> Dict:
        """Query metadata in iRODS.

        Args:
            avus: The query, expressed as AVUs.
            zone: The iRODS zone name.
            collection: Query collection metadata, default false.
            data_object: Query data object metadata, default false.
            timeout: Operation timeout.
            tries: Number of times to try the operation.

        Returns: The query result.
        """
        args = {}
        if collection:
            args["collection"] = True
        if data_object:
            args["object"] = True

        item = {Baton.AVUS: avus}
        if zone:
            item[Baton.COLL] = self._zone_hint_to_path(zone)

        return self._execute(Baton.METAQUERY, args, item, timeout=timeout, tries=tries)

    def set_permission(self, item: Dict, recurse=False, timeout=None, tries=1):
        """Set access permissions on a data object or collection.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be suitable input for baton-do.
            recurse: Recursively set permissions on a collection.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
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
        """Get a data object from iRODS.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be suitable input for baton-do.
            local_path: A local path to create.
            verify_checksum: Verify the data object's checksum on download.
            force: Overwrite any existing file.
            timeout: Operation timeout.
            tries: Number of times to try the operation.

        Returns: The number of bytes downloaded.
        """
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

        item[Baton.DIR] = local_path.parent
        item[Baton.FILE] = local_path.name

        self._execute(
            Baton.GET,
            {"save": True, "verify": verify_checksum, "force": force},
            item,
            timeout=timeout,
            tries=tries,
        )
        return local_path.stat().st_size

    def read(self, item: Dict, timeout=None, tries=1) -> str:
        """Read the contents of a data object as a string.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be suitable input for baton-do.
            timeout: Operation timeout.
            tries: Number of times to try the operation.

        Returns: The data object contents as a string.
        """
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
        """Put a data object into iRODS.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be suitable input for baton-do.
            local_path: The path of a file to upload.
            calculate_checksum: Calculate a remote checksum.
            verify_checksum: Verify the remote checksum after upload.
            force: Overwrite any existing data object.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
        item[Baton.DIR] = local_path.parent
        item[Baton.FILE] = local_path.name

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

    def create_collection(self, item: Dict, parents=False, timeout=None, tries=1):
        """Create a new collection.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be suitable input for baton-do.
            parents: Create the collection's parents, if necessary.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
        self._execute(
            Baton.MKDIR, {"recurse": parents}, item, timeout=timeout, tries=tries
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
        # connection. In order to time out these bad operations, round-trips to the
        # server are run in their own thread which provides API for managing the
        # timeout behaviour.
        #
        # Not all long-duration API calls are bad, so timeouts must be set by
        # operation type. A "put" operation of a multi-GiB file may legitimately take
        # significant time, a metadata change may not.
        lifo = LifoQueue(maxsize=1)

        t = Thread(target=lambda q, w: q.put(self._send(w)), args=(lifo, wrapped))
        t.start()

        for i in range(tries):
            t.join(timeout=timeout)
            if not t.is_alive():
                break
            log.warning("Timed out sending", client=self, tryno=i, doc=wrapped)

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
        """Get a client from the pool. If a timeout is supplied, waiting up to the
        timeout.

        Args:
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
        """Put a client back into the pool. If a timeout is supplied, waiting up to the
        timeout.

        Args:
            c: A baton client
            timeout: Timeout to put a client, in seconds. Raises queue.Full if the
            operation times out.
        """
        log.debug(f"Returning a client to the pool: {c}")
        self._queue.put(c, timeout=timeout)


@contextmanager
def client_pool(maxsize=4) -> BatonPool:
    """Yields a pool of clients that will be closed automatically when the pool goes
    out of scope.

    Args:
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
        timeout: Timeout for both getting the client and putting it back, in seconds.
    Raises:
         queue.Empty or queue.Full if the get or put operations respectively, time out.

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


def query_metadata(
    *avus: AVU,
    zone=None,
    collection=True,
    data_object=True,
    timeout=None,
    tries=1,
    pool=default_pool,
) -> List[Union[DataObject, Collection]]:
    """
    Query all metadata in iRODS (i.e. both on collections and data objects)

    Args:
        *avus: One or more AVUs to query.
        zone: Zone hint for the query. Defaults to None (query the current zone).
        collection: Query the collection namespace. Defaults to True.
        data_object: Query the data object namespace. Defaults to True.
        timeout: Operation timeout in seconds.
        tries: Number of times to try the operation.
        pool: Client pool to use. If omitted, the default pool is used.

    Returns: List[Union[DataObject, Collection]]
    """
    with client(pool) as c:
        result = c.query_metadata(
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

    ACs may be sorted, where they will be sorted lexically, first by
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

        if zone is not None:
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
            and self.perm == other.perm
            and (
                (self.zone is None and other.zone is None)
                or (
                    self.zone is not None
                    and other.zone is not None
                    and self.zone == other.zone
                )
            )
        )

    def __lt__(self, other):
        if self.zone is not None and other.zone is None:
            return True

        if self.zone is None and other.zone is not None:
            return False

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

    AVUs may be sorted, where they will be sorted lexically, first by
    namespace (if present), then by attribute, then by value and finally by
    units (if present).
    """

    SEPARATOR = ":"
    """The attribute namespace separator"""

    HISTORY_SUFFIX = "_history"
    """The attribute history suffix"""

    def __init__(self, attribute: Any, value: Any, units=None, namespace=None):
        """Create a new AVU instance.

        Args:
            attribute: The attribute to use. If this is not a string, the string
               representation of this argument becomes the attribute.
            value: The value to use. If this is not a string, the string
               representation of this argument becomes the value.
            units: The units to use. Optional, defaults to None.
            namespace: The namespace (prefix) to be used for the attribute. Optional,
               but if supplied, must be the same as any existing namespace on the
               attribute string.
        """
        if attribute is None:
            raise ValueError("AVU attribute may not be None")
        if value is None:
            raise ValueError("AVU value may not be None")
        if namespace is None:
            namespace = ""

        if namespace.find(AVU.SEPARATOR) >= 0:
            raise ValueError(
                f"AVU namespace contained a separator '{AVU.SEPARATOR}': '{namespace}'"
            )

        attr = str(attribute)
        if attr.find(AVU.SEPARATOR) >= 0:
            ns, at = attr.split(AVU.SEPARATOR, maxsplit=1)
            if namespace and ns != namespace:
                raise ValueError(
                    f"AVU attribute namespace '{ns}' did not match "
                    f"the declared namespace '{namespace}' for "
                    f"attribute '{attr}', value '{value}'"
                )
            namespace, attr = ns, at

        self._namespace = namespace
        self._attribute = attr
        self._value = str(value)
        self._units = units

    @classmethod
    def collate(cls, *avus: AVU) -> Dict[str : List[AVU]]:
        """Collates AVUs by attribute (including namespace, if any) and
        returns a dict mapping the attribute to a list of AVUs with that
        attribute.

        Args:
            avus: One or more AVUs to collate.

        Returns: Dict[str: List[AVU]]
        """
        collated = defaultdict(lambda: list())

        for avu in avus:
            collated[avu.attribute].append(avu)

        return collated

    @classmethod
    def history(cls, *avus: AVU, history_date=None) -> AVU:
        """Returns a history AVU describing the argument AVUs. A history AVU is
        sometimes added to an iRODS path to describe AVUs that were once
        present, but have been removed. Adding a history AVU can act as a poor
        man's audit trail. It used because iRODS does not have native history support.

        Args:
            avus: AVUs removed, which must share the same attribute
                and namespace (if any).
            history_date: A datetime to be embedded as part of the history
            AVU value.

        Returns: AVU
        """
        if history_date is None:
            history_date = datetime.utcnow()
        date = format_timestamp(history_date)

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
            and (
                (self.units is None and other.units is None)
                or (
                    self.units is not None
                    and other.units is not None
                    and self.units == other.units
                )
            )
        )

    def __lt__(self, other):
        if self.namespace and not other.namespace:
            return True

        if not self.namespace and other.namespace:
            return False

        if self.namespace and other.namespace:
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


@total_ordering
class Replica(object):
    """An iRODS data object replica.

    iRODS may maintain multiple copies of the data backing a data object. Each one of
    these as modeled as a Replica instance. Every data object has at least one Replica.
    """

    def __init__(
        self,
        resource: str,
        location: str,
        number: int,
        created=None,
        modified=None,
        checksum=None,
        valid=True,
    ):
        if resource is None:
            raise ValueError("Replica resource may not be None")
        if location is None:
            raise ValueError("Replica location may not be None")
        if number is None:
            raise ValueError("Replica number may not be None")

        self.resource = resource
        self.location = location
        self.number = number
        self.created = created
        self.modified = modified
        self.checksum = checksum
        self.valid = valid

    def __hash__(self):
        return (
            hash(self.number)
            + hash(self.resource)
            + hash(self.location)
            + hash(self.checksum)
            + hash(self.valid)
        )

    def __eq__(self, other):
        if not isinstance(other, Replica):
            return False

        # Timestamps are intentionally not included in checking equality because they
        # do not affect replica identity (defined by resource, location, number and
        # whether they are valid)

        return (
            self.number == other.number
            and self.resource == other.resource
            and self.location == other.location
            and (
                (self.checksum is None and other.checksum is None)
                or (
                    self.checksum is not None
                    and other.checksum is not None
                    and self.checksum == other.checksum
                )
            )
            and self.valid == other.valid
        )

    def __lt__(self, other):
        if self.number < other.number:
            return True

        if self.number == other.number:
            if self.resource < other.resource:
                return True

            if self.resource == other.resource:
                if self.location < other.location:
                    return True

                if self.location == other.location:
                    if self.checksum is not None and other.checksum is None:
                        return True

                    if self.checksum is None and other.checksum is not None:
                        return False

                    if self.checksum is not None and other.checksum is not None:
                        if self.checksum < other.checksum:
                            return True

                    if self.checksum == other.checksum:
                        return self.valid < other.valid

        return False

    def __repr__(self):
        return f"{self.number}:{self.resource}:{self.checksum}:{self.valid}"

    def __str__(self):
        return (
            f"<Replica {self.number} {self.resource} checksum={self.checksum} "
            f"created={self.created} modified={self.modified} "
            f"valid={self.valid}>"
        )


@total_ordering
class User(object):
    """An iRODS user.

    iRODS represents both individual user accounts and groups of users as "users".
    Users are compared for equality by Partisan on a combination of their user ID
    and their zone.
    """

    def __init__(self, name: str, id: str, type: str, zone: str):
        self.name = name
        self.id = id
        self.type = type
        self.zone = zone

    def is_rodsadmin(self):
        """Return True if the user is a rodsadmin."""
        return self.type == "rodsadmin"

    def is_group(self):
        """Return True if the user represents a rodsgroup."""
        return self.type == "rodsgroup"

    def is_rodsuser(self):
        """Return True if the user is a rodsuser."""
        return self.type == "rodsuser"

    def __hash__(self):
        return hash(self.id) + hash(self.zone)

    def __eq__(self, other):
        if not isinstance(other, User):
            return False

        return self.id == other.id and self.zone == other.zone

    def __lt__(self, other):
        if self.zone < other.zone:
            return True
        if self.id < other.id:
            return True

        return False

    def __repr__(self):
        return f"<{self.name}#{self.zone} ({self.type})>"

    def __str__(self):
        return f"{self.name}#{self.zone}"


def rods_user(name: str = None) -> Optional[User]:
    """Return information about an iRODS user.

    Args:
        name: A username. Optional, defaults to the name of the current user.

    Returns: A new instance of User.
    """
    ui = {}
    for line in iuserinfo(name).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            if key in ["name", "id", "type", "zone"]:
                ui[key] = value.strip()

    if not ui:
        # This is valid when the user is on present another zone. iuserinfo cannot query
        # across zones.
        return None

    return User(ui["name"], ui["id"], ui["type"], ui["zone"])


def rods_users(user_type: str = None, zone=None) -> list[User]:
    """Return a list of iRODS users registered in the specified zone, optionally
    limited to a specific user type.

    Args:
        user_type: An iRODS user type to select. Only users of this type will be
            reported. Optional, one of rodsadmin, rodsgroup or rodsuser.
        zone: An iRODS zone on which to run the query. Note that this NOT the same
            as a user's zone. e.g. user alice#foo may have permissions on zone bar.
            Using an argument zone=bar is asking for the query to run on zone bar,
            which may then return alice#foo i.e. a user with zone other than bar.

    Returns:
        A list of users in the specified zone.
    """
    if user_type is not None and user_type not in [
        "rodsadmin",
        "rodsgroup",
        "rodsuser",
    ]:
        raise ValueError(f"Invalid user type requested: {user_type}")

    args = []
    if zone is not None:
        args.extend(["-z", zone])
    args.extend(["%s\t%s\t%s\t%s", "select USER_NAME, USER_ID, USER_TYPE, USER_ZONE"])

    users = []
    for line in iquest(*args).splitlines():
        name, uid, utype, zone = line.split("\t")
        users.append(User(name, uid, utype, zone))

    if user_type is not None:
        users = [user for user in users if user.type == user_type]

    return sorted(users)


def current_user() -> User:
    """Return the current iRODS user.

    Returns: The user's name and their zone.
    """
    return rods_user()


def rods_type_check(method):
    """Add a check to RodsItem methods that ensures the item's path in iRODS has the
    appropriate type. i.e. that a Collection has a collection path and a DataObject has
    a data object path."""

    @wraps(method)
    def wrapper(*args, **kwargs):
        item = args[0]
        item.check_rods_type(**kwargs)
        return method(*args, **kwargs)

    return wrapper


def rods_path_exists(
    path: Union[PurePath, str], timeout=None, tries=1, pool=default_pool
) -> bool:
    """Return true if the specified path is a collection or data object in iRODS.

    Args:
        path: A remote path.
        timeout: Operation timeout in seconds.
        tries: Number of times to try the operation.
        pool: A baton client pool. Optional.

    Returns:
        True if the path exists.
    """
    return rods_path_type(path, timeout=timeout, tries=tries, pool=pool) is not None


def rods_path_type(
    path: Union[PurePath, str], timeout=None, tries=1, pool=default_pool
) -> Optional[Type[RodsItem]]:
    """Return a Python type representing the kind of iRODS path supplied,
    e.g. Collection for and iRODS collection, DataObject for an iRODS data object. If
    the path does not exist, returns None.

    Args:
        path: A remote path.
        timeout: Operation timeout in seconds.
        tries: Number of times to try the operation.
        pool: A baton client pool. Optional.

    Returns:
        Union[DataObject, Collection], or None.
    """
    try:
        with client(pool) as c:
            match c.list({Baton.COLL: path}, timeout=timeout, tries=tries):
                case [{Baton.COLL: _, Baton.OBJ: _}]:
                    return DataObject
                case [{Baton.COLL: _}]:
                    return Collection
                case [item]:
                    raise ValueError(f"Failed to recognised client response {item}")
    except RodsError as re:
        if re.code == -310000:  # iRODS error code for path not found
            return None
        raise re


def format_timestamp(ts: datetime) -> str:
    """Return a formatted representation of a timestamp, suitable for use in iRODS
    metadata.

    Args:
        ts: The timestamp to format
    """
    return ts.isoformat(timespec="seconds")


def make_rods_item(
    path: Union[PurePath, str], pool=default_pool
) -> Union[Collection, DataObject]:
    """A factory function for iRODS items.

    Args:
        path: A remote path.
        pool: A baton client pool. Optional.

    Returns:
        A Collection or DataObject, as appropriate.
    """
    with client(pool) as c:
        item = c.list({Baton.COLL: path}).pop()
        return _make_rods_item(item, pool=pool)


class RodsItem(PathLike):
    """A base class for iRODS path entities."""

    def __init__(self, path: Union[PurePath, str], check_type=False, pool=default_pool):
        """RodsItem constructor.

        Args:
            path: A remote path.
            check_type: Check the remote path type if True, defaults to False.
            pool: A baton client pool. Optional.
        """
        self.path = PurePath(path)
        self.check_type = check_type
        self.pool = pool

        self._rods_type = None

    def _exists(self, timeout=None, tries=1) -> bool:
        try:
            self._list(timeout=timeout, tries=tries)
        except RodsError as re:
            if re.code == -310000:  # iRODS error code for path not found
                return False
        return True

    @rods_type_check
    def exists(self, timeout=None, tries=1) -> bool:
        """Return True if the item exists in iRODS.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
        """
        return self._exists(timeout=timeout, tries=tries)

    def avu(self, attribute: Any, timeout=None, tries=1) -> AVU:
        """Return an unique AVU from the item's metadata, given an attribute, or raise
        an error.

        Args:
            attribute: The attribute of the expected AVU. If this is not a string,
               the string representation of this argument is used.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            A single AVU with the specified attribute.
        """
        attr = str(attribute)
        avus = [
            avu
            for avu in self.metadata(timeout=timeout, tries=tries)
            if avu.attribute == attr
        ]

        if not avus:
            raise ValueError(
                f"Metadata of {self} did not contain any AVU with "
                f"attribute '{attr}'"
            )
        if len(avus) > 1:
            raise ValueError(
                f"Metadata of '{self}' contained more than one AVU with "
                f"attribute '{attr}': {avus}"
            )

        return avus[0]

    def has_metadata(self, *avus: AVU, timeout=None, tries=1) -> bool:
        """Return True if all the argument AVUs are in the item's metadata.

        Args:
            *avus: One or more AVUs to test.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: True if every AVU (i.e. key, value and optionally, unit) is present.
        """
        return set(avus).issubset(self.metadata(timeout=timeout, tries=tries))

    def has_metadata_attrs(self, *attributes: Any, timeout=None, tries=1) -> bool:
        """Return True if all the argument attributes are in the item's metadata.

        Args:
            *attributes: One or more attributes to test. If any of these are not strings,
               their string representations is used.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: True if every attribute is present in at least one AVU.
        """
        collated = self.collated_metadata(timeout=timeout, tries=tries)
        attrs = [str(a) for a in attributes]
        return set(attrs).issubset(collated.keys())

    @rods_type_check
    def add_metadata(self, *avus: AVU, timeout=None, tries=1) -> int:
        """Add AVUs to the item's metadata, if they are not already present.
        Return the number of AVUs added.

        Args:
            *avus: One or more AVUs to add.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        current = self.metadata()
        to_add = sorted(set(avus).difference(current))

        if to_add:
            log.debug("Adding AVUs", path=self, avus=to_add)
            item = self._to_dict()
            item[Baton.AVUS] = to_add
            with client(self.pool) as c:
                c.add_metadata(item, timeout=timeout, tries=tries)

        return len(to_add)

    @rods_type_check
    def remove_metadata(self, *avus: AVU, timeout=None, tries=1) -> int:
        """Remove AVUs from the item's metadata, if they are present.
        Return the number of AVUs removed.

        Args:
            *avus: One or more AVUs to remove.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        current = self.metadata()
        to_remove = sorted(set(current).intersection(avus))

        if to_remove:
            log.debug("Removing AVUs", path=self, avus=to_remove)
            item = self._to_dict()
            item[Baton.AVUS] = to_remove
            with client(self.pool) as c:
                c.remove_metadata(item, timeout=timeout, tries=tries)

        return len(to_remove)

    @rods_type_check
    def supersede_metadata(
        self,
        *avus: AVU,
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
             avus: One or more AVUs to add in place of existing AVUs sharing
                 those attributes.
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
        log.debug("Superseding AVUs", path=self, old=current, new=avus)

        rem_attrs = set(map(lambda avu: avu.attribute, avus))
        to_remove = set(filter(lambda a: a.attribute in rem_attrs, current))

        # If the argument AVUs have some AVUs to remove amongst them, we don't want
        # to remove them from the item, just to add them back.
        to_remove.difference_update(avus)
        to_remove = sorted(to_remove)
        if to_remove:
            log.debug("Removing AVUs", path=self, avus=to_remove)
            item = self._to_dict()
            item[Baton.AVUS] = to_remove
            with client(self.pool) as c:
                c.remove_metadata(item, timeout=timeout, tries=tries)

        to_add = sorted(set(avus).difference(current))
        if history:
            hist = []
            for avus in AVU.collate(*to_remove).values():
                hist.append(AVU.history(*avus, history_date=history_date))
            to_add += hist

        if to_add:
            log.debug("Adding AVUs", path=self, avus=to_add)
            item = self._to_dict()
            item[Baton.AVUS] = to_add
            with client(self.pool) as c:
                c.add_metadata(item, timeout=timeout, tries=tries)

        return len(to_remove), len(to_add)

    @rods_type_check
    def add_permissions(self, *acs: AC, timeout=None, tries=1) -> int:
        """Add access controls to the item. Return the number of access
        controls added. If some argument access controls are already present,
        those arguments will be ignored.

        This method handles only the permissions of this RodsItem. See the Collection
        class for methods handling recursive operations.

        Args:
            *acs: Access controls.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        current = self.acl()
        to_add = sorted(set(acs).difference(current))
        log.debug(
            "Adding to ACL",
            path=self,
            curr=current,
            arg=acs,
            add=to_add,
        )

        if to_add:
            item = self._to_dict()
            item[Baton.ACCESS] = to_add
            with client(self.pool) as c:
                c.set_permission(item, timeout=timeout, tries=tries)

        return len(to_add)

    @rods_type_check
    def remove_permissions(self, *acs: AC, timeout=None, tries=1) -> int:
        """Remove access controls from the item. Return the number of access
        controls removed. If some argument access controls are not present, those
        arguments will be ignored.

        This method handles only the permissions of this RodsItem. See the Collection
        class for methods handling recursive operations.

        Args:
            *acs: Access controls.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        current = self.acl()
        to_remove = sorted(set(current).intersection(acs))
        log.debug(
            "Removing from ACL",
            path=self,
            curr=current,
            arg=acs,
            rem=to_remove,
        )

        if to_remove:
            # In iRODS we "remove" permissions by setting them to NULL
            to_null = [AC(ac.user, Permission.NULL, zone=ac.zone) for ac in to_remove]

            item = self._to_dict()
            item[Baton.ACCESS] = to_null
            with client(self.pool) as c:
                c.set_permission(item, timeout=timeout, tries=tries)

        return len(to_remove)

    @rods_type_check
    def supersede_permissions(self, *acs: AC, timeout=None, tries=1) -> Tuple[int, int]:
        """Remove all access controls from the item, replacing them with the
        specified access controls. Return the numbers of access controls
        removed and added.

        This method handles only the permissions of this RodsItem. See the Collection
        class for methods handling recursive operations.

        Args:
            *acs: Access controls.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: Tuple[int, int]
        """
        current = self.acl()
        log.debug("Superseding ACL", path=self, old=current, new=acs)

        to_remove = sorted(set(current).difference(acs))
        if to_remove:
            log.debug("Removing from ACL", path=self, ac=to_remove)

            # In iRODS we "remove" permissions by setting them to NULL
            to_null = [AC(ac.user, Permission.NULL, zone=ac.zone) for ac in to_remove]

            item = self._to_dict()
            item[Baton.ACCESS] = to_null
            with client(self.pool) as c:
                c.set_permission(item, timeout=timeout, tries=tries)

        to_add = sorted(set(acs).difference(current))
        if to_add:
            log.debug("Adding to ACL", path=self, ac=to_add)
            item = self._to_dict()
            item[Baton.ACCESS] = to_add
            with client(self.pool) as c:
                c.set_permission(item, timeout=timeout, tries=tries)

        return len(to_remove), len(to_add)

    @rods_type_check
    def metadata(self, attribute: Any = None, timeout=None, tries=1) -> List[AVU]:
        """Return the item's metadata.

        Args:
            attribute: Return only AVUs having this attribute. If this is not a string,
               the string representation of this argument is used.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: List[AVU]
        """
        item = self._list(avu=True, timeout=timeout, tries=tries).pop()
        if Baton.AVUS not in item:
            raise BatonError(f"{Baton.AVUS} key missing from {item}")

        avus = item[Baton.AVUS]
        if attribute is not None:
            attr = str(attribute)
            avus = [avu for avu in avus if avu.attribute == attr]

        return sorted(avus)

    def collated_metadata(self, timeout=None, tries=1) -> dict[str:list]:
        """Return a dictionary mapping AVU attributes to lists of corresponding AVU
        values.

        This method collates AVU values under their shared key. E.g. if an item had the
        AVUs AVU("Key1", "Value1") and AVU("Key1", "Value2"), the collated dictionary
        would be {"Key1": ["Value1", "Value2"]}.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: Dict[str:List]
        """
        collated = defaultdict(list)
        for avu in self.metadata(timeout=timeout, tries=tries):
            collated[avu.attribute].append(avu.value)

        return collated

    def permissions(self, user_type: str = None, timeout=None, tries=1) -> List[AC]:
        """Return the item's Access Control List (ACL). Synonym for acl().

        Args:
            user_type: Filter to include only permissions for users of this type.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: List[AC]
        """
        return self.acl(user_type=user_type, timeout=timeout, tries=tries)

    @rods_type_check
    def acl(self, user_type: str = None, timeout=None, tries=1) -> List[AC]:
        """Return the item's Access Control List (ACL). Synonym for permissions().

        Args:
            user_type: Filter to include only permissions for users of this type.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: List[AC]
        """
        if user_type is not None and user_type not in [
            "rodsadmin",
            "rodsgroup",
            "rodsuser",
        ]:
            raise ValueError(f"Invalid user type requested: {type}")

        item = self._list(acl=True, timeout=timeout, tries=tries).pop()
        if Baton.ACCESS not in item:
            raise BatonError(f"{Baton.ACCESS} key missing from {item}")

        # iRODS ACL queries can sometimes return multiple results where a user appears
        # to have has both "own" and "read" permissions simultaneously. Since "own"
        # subsumes "read" this is confusing and can have unwanted effects e.g. copying
        # permissions from one collection to another will effectively remove "own" if
        # the source collection has both "own" and "read", and the "read" permission
        # is copied after "own". Yes - when copying these permissions to another item,
        # iRODS now treats "own" and "read" are states that cannot be held
        # simultaneously and will delete the first when the second is applied.
        #
        # The source collection in the cases we observe are created with the iput
        # icommand, the destination collection with the mkdir API call.
        by_user = {}
        for ac in item[Baton.ACCESS]:
            key = (ac.user, ac.zone)
            if key in by_user:
                by_user[key].add(ac)
            else:
                by_user[key] = {ac}

        acl = []
        for key, acs in by_user.items():
            # If the item apparently has both "own" and "read" permissions, for a user,
            # only report "own".
            if len(acs) > 1:
                own = {x for x in acs if x.perm == Permission.OWN}
                read = {x for x in acs if x.perm == Permission.READ}
                if own and read:
                    acs.difference_update(read)
            acl.extend(acs)

        if user_type is not None:
            by_name = {user.name: user for user in rods_users(user_type=user_type)}
            acl = [ac for ac in acl if ac.user not in by_name]

        return sorted(acl)

    def __lt__(self, other):
        if isinstance(self, Collection) and isinstance(other, DataObject):
            return True
        if isinstance(self, DataObject) and isinstance(other, Collection):
            return False

        return self.path < other.path

    @abstractmethod
    def rods_type(self) -> Optional[Type[RodsItem]]:
        """Return a Python type representing the kind of iRODS path supplied."""

    pass

    @abstractmethod
    def check_rods_type(self, **kwargs):
        """Raise an error if the item does not have an appropriate type of iRODS path.
        e.g. raise an error if a Collection object has the path of a data object."""
        pass

    @abstractmethod
    def get(self, local_path: Union[Path, str], **kwargs):
        """Get the item from iRODS."""
        pass

    @abstractmethod
    def put(self, local_path: Union[Path, str], **kwargs):
        """Put the item into iRODS."""
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

    def __init__(
        self,
        remote_path: Union[PurePath, str],
        check_type=True,
        pool=default_pool,
    ):
        """DataObject constructor.

        Args:
            remote_path: A remote data object path.
            check_type: Check the remote path type if True, defaults to True.
            pool: A baton client pool. Optional.
        """
        super().__init__(PurePath(remote_path).parent, check_type=check_type, pool=pool)
        self.name = PurePath(remote_path).name

    @classmethod
    def query_metadata(
        cls,
        *avus: AVU,
        zone=None,
        timeout=None,
        tries=1,
        pool=default_pool,
    ) -> list[DataObject]:
        """Query data object metadata in iRODS.

        Args:
            *avus: One or more AVUs to query.
            zone: Zone hint for the query. Defaults to None (query the current zone).
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
            pool: Client pool to use. If omitted, the default pool is used.

        Returns: A list of data objects with matching metadata.
        """

        with client(pool) as c:
            items = c.query_metadata(
                avus,
                zone=zone,
                collection=False,
                data_object=True,
                timeout=timeout,
                tries=tries,
            )

        objects = [_make_rods_item(item, pool=pool) for item in items]
        objects.sort()
        return objects

    @property
    def rods_type(self):
        """Return a Python type representing the kind of iRODS path supplied."""
        if self._rods_type is None:
            self._rods_type = rods_path_type(PurePath(self.path, self.name))
        return self._rods_type

    def check_rods_type(self, **kwargs):
        """Raise an error if the path is not a data object in iRODS."""
        if not self.check_type:
            return

        rt = self.rods_type
        if rt is not None and rt != DataObject:
            raise BatonError(f"Invalid iRODS path type {rt} for a data object: {self}")

    @rods_type_check
    def list(self, timeout=None, tries=1) -> DataObject:
        """Return a new DataObject representing this one.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: DataObject
        """
        item = self._list(timeout=timeout, tries=tries).pop()
        return _make_rods_item(item, pool=self.pool)

    @rods_type_check
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

        Args:
            calculate_checksum: Calculate remote checksums for all replicas. If
                checksums exist, this is a no-op.
            recalculate_checksum: Force recalculation of remote checksums for all
                replicas.
            verify_checksum: Verify the local checksum against the remote checksum.
            Verification implies checksum calculation.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: A checksum
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

    @rods_type_check
    def size(self, timeout=None, tries=1) -> int:
        """Return the size of the data object according to the iRODS IES database, in
        bytes.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        item = self._list(size=True, timeout=timeout, tries=tries).pop()
        return item[Baton.SIZE]

    @rods_type_check
    def timestamp(self, timeout=None, tries=1) -> datetime:
        """Return the timestamp of the data object according to the iRODS IES
        database. This is a synonym for the `modified` method.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The data object's earliest modification timestamp
        """
        return self.modified(timeout=timeout, tries=tries)

    def created(self, timeout=None, tries=1):
        """Return the creation timestamp of the data object according to the
        iRODS IES database.

        There exist in the IES a creation timestamp and a modification timestamp for
        each replica. This method returns the creation timestamp.

        If the data object has more than ore replica, the earliest of the
        created timestamps is returned. The rationale for this is that the
        earliest timestamp is likely to be the time at which creation was
        initiated and any later timestamps, the time at which other replicas were
        made consistent.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The data object's earliest creation timestamp
        """
        return min([r.created for r in self.replicas(timeout=timeout, tries=tries)])

    def modified(self, timeout=None, tries=1) -> datetime:
        """Return the modification timestamp of the data object according to the
        iRODS IES database.

        There exist in the IES a creation timestamp and a modification timestamp for
        each replica. This method returns the modification timestamp.

        If the data object has more than ore replica, the earliest of the
        modification timestamps is returned. The rationale for this is that the
        earliest timestamp is likely to be the time at which modification was
        initiated and any later timestamps, the time at which other replicas were
        made consistent.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The data object's earliest modified timestamp
        """
        return min([r.modified for r in self.replicas(timeout=timeout, tries=tries)])

    @rods_type_check
    def replicas(self, timeout=None, tries=1) -> List[Replica]:
        """Return the replicas of the data object.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The object's replicas
        """

        item = self._list(replicas=True, timeout=timeout, tries=tries).pop()
        if Baton.REPLICAS not in item:
            raise BatonError(f"{Baton.REPLICAS} key missing from {item}")

        rep_args = {}
        for rep_val in item[Baton.REPLICAS]:
            match rep_val:
                case {Baton.NUMBER: n}:
                    rep_args[n] = rep_val
                case _:
                    raise BatonError(f"{Baton.NUMBER} key missing from {rep_val}")

        # Getting timestamps from baton currently requires a separate call from
        # getting replica information. The JSON property returned is (for two
        # replicas, in this example), of the form:
        #
        # 'timestamps': [{'created': '2022-09-09T11:11:03Z', 'replicates': 0},
        #                {'modified': '2022-09-09T11:11:03Z', 'replicates': 0},
        #                {'created': '2022-09-09T11:11:03Z', 'replicates': 1},
        #                {'modified': '2022-09-09T11:11:03Z', 'replicates': 1}]}

        item = self._list(timestamp=True, timeout=timeout, tries=tries).pop()
        if Baton.TIMESTAMPS not in item:
            raise BatonError(f"{Baton.TIMESTAMPS} key missing from {item}")

        for ts_val in item[Baton.TIMESTAMPS]:
            if Baton.REPLICAS not in ts_val:
                raise BatonError(f"{Baton.REPLICAS} key missing from {ts_val}")
            rep_num = ts_val[Baton.REPLICAS]

            match ts_val:
                case {Baton.CREATED: t}:
                    rep_args[rep_num][Baton.CREATED] = dateutil.parser.isoparse(t)
                case {Baton.MODIFIED: t}:
                    rep_args[rep_num][Baton.MODIFIED] = dateutil.parser.isoparse(t)
                case _:
                    raise BatonError(
                        f"{Baton.CREATED}/{Baton.MODIFIED} key missing "
                        f"from {ts_val}"
                    )

        replicas = [Replica(**args) for args in rep_args.values()]
        replicas.sort()

        return replicas

    @rods_type_check
    def get(
        self, local_path: Union[Path, str], verify_checksum=True, timeout=None, tries=1
    ):
        """Get the data object from iRODS and save to a local file.

        Args:
            local_path: The local path of a file to be created.
            verify_checksum: Verify the local checksum against the remote checksum.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
        """
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
            calculate_checksum: Calculate remote checksums for all replicas. If
                checksums exist, this is a no-op.
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

    @rods_type_check
    def read(self, timeout=None, tries=1) -> str:
        """Read the data object from iRODS into a string. This operation is supported
        for data objects containing UTF-8 text.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: str
        """
        item = self._to_dict()
        with client(self.pool) as c:
            return c.read(item, timeout=timeout, tries=tries)

    def trim_replicas(self, min_replicas=2, valid=False, invalid=True) -> (int, int):
        """Trim excess and invalid replicas of the data object.

        Args:
            min_replicas: The minimum number of valid replicas after the operation.
            valid: Trim valid replicas. Optional, defaults to False.
            invalid: Trim invalid replicas. Optional, defaults to True.

        Returns: The number of valid replicas trimmed and the number of invalid
            replicas trimmed.
        """
        vr = [r for r in self.replicas() if r.valid]
        ir = [r for r in self.replicas() if not r.valid]

        valid_trimmed = 0
        invalid_trimmed = 0
        path = str(self)

        if valid:
            trimmable = vr[min_replicas:]
            if not trimmable:
                log.warn(
                    "Not trimming valid replicas below minimum count",
                    path=path,
                    num_replicas=min_replicas,
                )
            for r in trimmable:
                valid_trimmed += 1
                itrim(path, r.number, min_replicas)

        if invalid:
            for r in ir:
                invalid_trimmed += 1
                itrim(path, r.number, min_replicas)

        return valid_trimmed, invalid_trimmed

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

    def __hash__(self):
        return hash(self.path) + hash(self.name)

    def __fspath__(self):
        return self.__repr__()

    def __repr__(self):
        return PurePath(self.path, self.name).as_posix()


class Collection(RodsItem):
    """An iRODS collection.

    Collection is a PathLike for the iRODS path it represents.
    """

    @classmethod
    def query_metadata(
        cls,
        *avus: AVU,
        zone=None,
        timeout=None,
        tries=1,
        pool=default_pool,
    ) -> list[Collection]:
        """Query collection metadata in iRODS.

        Args:
            *avus: AVUs to query.
            zone: Zone hint for the query. Defaults to None (query the current zone).
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
            pool: Client pool to use. If omitted, the default pool is used.

        Returns: A list of collections with matching metadata.
        """
        with client(pool) as c:
            items = c.query_metadata(
                avus,
                zone=zone,
                collection=True,
                data_object=False,
                timeout=timeout,
                tries=tries,
            )

        collections = [_make_rods_item(item, pool=pool) for item in items]
        collections.sort()
        return collections

    def __init__(
        self, remote_path: Union[PurePath, str], check_type=True, pool=default_pool
    ):
        """Collection constructor.

        Args:
            remote_path: A remote collection path.
            check_type: Check the remote path type if True, defaults to True.
            pool: A baton client pool. Optional.
        """
        super().__init__(remote_path, check_type=check_type, pool=pool)

    def create(self, parents=False, exist_ok=False, timeout=None, tries=1):
        """Create a new, empty Collection on the server side.

        Args:
            parents: Create parent collections as necessary.
            exist_ok: If the collection exists, do not raise an error.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
        """
        if exist_ok and self.exists():
            return

        item = self._to_dict()
        with client(self.pool) as c:
            c.create_collection(item, parents=parents, timeout=timeout, tries=tries)

    @property
    def rods_type(self):
        """Return a Python type representing the kind of iRODS path supplied."""
        if self._rods_type is None:
            self._rods_type = rods_path_type(self.path)
        return self._rods_type

    def check_rods_type(self, **kwargs):
        """Raise an error if the path is not a collection in iRODS."""
        if not self.check_type:
            return

        rt = self.rods_type
        if rt is not None and rt != Collection:
            raise BatonError(f"Invalid iRODS path type {rt} for a collection: {self}")

    @rods_type_check
    def contents(
        self, acl=False, avu=False, recurse=False, timeout=None, tries=1
    ) -> List[Union[DataObject, Collection]]:
        """Return a list of the Collection contents.

        Args:
          acl: Include ACL information.
          avu: Include AVU (metadata) information.
          recurse: Recurse into sub-collections.
          timeout: Operation timeout in seconds.
          tries: Number of times to try the operation.

        Returns: A list of collections and data objects directly in the collection.
        """
        items = self._list(
            acl=acl,
            avu=avu,
            contents=True,
            timeout=timeout,
            tries=tries,
        )

        contents = [_make_rods_item(item, pool=self.pool) for item in items]

        collect = []
        if recurse:
            for elt in contents:
                elt.path = self.path / elt.path  # Make an absolute path
                collect.append(elt)

                if isinstance(elt, Collection):
                    collect.extend(
                        elt.contents(
                            acl=acl,
                            avu=avu,
                            recurse=recurse,
                            timeout=timeout,
                            tries=tries,
                        )
                    )
        else:
            collect = contents

        collect.sort()
        return collect

    @rods_type_check
    def iter_contents(
        self, acl=False, avu=False, recurse=False, timeout=None, tries=1
    ) -> Iterable[Union[DataObject, Collection]]:
        """Return a generator for the Collection contents.

        Args:
          acl: Include ACL information.
          avu: Include AVU (metadata) information.
          recurse: Recurse into sub-collections in depth-first order.
          timeout: Operation timeout in seconds.
          tries: Number of times to try the operation.

        Returns: Iterable[Union[DataObject, Collection]]"""
        items = self._list(
            acl=acl,
            avu=avu,
            contents=True,
            timeout=timeout,
            tries=tries,
        )

        contents = [_make_rods_item(item, pool=self.pool) for item in items]
        contents.sort()

        for elt in contents:
            elt.path = self.path / elt.path  # Make an absolute path
            yield elt
            if recurse:
                if isinstance(elt, Collection):
                    yield from elt.iter_contents(
                        acl=acl,
                        avu=avu,
                        recurse=recurse,
                        timeout=timeout,
                        tries=tries,
                    )

    @rods_type_check
    def list(self, acl=False, avu=False, timeout=None, tries=1) -> Collection:
        """Return a new Collection representing this one.

        Args:
          acl: Include ACL information.
          avu: Include AVU (metadata) information.
          timeout: Operation timeout in seconds.
          tries: Number of times to try the operation.

        Returns: Collection
        """
        item = self._list(acl=acl, avu=avu, timeout=timeout, tries=tries).pop()
        return _make_rods_item(item, pool=self.pool)

    @rods_type_check
    def get(self, local_path: Union[Path, str], **kwargs):
        """Get the collection and contents from iRODS and save to a local directory.

        Args:
            local_path: The local path of a directory to be created.
        Keyword Args:
            **kwargs:
        """
        raise NotImplementedError()

    def put(self, local_path: Union[Path, str], recurse=True, timeout=None, tries=1):
        """Put the collection into iRODS.

        Args:
            local_path: The local path of a directory to put into iRODS at the path
                specified by this collection.
            recurse: Recurse through subdirectories.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
        """
        raise NotImplementedError()

    def add_permissions(self, *acs: AC, recurse=False, timeout=None, tries=1) -> int:
        """Add access controls to the collection. Return the number of access
        controls added. If some argument access controls are already present,
        those arguments will be ignored.

        Args:
            *acs: Access controls.
            recurse: Recursively add access controls.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        num_added = super().add_permissions(*acs, timeout=timeout, tries=tries)
        if recurse:
            for item in self.iter_contents(recurse=recurse):
                num_added += item.add_permissions(*acs, timeout=timeout, tries=tries)
        return num_added

    def remove_permissions(self, *acs: AC, recurse=False, timeout=None, tries=1) -> int:
        """Remove access controls from the collection. Return the number of access
        controls removed. If some argument access controls are not present, those
        arguments will be ignored.

        Args:
            *acs: Access controls.
            recurse: Recursively remove access controls.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: int
        """
        num_removed = super().remove_permissions(*acs, timeout=timeout, tries=tries)
        if recurse:
            for item in self.iter_contents(recurse=recurse):
                num_removed += item.remove_permissions(
                    *acs, timeout=timeout, tries=tries
                )
        return num_removed

    def supersede_permissions(
        self, *acs: AC, recurse=False, timeout=None, tries=1
    ) -> Tuple[int, int]:
        """Remove all access controls from the collection, replacing them with the
        specified access controls. Return the numbers of access controls
        removed and added.

        Args:
            *acs: Access controls.
            recurse: Recursively supersede access controls.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: Tuple[int, int]
        """
        num_removed, num_added = super().supersede_permissions(
            *acs, timeout=timeout, tries=tries
        )
        if recurse:
            for item in self.iter_contents(recurse=recurse):
                nr, na = item.supersede_permissions(*acs, timeout=timeout, tries=tries)
                num_removed += nr
                num_added += na
        return num_removed, num_added

    def _list(self, **kwargs) -> List[dict]:
        with client(self.pool) as c:
            return c.list({Baton.COLL: self.path}, **kwargs)

    def _to_dict(self):
        return {Baton.COLL: self.path}

    def __eq__(self, other):
        if not isinstance(other, Collection):
            return False

        return self.path == other.path

    def __hash__(self):
        return hash(self.path)

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
    match item:
        case {Baton.COLL: c, Baton.OBJ: o}:
            log.debug(f"Making a DataObject from {item}")
            return DataObject(PurePath(c, o), pool=pool)
        case {Baton.COLL: c}:
            log.debug(f"Making a Collection from {item}")

            return Collection(PurePath(c), pool=pool)
        case _:
            raise BatonError(f"{Baton.COLL} key missing from {item}")
