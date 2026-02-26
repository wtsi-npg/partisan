# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021, 2022, 2023, 2024, 2025, 2026 Genome Research
# Ltd. All rights reserved.
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

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto, unique
from functools import total_ordering, wraps
from os import PathLike
from pathlib import Path, PurePath
from queue import LifoQueue, Queue
from threading import Thread
from typing import (
    Annotated,
    Any,
    Generator,
    Iterable,
    Type,
)

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

log = get_logger()

"""This module provides a basic API for accessing iRODS using the native
iRODS client 'baton' (https://github.com/wtsi-npg/baton).
"""


class Baton:
    """A wrapper around the baton-do client program, used for interacting with
    iRODS.
    """

    CLIENT = "baton-do"

    BACKOFF_FACTOR = 2.0
    BACKOFF_MAX = 60.0

    AVUS = "avus"
    ATTRIBUTE = "attribute"
    VALUE = "value"
    UNITS = "units"
    OPERATOR = "operator"

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

        version = client_version()
        if version < (6, 0, 0):
            ver_str = ".".join([str(i) for i in version])
            raise BatonError(
                "This version of partisan requires a baton version >=6.0.0 "
                f"(detected version '{ver_str}')"
            )

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
        item: dict,
        acl=False,
        avu=False,
        checksum=False,
        contents=False,
        replicas=False,
        size=False,
        timestamp=False,
        timeout=None,
        tries=1,
    ) -> list[dict]:
        """Lists i.e. reports on items in iRODS.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be a suitable input for baton-do.
            acl: Include ACL information in the result.
            avu: Include AVU information in the result.
            checksum: Include checksum information in the result (for a data object).
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
                "checksum": checksum,
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
            item: A dictionary representing the item. When serialized as JSON,
                this must be a suitable input for baton-do.
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

    def add_metadata(self, item: dict, timeout=None, tries=1):
        """Add metadata to an item in iRODS.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
            this must be a suitable input for baton-do.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
        self._execute(
            Baton.METAMOD, {Baton.OP: Baton.ADD}, item, timeout=timeout, tries=tries
        )

    def remove_metadata(self, item: dict, timeout=None, tries=1):
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
        avus: list[AVU],
        timestamps: list[Timestamp] | None = None,
        zone=None,
        collection=False,
        data_object=False,
        timeout=None,
        tries=1,
    ) -> dict:
        """Query metadata in iRODS.

        Args:
            avus: The query, expressed as AVUs.
            timestamps: A list of timestamps to narrow the search. Each timestamp value
                and operator is combined into the query (using AND logic, if there are
                more than one).
            zone: An iRODS zone hint. This can be the name of a zone to search or a path
                into a zone. If a path is used, results outside that collection will
                be removed from any results. If None, results from the current zone
                will be returned.
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

        if timestamps is not None:
            item[Baton.TIMESTAMPS] = timestamps
        if zone is not None:
            item[Baton.COLL] = self._zone_hint_to_path(zone)

        return self._execute(Baton.METAQUERY, args, item, timeout=timeout, tries=tries)

    def set_permission(self, item: dict, recurse=False, timeout=None, tries=1):
        """Set access permissions on a data object or collection.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be a suitable input for baton-do.
            recurse: Recursively set permissions on a collection.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
        self._execute(
            Baton.CHMOD, {"recurse": recurse}, item, timeout=timeout, tries=tries
        )

    def get(
        self,
        item: dict,
        local_path: Path,
        force=True,
        verify_checksum=False,
        redirect=False,
        timeout=None,
        tries=1,
    ) -> int:
        """Get a data object from iRODS.

        Args:
            item: A dictionary representing the item. When serialised as JSON,
                this must be a suitable input for baton-do.
            local_path: A local path to create.
            verify_checksum: Verify the data object's checksum on download.
                Defaults to False.
            redirect: Redirect the operation to the best server, decided by iRODS.
                Defaults to False.
            timeout: Operation timeout.
            tries: Number of times to try the operation.

        Returns: The number of bytes downloaded.
        """

        item[Baton.DIR] = local_path.parent
        item[Baton.FILE] = local_path.name

        self._execute(
            Baton.GET,
            {
                "force": force,
                "save": True,
                "verify": verify_checksum,
                "redirect": redirect,
            },
            item,
            timeout=timeout,
            tries=tries,
        )
        return local_path.stat().st_size

    def read(self, item: dict, timeout=None, tries=1) -> str:
        """Read the contents of a data object as a string.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be a suitable input for baton-do.
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
        item: dict,
        local_path: Path,
        calculate_checksum=False,
        force=True,
        verify_checksum=False,
        redirect=False,
        timeout=None,
        tries=1,
    ):
        """Put a data object into iRODS.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be a suitable input for baton-do.
            local_path: The path of a file to upload.
            calculate_checksum: Calculate a remote checksum.
            verify_checksum: Verify the remote checksum after upload.
            redirect: Redirect the operation to the best server, decided by iRODS.
                Defaults to False.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
        item[Baton.DIR] = local_path.parent
        item[Baton.FILE] = local_path.name

        self._execute(
            Baton.PUT,
            {
                "checksum": calculate_checksum,
                "force": force,
                "verify": verify_checksum,
                "redirect": redirect,
            },
            item,
            timeout=timeout,
            tries=tries,
        )

    def create_collection(self, item: dict, parents=False, timeout=None, tries=1):
        """Create a new collection.

        Args:
            item: A dictionary representing the item. When serialized as JSON,
                this must be a suitable input for baton-do.
            parents: Create the collection's parents, if necessary.
            timeout: Operation timeout.
            tries: Number of times to try the operation.
        """
        self._execute(
            Baton.MKDIR, {"recurse": parents}, item, timeout=timeout, tries=tries
        )

    def _execute(
        self, operation: str, args: dict, item: dict, timeout=None, tries=1
    ) -> dict:
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
        def _start_send():
            q = LifoQueue(maxsize=1)
            thread = Thread(target=lambda q, w: q.put(self._send(w)), args=(q, wrapped))
            thread.start()
            return q, thread

        lifo, t = _start_send()

        def _backoff_timeout(base_timeout, attempt: int):
            if base_timeout is None:
                return None
            if base_timeout <= 0:
                return 0

            timeout = base_timeout * (self.BACKOFF_FACTOR**attempt)
            if self.BACKOFF_MAX is not None:
                timeout = min(timeout, self.BACKOFF_MAX)

            return timeout

        for i in range(tries):
            attempt_timeout = _backoff_timeout(timeout, i)
            t.join(timeout=attempt_timeout)
            if t.is_alive():
                log.warning(
                    "Timed out sending",
                    client=self,
                    tryno=i,
                    doc=wrapped,
                    timeout=attempt_timeout,
                )
                continue

            response = lifo.get(timeout=0.1)

            try:
                return self._unwrap(response)
            except RodsError as e:
                if i >= tries - 1:
                    raise
                log.warning(
                    "RodsError, retrying",
                    client=self,
                    tryno=i,
                    code=e.code,
                    msg=str(e),
                )
                lifo, t = _start_send()

        # Still alive after all the tries?
        if t.is_alive():
            self.stop()
            raise BatonTimeoutError(
                "Exhausted all timeouts, stopping client", client=self, tryno=tries
            )

        raise BatonError(
            f"Baton '{operation}' operation on {item} " "finished without a response"
        )

    @staticmethod
    def _wrap(operation: str, args: dict, item: dict) -> dict:
        return {
            Baton.OP: operation,
            Baton.ARGS: args,
            Baton.TARGET: item,
        }

    @staticmethod
    def _unwrap(envelope: dict) -> dict:
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

    def _send(self, envelope: dict) -> dict:
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

        def hook(d: dict) -> Any:
            """Object hook for partially decoding baton JSON (just AVUs and ACs)."""

            # Match an AVU sub-document
            if Baton.ATTRIBUTE in d:
                attr = d[Baton.ATTRIBUTE]
                value = d[Baton.VALUE]
                units = d.get(Baton.UNITS, None)
                return AVU(attr, value, units)

            # Match an access permission sub-document
            if Baton.OWNER in d and Baton.LEVEL in d:
                user = d[Baton.OWNER]
                zone = d[Baton.ZONE]
                level = d[Baton.LEVEL]

                return AC(user, Permission[level.upper()], zone=zone)

            return d

        return json.loads(resp, object_hook=hook)

    @staticmethod
    def _zone_hint_to_path(zone) -> str:
        z = str(zone)
        if z.startswith("/"):
            return z

        return "/" + z


class BatonPool:
    """A pool of Baton clients."""

    def __init__(self, maxsize=4):
        self.maxsize = maxsize
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

        if not c.is_running():
            log.warn(f"Client returned to the pool is not running: {c}")

        self._queue.put(c, timeout=timeout)


@contextmanager
def client_pool(maxsize=4) -> Generator[BatonPool, Any, None]:
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
def client(pool: BatonPool, timeout=None) -> Generator[Baton, Any, None]:
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
    timestamps: list[Timestamp] | None = None,
    zone=None,
    collection=True,
    data_object=True,
    timeout=None,
    tries=1,
    pool=default_pool,
) -> list[Collection | DataObject]:
    """
    Query all metadata in iRODS (i.e. both on collections and data objects)

    Args:
        *avus: One or more AVUs to query.
        timestamps: A list of Timestamp objects to narrow the search. Each timestamp
            value and operator is combined into the query (using AND logic, if there
            are more than one).
        zone: An iRODS zone hint. This can be the name of a zone to search or a path
            into a zone. If a path is used, results outside that collection will
            be removed from any results. If None, results from the current zone
            will be returned.
        collection: Query the collection namespace. Defaults to True.
        data_object: Query the data object namespace. Defaults to True.
        timeout: Operation timeout in seconds.
        tries: Number of times to try the operation.
        pool: Client pool to use. If omitted, the default pool is used.

    Returns: A list of collections and data objects matching the query.
    """
    with client(pool) as c:
        result = c.query_metadata(
            avus=avus,
            timestamps=timestamps,
            zone=zone,
            collection=collection,
            data_object=data_object,
            timeout=timeout,
            tries=tries,
        )
        items = [_make_rods_item(item, pool=pool) for item in result]
        items.sort()

        return items


class Timestamp:
    """An iRODS path creation or modification timestamp."""

    @unique
    class Event(Enum):
        """The types of event marked by timestamps."""

        CREATED = auto()
        MODIFIED = auto()

    def __init__(
        self,
        value: datetime,
        event: Event = Event.MODIFIED,
        operator: str = "n>=",
    ):
        """Create a new Timestamp instance.

        Args:
            value: The datetime of the event.
            event: The type of event.
            operator: An operator to use when searching for timestamps. Must be one of
                the timestamp operators used by iRODS (>, <, <=, >=, n>=, n<=, n>, n<).
                The default is >=.
        """
        self.value = value
        self.event = event
        self.operator = operator

    def __repr__(self):
        return (
            f"<Timestamp {self.value.isoformat(timespec='seconds')}, {self.event.name}, "
            f"op: '{self.operator}'>"
        )


@unique
class Permission(Enum):
    """The kinds of data access permission available to iRODS users."""

    NULL = "null"
    OWN = "own"
    READ = "read"
    WRITE = "write"


@total_ordering
class AC:
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
class AVU:
    """AVU is an iRODS attribute, value, units tuple.

    AVUs may be sorted, where they will be sorted lexically, first by
    namespace (if present), then by attribute, then by value and finally by
    units (if present).
    """

    SEPARATOR = ":"
    """The attribute namespace separator"""

    IRODS_NAMESPACE = "irods"

    IRODS_SEPARATOR = "::"
    """The attribute namespace separator used by iRODS' system AVUs"""

    HISTORY_SUFFIX = "_history"
    """The attribute history suffix"""

    def __init__(
        self,
        attribute: Any,
        value: Any,
        units: str = None,
        namespace: str = None,
        operator: str = None,
    ):
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
            operator: The operator to use if the AVU is used as a query argument.
                Optional, defaults to '='. Must be one of the available iRODS
                query operators. Operators are not considered when comparing AVUs.
        """
        if attribute is None:
            raise ValueError("AVU attribute may not be None")
        if value is None:
            raise ValueError("AVU value may not be None")
        if namespace is None:
            namespace = ""
        if operator is None:
            operator = "="

        attr = str(attribute)
        if re.match(r"\s+$", attr):
            raise ValueError("AVU attribute may not be entirely whitespace")

        # If the operator is `IN`, then the value will be a collection, otherwise it is
        # something to stringify
        operator = operator.lower()
        if operator != "in":
            value = str(value)
            if re.match(r"\s+$", value):
                raise ValueError("AVU value may not be entirely whitespace")

        if re.match(r"\s+$", namespace):
            raise ValueError("AVU namespace may not be entirely whitespace")

        # Handle iRODS' own namespaced AVUs
        if attr.startswith(AVU.IRODS_NAMESPACE) and (
            attr.find(AVU.IRODS_SEPARATOR) == len(AVU.IRODS_NAMESPACE)
        ):
            self._separator = AVU.IRODS_SEPARATOR
        # Handle all other AVUs, namespaced or not
        else:
            self._separator = AVU.SEPARATOR

        if namespace and namespace.find(self._separator) >= 0:
            raise ValueError(
                f"AVU namespace contained a separator '{self._separator}': "
                f"'{namespace}'"
            )

        if attr.find(self._separator) >= 0:
            ns, at = attr.split(self._separator, maxsplit=1)
            if namespace and ns != namespace:
                raise ValueError(
                    f"AVU attribute namespace '{ns}' did not match "
                    f"the declared namespace '{namespace}' for "
                    f"attribute '{attr}', value '{value}'"
                )
            namespace, attr = ns, at

        self._namespace = namespace
        self._attribute = attr
        self._value = value
        self._units = units
        self._operator = operator

    @classmethod
    def collate(cls, *avus: AVU) -> dict[str, list[AVU]]:
        """Collates AVUs by attribute (including namespace, if any) and
        returns a dict mapping the attribute to a list of AVUs with that
        attribute.

        Args:
            avus: One or more AVUs to collate.

        Returns: A mapping of each attribute to a list of AVUs with that attribute.
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
            history_date = datetime.now(timezone.utc)
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
        separated by AVU.SEPARATOR (or AVU.IRODS_SEPARATOR in the case of iRODS'
        internal AVUs)."""
        if self._namespace:
            return f"{self._namespace}{self._separator}{self._attribute}"
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

    @property
    def operator(self):
        """The operator associated with the AVU. The default is '='."""
        return self._operator

    def with_namespace(self, namespace: str):
        """make a new copy of this AVU with the specified namespace.

        Args:
            namespace: The new namespace.

        Returns: A new AVU with the specified namespace.
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
class Replica:
    """An iRODS data object replica.

    iRODS may maintain multiple copies of the data backing a data object. Each one of
    these is modelled as a Replica instance. Every data object has at least one Replica.
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
class User:
    """An iRODS user.

    iRODS represents both individual user accounts and groups of users as "users".
    Users are compared for equality by Partisan on a combination of their user ID
    and their zone.
    """

    def __init__(self, name: str, user_id: str, user_type: str, zone: str):
        self.name = name
        self.id = user_id
        self.type = user_type
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


def rods_user(name: str = None) -> User | None:
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
        "groupadmin",
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


def client_version() -> tuple[int, ...]:
    """Return the baton client version."""
    completed = subprocess.run(["baton-do", "--version"], capture_output=True)
    if completed.returncode == 0:
        version = completed.stdout.decode("utf-8").strip()
        number = version.split("-", maxsplit=1)
        if not number:
            raise BatonError(f"Failed parse client version '{version}")

        return tuple(int(i) for i in number[0].split("."))

    raise BatonError(completed.stderr.decode("utf-8").strip())


def server_version() -> tuple[int, ...]:
    """Return the version reported by the iRODS server."""
    completed = subprocess.run(["baton-do", "--server-version"], capture_output=True)
    if completed.returncode == 0:
        version = completed.stdout.decode("utf-8").strip()
        number = version.split("-", maxsplit=1)
        if not number:
            raise BatonError(f"Failed parse server version '{version}")

        return tuple(int(i) for i in number[0].split("."))

    raise RodsError(completed.stderr.decode("utf-8").strip())


def connected(method):
    """Add a check to RodsItem methods that ensures the item is connected to iRODS."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.connected():
            raise BatonError(
                f"Cannot perform operation '{method.__name__}' on '{self}' unless connected:"
            )
        return method(self, *args, **kwargs)

    return wrapper


def rods_type_check(method):
    """Add a check to RodsItem methods that ensures the item's path in iRODS has the
    appropriate type, i.e. that a Collection has a collection path and a DataObject has
    a data object path."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.connected():
            self.check_rods_type(**kwargs)
        return method(self, *args, **kwargs)

    return wrapper


def rods_path_exists(
    path: PurePath | str, timeout=None, tries=1, pool=default_pool
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
    path: PurePath | str, timeout=None, tries=1, pool: BatonPool = default_pool
) -> Type[RodsItem] | None:
    """Return a Python type representing the kind of iRODS path supplied.

    e.g. Collection for an iRODS collection, DataObject for an iRODS data object.
    The value returned is what the iRODS server recognises the path as. If the
    remote path does not exist, returns None.

    Args:
        path: A remote path.
        timeout: Operation timeout in seconds.
        tries: Number of times to try the operation.
        pool: A baton client pool. Optional.

    Returns:
        Collection | DataObject, or None.
    """
    try:
        with client(pool) as c:
            match c.list({Baton.COLL: path}, timeout=timeout, tries=tries):
                case [{Baton.COLL: _, Baton.OBJ: _}]:
                    return DataObject
                case [{Baton.COLL: _}]:
                    return Collection
                case [item]:
                    raise ValueError(f"Failed to recognised client response '{item}'")
    except RodsError as e:
        if e.code == -310000:  # iRODS error code for path not found
            return None
        raise e


def format_timestamp(ts: datetime) -> str:
    """Return a formatted representation of a timestamp, suitable for use in iRODS
    metadata.

    Args:
        ts: The timestamp to format
    """
    return ts.isoformat(timespec="seconds")


def make_rods_item(path: PurePath | str, pool=default_pool) -> Collection | DataObject:
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
    """A base class for iRODS path entities.

    RodsItems can be either 'connected' or not, depending on whether they have a
    BatonPool associated with them.

    If they are connected, they can interact with the iRODS server and retrieve or
    update remote information. If they are not connected, their main use is in their
    ability to be serialized/deserialized to/from JSON, including their metadata and
    ACLs.

    The decision to connect or not is made on instance construction by passing a BatonPool
    and cannot be changed later. This is to avoid the complexity of reconciling the two
    sets of metadata and ACLs (local and remote) that would be required. The best way
    to convert an item from disconnected to connected is to create a new instance with
    the same path and a pool, then copy the metadata and ACLs across.
    """

    INTERNAL_TIMEOUT = 10.0
    INTERNAL_TRIES = 3

    def __init__(
        self,
        remote_path: PurePath | str,
        local_path: Path | str = None,
        check_type=False,
        pool: BatonPool | None = default_pool,
    ):
        """RodsItem constructor.

        Args:
            remote_path: A remote path.
            check_type: Check the remote path type if True, defaults to False.
            pool: A baton client pool. Optional.
        """
        self.path = PurePath(remote_path)
        self.local_path = local_path
        self.check_type = check_type
        self._pool = pool

        self._rods_type = None

        self._local_metadata = set()
        self._local_acl = set()

    def _exists(self, timeout=None, tries=1) -> bool:
        try:
            self._list(timeout=timeout, tries=tries)
        except RodsError as e:
            if e.code == -310000:  # iRODS error code for path not found
                return False
        return True

    @rods_type_check
    @connected
    def exists(self, timeout=None, tries=1) -> bool:
        """Return True if the item exists in iRODS.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
        """
        return self._exists(timeout=timeout, tries=tries)

    def connected(self):
        """Return True if the item is connected."""
        return self._pool is not None

    def avu(self, attribute: Any, ancestors=False, timeout=None, tries=1) -> AVU:
        """Return an unique AVU from the item's metadata, given an attribute, or raise
        an error.

        Args:
            attribute: The attribute of the expected AVU. If this is not a string,
               the string representation of this argument is used.
            ancestors: Include metadata collated from ancestor collections.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            A single AVU with the specified attribute.
        """
        attr = str(attribute)
        avus = [
            avu
            for avu in self.metadata(ancestors=ancestors, timeout=timeout, tries=tries)
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

    def has_metadata(self, *avus: AVU, ancestors=False, timeout=None, tries=1) -> bool:
        """Return True if all the argument AVUs are in the item's metadata.

        Args:
            *avus: One or more AVUs to test.
            ancestors: Include metadata collated from ancestor collections.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: True if every AVU (i.e. key, value and optionally, unit) is present.
        """
        return set(avus).issubset(
            self.metadata(ancestors=ancestors, timeout=timeout, tries=tries)
        )

    def has_metadata_attrs(
        self, *attributes: Any, ancestors=False, timeout=None, tries=1
    ) -> bool:
        """Return True if all the argument attributes are in the item's metadata.

        Args:
            *attributes: One or more attributes to test. If any of these are not strings,
               their string representation is used.
            ancestors: Include metadata collated from ancestor collections.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: True if every attribute is present in at least one AVU.
        """
        collated = self.collated_metadata(
            ancestors=ancestors, timeout=timeout, tries=tries
        )
        attrs = [str(a) for a in attributes]
        return set(attrs).issubset(collated.keys())

    @rods_type_check
    def add_metadata(self, *avus: AVU, timeout=None, tries=1) -> int:
        """Add AVUs to the item's metadata if they are not already present.
        Return the number of AVUs added.

        Args:
            *avus: One or more AVUs to add.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The number of AVUs added.
        """
        current = self.metadata()
        to_add = sorted(set(avus).difference(current))

        if to_add:
            log.debug("Adding AVUs", path=self, avus=to_add)

            if not self.connected():
                self._local_metadata.update(to_add)
            else:
                item = self.to_dict()
                item[Baton.AVUS] = to_add
                with client(self._pool) as c:
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

        Returns: The number of AVUs removed.
        """
        current = self.metadata()
        to_remove = sorted(set(current).intersection(avus))

        if to_remove:
            log.debug("Removing AVUs", path=self, avus=to_remove)

            if not self.connected():
                self._local_metadata.difference_update(to_remove)
            else:
                item = self.to_dict()
                item[Baton.AVUS] = to_remove
                with client(self._pool) as c:
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
    ) -> tuple[int, int]:
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
            history_date = datetime.now(timezone.utc)

        current = self.metadata()

        rem_attrs = {avu.attribute for avu in avus}
        to_remove = {a for a in current if a.attribute in rem_attrs}

        # If the argument AVUs have some AVUs to remove amongst them, we don't want
        # to remove them from the item, just to add them back.
        to_remove.difference_update(avus)
        to_remove = sorted(to_remove)
        to_add = sorted(set(avus).difference(current))
        log.debug(
            "Preparing AVUs",
            path=self,
            current=current,
            new=[*avus],
            add=to_add,
            rem=to_remove,
        )

        if to_remove:
            log.info("Removing AVUs", path=self, avus=to_remove)

            if not self.connected():
                self._local_metadata.difference_update(to_remove)
            else:
                item = self.to_dict()
                item[Baton.AVUS] = to_remove
                with client(self._pool) as c:
                    c.remove_metadata(item, timeout=timeout, tries=tries)

        if history:
            hist = []
            for avus in AVU.collate(*to_remove).values():
                hist.append(AVU.history(*avus, history_date=history_date))
            to_add += hist

        if to_add:
            log.info("Adding AVUs", path=self, avus=to_add)

            if not self.connected():
                self._local_metadata.update(to_add)
            else:
                item = self.to_dict()
                item[Baton.AVUS] = to_add
                with client(self._pool) as c:
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

        Returns: The number of access controls added.
        """
        current = self.acl()
        to_add = sorted(set(acs).difference(current))
        log.debug("Preparing ACL", path=self, curr=current, arg=acs, add=to_add)

        if to_add:
            log.info("Adding to ACL", path=self, ac=to_add)

            if not self.connected():
                self._local_acl.update(to_add)
            else:
                item = self.to_dict()
                item[Baton.ACCESS] = to_add
                with client(self._pool) as c:
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

        Returns: The number of access controls removed.
        """
        current = self.acl()
        to_remove = sorted(set(current).intersection(acs))
        log.debug("Preparing ACL", path=self, curr=current, arg=acs, rem=to_remove)

        if to_remove:
            log.info("Removing from ACL", path=self, ac=to_remove)
            # In iRODS we "remove" permissions by setting them to NULL
            to_null = [AC(ac.user, Permission.NULL, zone=ac.zone) for ac in to_remove]

            if not self.connected():
                self._local_acl.difference_update(to_remove)
            else:
                item = self.to_dict()
                item[Baton.ACCESS] = to_null
                with client(self._pool) as c:
                    c.set_permission(item, timeout=timeout, tries=tries)

        return len(to_remove)

    @rods_type_check
    def supersede_permissions(self, *acs: AC, timeout=None, tries=1) -> tuple[int, int]:
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
        to_remove = sorted(set(current).difference(acs))
        to_add = sorted(set(acs).difference(current))
        log.debug(
            "Preparing ACL",
            path=self,
            current=current,
            new=acs,
            add=to_add,
            rem=to_remove,
        )

        if to_remove:
            log.info("Removing from ACL", path=self, ac=to_remove)
            # In iRODS we "remove" permissions by setting them to NULL
            to_null = [AC(ac.user, Permission.NULL, zone=ac.zone) for ac in to_remove]

            if not self.connected():
                self._local_acl.difference_update(to_remove)
            else:
                item = self.to_dict()
                item[Baton.ACCESS] = to_null
                with client(self._pool) as c:
                    c.set_permission(item, timeout=timeout, tries=tries)

        if to_add:
            log.info("Adding to ACL", path=self, ac=to_add)

            if not self.connected():
                self._local_acl.update(to_add)
            else:
                item = self.to_dict()
                item[Baton.ACCESS] = to_add
                with client(self._pool) as c:
                    c.set_permission(item, timeout=timeout, tries=tries)

        return len(to_remove), len(to_add)

    def ancestors(self) -> list[Collection]:
        """Return a list of the item's ancestors.

        This list ultimately includes the root collection "/", to be consistent with
        the behaviour of the Python pathlib API, even though "/" is not directly usable
        as a collection on an iRODS system, being host to the iRODS zones.
        """
        if isinstance(self, DataObject):
            return [Collection(p) for p in PurePath(self.path, self.name).parents]

        return [Collection(p) for p in self.path.parents]

    def ancestor_metadata(self, timeout=None, tries=1) -> list[AVU]:
        """Return the metadata of the item's ancestors.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: All the AVUs of the item's ancestors.
        """
        avus = set()
        for coll in self.ancestors():
            avus.update(coll.metadata(timeout=timeout, tries=tries))
        return sorted(avus)

    @rods_type_check
    def metadata(
        self, attribute: Any = None, ancestors=False, timeout=None, tries=1
    ) -> list[AVU]:
        """Return the item's metadata.

        Args:
            attribute: Return only AVUs having this attribute. If this is not a string,
               the string representation of this argument is used.
            ancestors: Include metadata collated from ancestor collections.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: List[AVU]
        """
        if not self.connected():
            if ancestors:
                raise ValueError("Cannot retrieve ancestor metadata when disconnected")
            return sorted(self._local_metadata)

        item = self._list(avu=True, timeout=timeout, tries=tries).pop()
        if Baton.AVUS not in item:
            raise BatonError(f"{Baton.AVUS} key missing from {item}")

        avus = item[Baton.AVUS]
        if ancestors:
            avus.extend(self.ancestor_metadata(timeout=timeout, tries=tries))

        if attribute is not None:
            attr = str(attribute)
            avus = [avu for avu in avus if avu.attribute == attr]

        return sorted(avus)

    def collated_metadata(
        self, ancestors=False, timeout=None, tries=1
    ) -> dict[str, list[str]]:
        """Return a dictionary mapping AVU attributes to lists of corresponding AVU
        values.

        This method collates AVU values under their shared key. E.g. if an item had the
        AVUs AVU("Key1", "Value1") and AVU("Key1", "Value2"), the collated dictionary
        would be {"Key1": ["Value1", "Value2"]}.

        Args:
            ancestors: Include metadata collated from ancestor collections.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: Map of AVU attributes to lists of corresponding AVU values.
        """
        collated = defaultdict(list)
        for avu in self.metadata(ancestors=ancestors, timeout=timeout, tries=tries):
            collated[avu.attribute].append(avu.value)

        return collated

    def permissions(self, user_type: str = None, timeout=None, tries=1) -> list[AC]:
        """Return the item's Access Control List (ACL). Synonym for acl().

        Args:
            user_type: Filter to include only permissions for users of this type.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The ACL of the item.
        """
        return self.acl(user_type=user_type, timeout=timeout, tries=tries)

    @rods_type_check
    def acl(self, user_type: str = None, timeout=None, tries=1) -> list[AC]:
        """Return the item's Access Control List (ACL). Synonym for permissions().

        Args:
            user_type: Filter to include only permissions for users of this type.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The ACL of the item.
        """
        if user_type is not None and user_type not in [
            "rodsadmin",
            "rodsgroup",
            "rodsuser",
        ]:
            raise ValueError(f"Invalid user type requested: {type}")

        if not self.connected():
            return sorted(self._local_acl)

        item = self._list(acl=True, timeout=timeout, tries=tries).pop()
        if Baton.ACCESS not in item:
            raise BatonError(f"{Baton.ACCESS} key missing from {item}")

        # iRODS ACL queries can sometimes return multiple results where a user appears
        # to have both "own" and "read" permissions simultaneously. Since "own"
        # subsumes "read", this is confusing and can have unwanted effects, e.g. copying
        # permissions from one collection to another will effectively remove "own" if
        # the source collection has both "own" and "read", and the "read" permission
        # is copied after "own". Yes - when copying these permissions to another item,
        # iRODS now treats "own" and "read" are states that cannot be held
        # simultaneously and will delete the first when the second is applied.
        #
        # The source collection in the cases we observe is created with the iput
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
            acl = [ac for ac in acl if ac.user in by_name]

        return sorted(acl)

    def __lt__(self, other):
        if isinstance(self, Collection) and isinstance(other, DataObject):
            return True
        if isinstance(self, DataObject) and isinstance(other, Collection):
            return False

        return self.path < other.path

    @abstractmethod
    def rods_type(self) -> Type[RodsItem] | None:
        """Return a Python type representing the kind of iRODS path supplied."""
        pass

    @abstractmethod
    def check_rods_type(self, **kwargs):
        """Raise an error if the item does not have an appropriate type of iRODS path.
        e.g. raise an error if a Collection object has the path of a data object."""
        pass

    @abstractmethod
    def get(self, local_path: Path | str, **kwargs):
        """Get the item from iRODS."""
        pass

    @abstractmethod
    def put(self, local_path: Path | str, **kwargs):
        """Put the item into iRODS."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a minimal dictionary representation of the item."""
        pass

    @abstractmethod
    def to_json(self, **kwargs) -> str:
        """Return a JSON representation of the item, including metadata and permissions.
        Args:
            **kwargs: See json.dumps() for options. All keywords except 'cls' are
             passed through to a json.dumps() call by this method.
        Returns:
            A baton-format JSON string.
        """
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, json_str: str):
        """Load the item from a JSON string.

        Args:
            json_str: A baton-format JSON string.
        """
        pass

    @abstractmethod
    def _list(self, **kwargs):
        pass


class DataObject(RodsItem):
    """An iRODS data object.

    DataObject is a PathLike for the iRODS path it represents.
    """

    EMPTY_FILE_CHECKSUM = "d41d8cd98f00b204e9800998ecf8427e"

    @dataclass(frozen=True)
    class Version:
        """A record of a data object's state at a specific time."""

        checksum: str
        timestamp: datetime

        def __repr__(self):
            return f"({self.checksum}, {self.timestamp.isoformat(timespec='seconds')})"

    def __init__(
        self,
        remote_path: PurePath | str,
        local_path: Path | str = None,
        check_type=True,
        pool=default_pool,
    ):
        """DataObject constructor.

        Args:
            remote_path: A remote data object path.
            check_type: Check the remote path type if True, defaults to True.
            pool: A baton client pool. Optional.
        """
        lp = Path(local_path) if local_path else None
        rp = PurePath(remote_path)

        super().__init__(
            rp.parent,
            local_path=lp.parent if lp else None,
            check_type=check_type,
            pool=pool,
        )
        self.name = rp.name

        if lp is not None:
            self.dir = lp.parent
            self.file = lp.name

        self.versions = []

    @classmethod
    def query_metadata(
        cls,
        *avus: AVU,
        timestamps: list[Timestamp] = None,
        zone=None,
        timeout=None,
        tries=1,
        pool=default_pool,
    ) -> list[DataObject]:
        """Query data object metadata in iRODS.

        Args:
            *avus: One or more AVUs to query.
            timestamps: A list of Timestamp objects to narrow the search. Each timestamp
                value and operator is combined into the query (using AND logic, if there
                are more than one).
            zone: An iRODS zone hint. This can be the name of a zone to search or a path
                into a zone. If a path is used, results outside that collection will
                be removed from any results. If None, results from the current zone
                will be returned.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
            pool: Client pool to use. If omitted, the default pool is used.

        Returns: A list of data objects with matching metadata.
        """

        with client(pool) as c:
            items = c.query_metadata(
                avus=avus,
                timestamps=timestamps,
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

        if not self.connected():
            return None

        if self._rods_type is None:
            self._rods_type = rods_path_type(
                PurePath(self.path, self.name),
                timeout=RodsItem.INTERNAL_TIMEOUT,
                tries=RodsItem.INTERNAL_TRIES,
                pool=self._pool,
            )
        return self._rods_type

    def check_rods_type(self, **kwargs):
        """Raise an error if the path is not a data object in iRODS."""
        if not self.check_type:
            return

        rt = self.rods_type
        if rt is not None and rt != DataObject:
            raise BatonError(f"Invalid iRODS path type {rt} for a data object: {self}")

    @rods_type_check
    @connected
    def list(self, timeout=None, tries=1) -> DataObject:
        """Return a new DataObject representing this one.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: A new DataObject.
        """
        item = self._list(timeout=timeout, tries=tries).pop()
        return _make_rods_item(item, pool=self._pool)

    @rods_type_check
    @connected
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
        if calculate_checksum or recalculate_checksum or verify_checksum:
            item = self.to_dict()
            with client(self._pool) as c:
                return c.checksum(
                    item,
                    calculate_checksum=calculate_checksum,
                    recalculate_checksum=recalculate_checksum,
                    verify_checksum=verify_checksum,
                    timeout=timeout,
                    tries=tries,
                )
        else:
            item = self._list(checksum=True, timeout=timeout, tries=tries).pop()
            return item[Baton.CHECKSUM]

    @rods_type_check
    @connected
    def size(self, timeout=None, tries=1) -> int:
        """Return the size of the data object according to the iRODS IES database, in
        bytes.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The size of the data object in bytes.
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

    @connected
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

    @connected
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
    @connected
    def replicas(self, timeout=None, tries=1) -> list[Replica]:
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
        for rep in item[Baton.REPLICAS]:
            match rep:
                case {Baton.NUMBER: n}:
                    rep_args[n] = rep
                case _:
                    raise BatonError(f"{Baton.NUMBER} key missing from {rep}")

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

        for ts in item[Baton.TIMESTAMPS]:
            if Baton.REPLICAS not in ts:
                raise BatonError(f"{Baton.REPLICAS} key missing from {ts}")
            rep_num = ts[Baton.REPLICAS]

            match ts:
                case {Baton.CREATED: t}:
                    rep_args[rep_num][Baton.CREATED] = dateutil.parser.isoparse(t)
                case {Baton.MODIFIED: t}:
                    rep_args[rep_num][Baton.MODIFIED] = dateutil.parser.isoparse(t)
                case _:
                    raise BatonError(
                        f"{Baton.CREATED}/{Baton.MODIFIED} key missing " f"from {ts}"
                    )

        replicas = [Replica(**args) for args in rep_args.values()]
        replicas.sort()

        return replicas

    @rods_type_check
    @connected
    def get(
        self,
        local_path: Path | str,
        verify_checksum=False,
        local_checksum=None,
        compare_checksums=False,
        fill=False,
        force=True,
        redirect=False,
        timeout=None,
        tries=1,
    ):
        """Get the data object from iRODS and save to a local file.

        Args:
            local_path: The local path of a file to be created.
            verify_checksum: Verify the local checksum against the remote checksum.
            compare_checksums: Compare the local checksum to the remote checksum
                calculated by the iRODS server after the get operation. If the checksums
                do not match, raise an error. This is in addition to the comparison
                provided by the verify_checksum option. Defaults to False.
            local_checksum: A caller-supplied checksum of the local file. This may be a
                string, a path to a file containing a string, or a file path
                transformation function. If the latter, it must accept the local path as
                its only argument and return a string checksum. Typically, this is
                useful when this checksum is available from an earlier process that
                calculated it. Defaults to None.
            fill: Fill in a missing local file. If the local file already
                exists, the operation is skipped. That option may be combined with
                compare_checksums to ensure that the local file is up to date
                Defaults to False.
            force: Force overwrite any existing local file. Defaults to True.
            redirect: Redirect the operation to the best server, decided by iRODS
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
        """
        kwargs = {
            "verify_checksum": verify_checksum,
            "redirect": redirect,
            "timeout": timeout,
            "tries": tries,
        }

        glog = log.bind(path=self)

        def _compare_checksums(msg, _local: str, _remote: str, error=True) -> bool:
            match = _remote == _local
            if not match and error:
                raise ValueError(
                    f"Checksum mismatch for '{self}': {_remote} != {_local}"
                )
            glog.info(msg, local_checksum=_local, remote_checksum=_remote)
            return match

        local = Path(local_path)
        local_chk = None
        remote_chk = self.checksum()

        if compare_checksums:
            local_chk = _local_file_checksum(local, local_checksum)

        if local.exists():
            if fill:
                if compare_checksums:
                    if _compare_checksums(
                        "Local file already exists with matching checksum; skipping",
                        local_chk,
                        remote_chk,
                        error=False,
                    ):
                        return self.size()

                # Fill can force update mismatched objects
                self._get(local, force=True, **kwargs)
                if compare_checksums:
                    _compare_checksums(
                        "Comparing local and remote checksums", local_chk, remote_chk
                    )

                glog.info("Updated existing local file", remote_checksum=remote_chk)

                return self.size()

            if not force:
                # Note: this is implemented in Python because baton's `get` operation
                # by design always forces overwriting any existing data object.
                raise FileExistsError(
                    f"Local file already exists at '{self}' and force is False"
                )

        self._get(local, **kwargs)
        if compare_checksums:
            _compare_checksums(
                "Comparing local and remote checksums", local_chk, remote_chk
            )

        glog.info("Added new local file", remote_checksum=remote_chk)

        return self.size()

    @connected
    def put(
        self,
        local_path: Path | str,
        calculate_checksum=False,
        verify_checksum=False,
        local_checksum=None,
        compare_checksums=False,
        fill=False,
        force=True,
        redirect=False,
        timeout=None,
        tries=1,
    ) -> DataObject:
        """Put the data object into iRODS.

        If the put operation overwrites an existing data object, the previous version's
        checksum and timestamp are recorded in the versions attribute. Multiple versions
        are supported. Versions are not recorded if the data object is new. Version
        changes that occurred outside the lifetime of the DataObject instance are not
        retained.

        Args:
            local_path: The local path of a file to put into iRODS at the path
                specified by this data object.
            calculate_checksum: Calculate remote checksums for all replicas on the iRODS
                server after the put operation. If checksums exist, this is a no-op.
                Defaults to False.
            verify_checksum: Verify the local checksum calculated by the iRODS C API
                against the remote checksum calculated by the iRODS server for data
                objects. Defaults to False.
            local_checksum: A caller-supplied checksum of the local file. This may be a
                string, a path to a file containing a string, or a file path
                transformation function. If the latter, it must accept the local path as
                its only argument and return a string checksum. Typically, this is
                useful when this checksum is available from an earlier process that
                calculated it. Defaults to None.
            compare_checksums: Compare the local checksum to the remote checksum
                calculated by the iRODS server after the put operation. If the checksums
                do not match, raise an error. This is in addition to the comparison
                provided by the verify_checksum option. Defaults to False.
            fill: Fill in a missing data object in iRODS. If the data object already
                exists, the operation is skipped. That option may be combined with
                compare_checksums to ensure that the data object is up to date.
                Defaults to False.
            force: Overwrite any data object already present in iRODS. This option
                cannot be used with the fill option. Defaults to True.
            redirect: Redirect the operation to the best server, decided by iRODS.
                Defaults to False.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            The DataObject.
        """
        kwargs = {
            "calculate_checksum": calculate_checksum,
            "verify_checksum": verify_checksum,
            "redirect": redirect,
            "timeout": timeout,
            "tries": tries,
        }
        plog = log.bind(
            path=self, prev_version=self.versions[-1] if self.versions else None
        )

        def _compare_checksums(msg, _local: str, _remote: str, error=True) -> bool:
            match = _remote == _local
            if not match and error:
                raise ValueError(
                    f"Checksum mismatch for '{self}': {_remote} != {_local}"
                )
            plog.info(msg, local_checksum=_local, remote_checksum=_remote)
            return match

        local_chk = None

        if compare_checksums:
            local_chk = _local_file_checksum(local_path, local_checksum)

        if self.exists():
            remote_chk = self.checksum()

            if fill:
                if compare_checksums:
                    if _compare_checksums(
                        "Data object already exists with matching checksum; skipping",
                        local_chk,
                        remote_chk,
                        error=False,
                    ):
                        return self

                # Fill can force update mismatched objects
                self._put(local_path, force=True, **kwargs)
                remote_chk = self.checksum()

                if compare_checksums:
                    _compare_checksums(
                        "Comparing local and remote checksums", local_chk, remote_chk
                    )

                plog.info("Updated existing data object", remote_checksum=remote_chk)

                return self

            if not force:
                # Note: this is implemented in Python because baton's `put` operation
                # by design always forces overwriting any existing data object.
                raise FileExistsError(
                    f"Data object already exists at '{self}' and force is False"
                )

        self._put(local_path, **kwargs)
        remote_chk = self.checksum()
        if compare_checksums:
            _compare_checksums(
                "Comparing local and remote checksums", local_chk, remote_chk
            )

        plog.info("Added new data object", remote_checksum=remote_chk)

        return self

    @rods_type_check
    @connected
    def read(self, timeout=None, tries=1) -> str:
        """Read the data object from iRODS into a string. This operation is supported
        for data objects containing UTF-8 text.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: str
        """
        item = self.to_dict()
        with client(self._pool) as c:
            return c.read(item, timeout=timeout, tries=tries)

    @connected
    def trim_replicas(
        self, min_replicas=2, valid=False, invalid=True
    ) -> tuple[int, int]:
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

    def is_consistent_size(self, timeout=None, tries=1) -> bool:
        """Return true if the data object in iRODS is internally consistent.
        This is defined as:

        1. If the file is zero length, it has the checksum of an empty file.
        2. If the file is not zero length, it does not have the checksum of an empty file.

        In iRODS <= 4.2.8 it is possible for a data object to get into a bad state
        where it has zero length, but still reports as not stale and having the
        checksum of the full-length file.

        We can trigger this behaviour in iRODS by having more than one client uploading
        to a single path. iRODS <= 4.2.8 does not support any form of locking and allows
        uncoordinated writes to the filesystem. It does recognise this as a failure,
        but does not clean up the damaged file.

        This method looks for data object size and checksum consistency. It checks the
        values that iRODS reports for the whole data object; it does not check
        individual replicas.

        If the data object is absent, this method returns true as there can be no
        conflict where neither value exists.

        If the data object has no checksum, this method returns true as there is no
        evidence to dispute its reported size.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            True if the data object is internally consistent, False otherwise.
        """
        if not self.exists():
            return True

        chk = self.checksum(timeout=timeout, tries=tries)
        if chk is None:
            return True

        if self.size(timeout=timeout, tries=tries) == 0:
            return chk == DataObject.EMPTY_FILE_CHECKSUM

        return chk != DataObject.EMPTY_FILE_CHECKSUM

    def to_dict(self) -> dict:
        return {Baton.COLL: self.path.as_posix(), Baton.OBJ: self.name}

    def to_json(self, **kwargs) -> str:
        return json.dumps(self, cls=BatonJSONEncoder, **kwargs)

    @classmethod
    def from_json(cls, s: str):
        o = json.loads(s, object_hook=DISCONNECTED_JSON_DECODER)
        if isinstance(o, DataObject):
            return o

        raise ValueError(f"Expected a DataObject, got {type(o)}")

    def _get(
        self,
        local_path: Path | str,
        force=True,
        verify_checksum=False,
        redirect=False,
        timeout=None,
        tries=1,
    ) -> DataObject:
        item = self.to_dict()

        with client(self._pool) as c:
            c.get(
                item,
                Path(local_path),
                force=force,
                verify_checksum=verify_checksum,
                redirect=redirect,
                timeout=timeout,
                tries=tries,
            )

        return self

    def _put(
        self,
        local_path: Path | str,
        calculate_checksum=False,
        force=True,
        verify_checksum=False,
        redirect=False,
        timeout=None,
        tries=1,
    ) -> DataObject:
        item = self.to_dict()

        prev = None
        if self.exists():
            prev = DataObject.Version(self.checksum(), self.modified())

        with client(self._pool) as c:
            log.debug("Putting data object", path=self, client=c)
            c.put(
                item,
                Path(local_path),
                calculate_checksum=calculate_checksum,
                force=force,
                verify_checksum=verify_checksum,
                redirect=redirect,
                timeout=timeout,
                tries=tries,
            )

        if prev is not None and self.checksum() != prev.checksum:
            self.versions.append(prev)

        return self

    def _list(self, **kwargs) -> list[dict]:
        item = self.to_dict()
        with client(self._pool) as c:
            return c.list(item, **kwargs)

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

    DEFAULT_FILTER = lambda _: False

    @classmethod
    def query_metadata(
        cls,
        *avus: AVU,
        timestamps: list[Timestamp] | None = None,
        zone=None,
        timeout=None,
        tries=1,
        pool=default_pool,
    ) -> list[Collection]:
        """Query collection metadata in iRODS.

        Args:
            *avus: AVUs to query.
            timestamps: A list of Timestamp objects to narrow the search. Each timestamp
                value and operator is combined into the query (using AND logic, if there
                are more than one).
            zone: An iRODS zone hint. This can be the name of a zone to search or a path
                into a zone. If a path is used, results outside that collection will
                be removed from any results. If None, results from the current zone
                will be returned.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.
            pool: Client pool to use. If omitted, the default pool is used.

        Returns: A list of collections with matching metadata.
        """
        with client(pool) as c:
            items = c.query_metadata(
                avus=avus,
                timestamps=timestamps,
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
        self,
        remote_path: PurePath | str,
        local_path: Path | str = None,
        check_type=True,
        pool=default_pool,
    ):
        """Collection constructor.

        Args:
            remote_path: A remote collection path.
            check_type: Check the remote path type if True, defaults to True.
            pool: A baton client pool. Optional.
        """
        super().__init__(
            remote_path, local_path=local_path, check_type=check_type, pool=pool
        )

    def create(
        self, parents=False, exist_ok=False, timeout=None, tries=1
    ) -> Collection:
        """Create a new, empty Collection on the server side.

        Args:
            parents: Create parent collections as necessary.
            exist_ok: If the collection exists, do not raise an error.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            The Collection.
        """
        if exist_ok and self.exists():
            return self

        item = self.to_dict()
        with client(self._pool) as c:
            c.create_collection(item, parents=parents, timeout=timeout, tries=tries)
        return self

    @property
    def rods_type(self):
        """Return a Python type representing the kind of iRODS path supplied."""
        if not self.connected():
            return None

        if self._rods_type is None and self.connected():
            self._rods_type = rods_path_type(
                self.path,
                timeout=RodsItem.INTERNAL_TIMEOUT,
                tries=RodsItem.INTERNAL_TRIES,
                pool=self._pool,
            )
        return self._rods_type

    def check_rods_type(self, **kwargs):
        """Raise an error if the path is not a collection in iRODS."""
        if not self.check_type:
            return

        rt = self.rods_type
        if rt is not None and rt != Collection:
            raise BatonError(f"Invalid iRODS path type {rt} for a collection: {self}")

    @rods_type_check
    @connected
    def contents(
        self, acl=False, avu=False, recurse=False, timeout=None, tries=1
    ) -> list[Collection | DataObject]:
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

        contents = [_make_rods_item(item, pool=self._pool) for item in items]

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
    @connected
    def iter_contents(
        self, acl=False, avu=False, recurse=False, timeout=None, tries=1
    ) -> Iterable[Collection | DataObject]:
        """Return a generator for the Collection contents.

        Args:
          acl: Include ACL information.
          avu: Include AVU (metadata) information.
          recurse: Recurse into sub-collections in depth-first order.
          timeout: Operation timeout in seconds.
          tries: Number of times to try the operation.

        Returns: Iterable[Collection | DataObject]"""
        items = self._list(
            acl=acl,
            avu=avu,
            contents=True,
            timeout=timeout,
            tries=tries,
        )

        contents = [_make_rods_item(item, pool=self._pool) for item in items]
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
    @connected
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
        return _make_rods_item(item, pool=self._pool)

    def timestamp(self, timeout=None, tries=1) -> datetime:
        """Return the timestamp of the collection according to the iRODS IES database.
        This is a synonym of the `modified` method.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            The collection's modification timestamp.
        """
        return self.modified(timeout=timeout, tries=tries)

    @connected
    def created(self, timeout=None, tries=1) -> datetime:
        """Return the creation timestamp of the collection according to the
        iRODS IES database.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            The collection's creation timestamp.
        """

        item = self._list(timestamp=True, timeout=timeout, tries=tries).pop()
        if Baton.TIMESTAMPS not in item:
            raise BatonError(f"{Baton.TIMESTAMPS} key missing from '{item}'")

        for ts in item[Baton.TIMESTAMPS]:
            match ts:
                case {Baton.CREATED: t}:
                    return dateutil.parser.isoparse(t)
                case _:
                    continue

        raise BatonError(f"{Baton.CREATED} key missing from '{item}'")

    @connected
    def modified(self, timeout=None, tries=1) -> datetime:
        """Return the modification timestamp of the collection according to the
        iRODS IES database.

        Args:
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            The collection's modification timestamp.
        """

        item = self._list(timestamp=True, timeout=timeout, tries=tries).pop()
        if Baton.TIMESTAMPS not in item:
            raise BatonError(f"{Baton.TIMESTAMPS} key missing from '{item}'")

        for ts in item[Baton.TIMESTAMPS]:
            match ts:
                case {Baton.MODIFIED: t}:
                    return dateutil.parser.isoparse(t)
                case _:
                    continue

        raise BatonError(f"{Baton.MODIFIED} key missing from '{item}'")

    @rods_type_check
    @connected
    def get(
        self,
        local_path: Path | str,
        recurse=False,
        verify_checksum=False,
        filter_fn: callable[[any], bool] = DEFAULT_FILTER,
        fill=False,
        force=True,
        redirect=False,
        yield_exceptions=False,
        timeout: float | None = None,
        tries: int = 1,
    ) -> Generator[Collection | DataObject | Exception, Any, None]:
        """
        Fetches a remote collection or data object to a specified local path from the server.

        This method retrieves collections or data objects from the server to a specified local
        directory, handling recursive operations, checksum verification, and applying optional
        filters. It supports concurrent operations using threads while also handling exceptions
        gracefully if specified.

        Args:
            local_path: The local file system path where the remote data will be
                downloaded.
            recurse: Get the contents of sub-collections recursively. Defaults to False.
            verify_checksum: Verify checksums of data at rest after transfer. Defaults
                to False.
            filter_fn:  A predicate accepting a single RodsItem argument to which
                each remote path (collections and data objects) will be passed before
                getting from into iRODS. If the predicate returns True, the path will be
                filtered i.e. not be got from iRODS. Filtering collections will result
                in them being pruned. Filtering data objects will result in them being
                skipped.
            fill: Fill in missing local filesS. If the local file already exists,
                the operation is skipped. See DataObject.get() for more information.
                Defaults to False.
            force: Force overwriting of existing local files. Defaults to True.
            redirect: Redirect the operation to the best server, decided by iRODS.
                Defaults to False.
            yield_exceptions:  If True, yield exceptions instead of raising them.
                Defaults to False.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            A generator over the downloaded local directories and files.
        """

        def _handle_exception(ex):
            if yield_exceptions:
                yield ex
            else:
                raise ex

        def _batch_get(pairs):
            def _get_obj(_remote: PurePath, _local: Path):
                start = time.monotonic()
                log.debug(
                    "Started data object",
                    remote_path=_remote.as_posix(),
                    local_path=_local.as_posix(),
                    start=start,
                )

                obj = DataObject(_remote, pool=self._pool)
                num_bytes = obj.get(
                    _local,
                    fill=fill,
                    force=force,
                    redirect=redirect,
                    verify_checksum=verify_checksum,
                    timeout=timeout,
                    tries=tries,
                )

                end = time.monotonic()
                log.debug(
                    "Finished data object",
                    remote_path=_remote.as_posix(),
                    local_path=_local.as_posix(),
                    end=end,
                    duration=end - start,
                    num_bytes=num_bytes,
                )
                return obj

            with ThreadPoolExecutor(
                max_workers=self._pool.maxsize, thread_name_prefix="coll-get"
            ) as executor:
                futures = [executor.submit(_get_obj, r, l) for r, l in pairs]
                for future in as_completed(futures):
                    try:
                        yield future.result()
                    except Exception as e:
                        yield from _handle_exception(e)

        local_root = Path(local_path)

        try:
            local_root = local_root.resolve(strict=True)
            if not local_root.is_dir():
                local_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            yield from _handle_exception(e)

        stack = [self]
        while stack:
            coll = stack.pop(0)

            try:
                d = local_root / coll.path.relative_to(self.path)
                d.mkdir(exist_ok=True)
                yield coll
            except Exception as e:
                yield from _handle_exception(e)

            path_pairs = []
            for item in coll.contents(timeout=timeout, tries=tries):
                if filter_fn(item):
                    continue

                if isinstance(item, Collection):
                    if recurse:
                        stack.append(item)
                else:
                    remote = PurePath(item.path, item.name)
                    local = local_root / item.path.relative_to(self.path) / item.name
                    path_pairs.append((remote, local))

            yield from _batch_get(path_pairs)

    @connected
    def put(
        self,
        local_path: Path | str,
        recurse=False,
        calculate_checksum=False,
        verify_checksum=False,
        local_checksum=None,
        compare_checksums=False,
        fill=False,
        filter_fn: callable[[any], bool] = DEFAULT_FILTER,
        force=True,
        redirect=False,
        yield_exceptions=False,
        timeout: float | None = None,
        tries: int = 1,
    ) -> Generator[Collection | DataObject | Exception, Any, None]:
        """Put the collection into iRODS.

        The returned generator yields the collection and contents as they are created.
        Data objects within each collection are put in parallel using a thread
        pool. The order of the yielded items is not guaranteed to be the same as the
        order of the items in the collection.

        The generator has two modes of error handling. The first (and default) is to
        raise an exception as soon as an error occurs, terminating the generator. The
        second is to catch the exception and yield it in place of the affected item.
        This allows the caller to decide how to handle the error (e.g. by skipping the
        item). The generator will continue to yield items (and/or further exceptions)
        until it has finished processing the collection. To enable this behaviour, set
        the `yield_exceptions` argument to True.

        Args:
            local_path: The local path of a directory to put into iRODS at the path
                specified by this collection.
            recurse: Recurse through subdirectories.
            filter_fn: A predicate accepting a single pathlib.Path argument to which
                each local path (directories and files) will be passed before putting
                into iRODS. If the predicate returns True, the path will be filtered
                i.e. not be put into iRODS. Filtering directories will result in them
                being pruned. Filtering files will result in them being skipped.
            calculate_checksum: Calculate remote checksums for all data object replicas.
                See DataObject.put() for more information. Defaults to False.
            verify_checksum: Verify the local checksum calculated by the iRODS C API
                against the remote checksum calculated by the iRODS server for data
                objects. See DataObject.put() for more information. Defaults to False.
            local_checksum: A callable that returns a checksum for a local file. See
                DataObject.put() for more information. This is called for each file in
                encountered while recursing, with the file path as its argument.
                (Also accepts a string or a path to a file containing a string, as does
                DataObject.put(), however, this is not useful for collections except in
                the edge where all the files have identical contents). Defaults to None.
            compare_checksums: Compare caller-supplied local checksums to the remote
                checksums calculated by the iRODS server after the put operation for
                data objects. If the checksums do not match, raise an error. See
                DataObject.put() for more information. Defaults to False.
            fill: Fill in missing data objects in iRODS. If the data object already
                exists, the operation is skipped. See DataObject.put() for more
                information. Defaults to False.
            force: Overwrite any data objects already present in iRODS. Defaults to
                True.
            redirect: Redirect the operation to the best server, decided by iRODS.
                Defaults to False.
            yield_exceptions: If True, yield exceptions instead of raising them.
                Defaults to False.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns:
            A generator over the collection and contents.
        """

        def _handle_exception(ex):
            if yield_exceptions:
                yield ex
            else:
                raise ex

        def _batch_put(pairs):
            def _put_obj(_local: Path, _remote: PurePath):
                start = time.monotonic()
                log.debug(
                    "Started data object",
                    local_path=_local.as_posix(),
                    remote_path=_remote.as_posix(),
                    start=start,
                )

                obj = DataObject(_remote, pool=self._pool).put(
                    _local,
                    calculate_checksum=calculate_checksum,
                    verify_checksum=verify_checksum,
                    local_checksum=local_checksum,
                    compare_checksums=compare_checksums,
                    fill=fill,
                    force=force,
                    redirect=redirect,
                    timeout=timeout,
                    tries=tries,
                )
                end = time.monotonic()
                log.debug(
                    "Completed data object",
                    local_path=_local.as_posix(),
                    remote_path=_remote.as_posix(),
                    start=start,
                    end=end,
                    duration=end - start,
                )
                return obj

            with ThreadPoolExecutor(
                max_workers=self._pool.maxsize, thread_name_prefix="coll-put"
            ) as executor:
                futures = [executor.submit(_put_obj, l, r) for l, r in pairs]

                for future in as_completed(futures):
                    try:
                        yield future.result()
                    except Exception as e:
                        yield from _handle_exception(e)

        try:
            if not Path(local_path).resolve(strict=True).is_dir():
                raise ValueError(f"Local path '{local_path}' is not a directory")
        except Exception as e:
            yield from _handle_exception(e)
            return

        try:
            yield self.create(exist_ok=True, timeout=timeout, tries=tries)
        except Exception as e:
            yield from _handle_exception(e)

        if recurse:
            for dirpath, dirnames, filenames in os.walk(local_path, topdown=True):
                # As topdown is True, we can sort dirnames in-place to get a predictable
                # walk order
                dirnames.sort()
                filenames.sort()

                # As topdown is True, we can prune the walk by removing from dirnames
                # in-place. N.B that we iterate over a shallow copy of dirnames.
                for d in dirnames[:]:
                    try:
                        local = Path(dirpath, d)
                        if filter_fn(local):
                            dirnames.remove(d)
                            continue

                        remote = PurePath(self.path, local.relative_to(local_path))
                        yield Collection(remote, pool=self._pool).create(
                            exist_ok=True, timeout=timeout
                        )
                    except Exception as e:
                        yield from _handle_exception(e)

                path_pairs = []
                for f in filenames:
                    local = Path(dirpath, f)
                    if filter_fn(local):
                        continue

                    remote = PurePath(self.path, local.relative_to(local_path))
                    path_pairs.append((local, remote))

                yield from _batch_put(path_pairs)
        else:
            dirs, files = [], []
            try:
                for local in Path(local_path).iterdir():
                    dirs.append(local) if local.is_dir() else files.append(local)
            except Exception as e:
                yield from _handle_exception(e)
                return

            for d in sorted(dirs):
                if filter_fn(d):
                    continue

                remote = PurePath(self.path, d.relative_to(local_path))
                try:
                    yield Collection(remote, pool=self._pool).create(
                        exist_ok=True, timeout=timeout
                    )
                except Exception as e:
                    yield from _handle_exception(e)

            path_pairs = []
            for local in sorted(files):
                if filter_fn(local):
                    continue

                remote = PurePath(self.path, local.name)
                path_pairs.append((local, remote))

            yield from _batch_put(path_pairs)

    def add_permissions(
        self,
        *acs: AC,
        recurse=False,
        filter_fn: callable[any, bool] = DEFAULT_FILTER,
        timeout: float | None = None,
        tries: int = 1,
    ) -> int:
        """Add access controls to the collection. Return the number of access
        controls added. If some argument access controls are already present,
        those arguments will be ignored.

        Args:
            *acs: Access controls.
            recurse: Recursively add access controls.
            filter_fn: A predicate accepting a single RodsItem argument to which each
                iRODS path will be passed during recursive operations, before adding
                permissions. If the predicate returns True, the path will be filtered
                i.e. not have permissions added.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The number of access controls added.
        """
        num_added = super().add_permissions(*acs, timeout=timeout, tries=tries)

        if recurse:

            def do_add(it) -> int:
                if filter_fn(it):
                    log.debug("Skipping permissions add", path=it, acl=acs)
                    return 0
                return it.add_permissions(*acs, timeout=timeout, tries=tries)

            with ThreadPoolExecutor(
                max_workers=self._pool.maxsize,
                thread_name_prefix="coll-add-permissions",
            ) as executor:
                num_added += sum(
                    executor.map(do_add, self.iter_contents(recurse=recurse))
                )

        return num_added

    def remove_permissions(
        self,
        *acs: AC,
        recurse=False,
        filter_fn: callable[any, bool] = DEFAULT_FILTER,
        timeout: float | None = None,
        tries: int = 1,
    ) -> int:
        """Remove access controls from the collection. Return the number of access
        controls removed. If some argument access controls are not present, those
        arguments will be ignored.

        Args:
            *acs: Access controls.
            recurse: Recursively remove access controls.
            filter_fn: A predicate accepting a single RodsItem argument to which each
                iRODS path will be passed during recursive operations, before removing
                permissions. If the predicate returns True, the path will be filtered
                i.e. not have permissions removed.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The number of access controls removed.
        """
        num_removed = super().remove_permissions(*acs, timeout=timeout, tries=tries)
        if recurse:

            def do_remove(it) -> int:
                if filter_fn(it):
                    log.debug("Skipping permissions remove", path=it, acl=acs)
                    return 0
                return it.remove_permissions(*acs, timeout=timeout, tries=tries)

            with ThreadPoolExecutor(
                max_workers=self._pool.maxsize,
                thread_name_prefix="coll-rem-permissions",
            ) as executor:
                num_removed += sum(
                    executor.map(do_remove, self.iter_contents(recurse=recurse))
                )

        return num_removed

    def supersede_permissions(
        self,
        *acs: AC,
        recurse=False,
        filter_fn: callable[any, bool] = DEFAULT_FILTER,
        timeout: float | None = None,
        tries: int = 1,
    ) -> tuple[int, int]:
        """Remove all access controls from the collection, replacing them with the
        specified access controls. Return the numbers of access controls
        removed and added.

        Args:
            *acs: Access controls.
            recurse: Recursively supersede access controls.
            filter_fn: A predicate accepting a single RodsItem argument to which each
                iRODS path will be passed during recursive operations, before
                superseding permissions. If the predicate returns True, the path will be
                filtered i.e. not have permissions superseded.
            timeout: Operation timeout in seconds.
            tries: Number of times to try the operation.

        Returns: The number of access controls removed and added.
        """
        num_removed, num_added = super().supersede_permissions(
            *acs, timeout=timeout, tries=tries
        )

        if recurse:

            def do_supersede(it) -> tuple[int, int]:
                if filter_fn(it):
                    log.debug("Skipping permissions supersede", path=it)
                    return 0, 0
                return it.supersede_permissions(*acs, timeout=timeout, tries=tries)

            with ThreadPoolExecutor(
                max_workers=self._pool.maxsize,
                thread_name_prefix="coll-super-permissions",
            ) as executor:
                for nr, na in executor.map(
                    do_supersede, self.iter_contents(recurse=recurse)
                ):
                    num_removed += nr
                    num_added += na

        return num_removed, num_added

    def to_dict(self) -> dict:
        return {Baton.COLL: self.path.as_posix()}

    def to_json(self, **kwargs) -> str:
        return json.dumps(self, cls=BatonJSONEncoder, **kwargs)

    def _list(self, **kwargs) -> list[dict]:
        with client(self._pool) as c:
            return c.list(self.to_dict(), **kwargs)

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

    @classmethod
    def from_json(cls, s: str):
        o = json.loads(s, object_hook=DISCONNECTED_JSON_DECODER)
        if isinstance(o, Collection):
            return o

        raise ValueError(f"Expected a Collection, got {type(o)}")


class BatonJSONEncoder(json.JSONEncoder):
    """Encoder for baton JSON.

    This encoder is general-purpose. It is used to serialise Python objects before
    passing them to the baton client."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Collection):
            enc = {
                Baton.COLL: o.path,
                Baton.AVUS: o.metadata(),
                Baton.ACCESS: o.acl(),
            }
            if o.local_path is not None:
                enc[Baton.DIR] = o.local_path

            return enc

        if isinstance(o, DataObject):
            enc = {
                Baton.COLL: o.path,
                Baton.OBJ: o.name,
                Baton.AVUS: o.metadata(),
                Baton.ACCESS: o.acl(),
            }
            if o.local_path is not None:
                enc[Baton.DIR] = o.dir
                enc[Baton.FILE] = o.file

            if o.connected():
                enc[Baton.SIZE] = o.size()
                enc[Baton.CHECKSUM] = o.checksum()

            return enc

        if isinstance(o, AVU):
            enc = {Baton.ATTRIBUTE: o.attribute, Baton.VALUE: o.value}
            if o.units:
                enc[Baton.UNITS] = o.units
            if o.operator and o.operator != "=":
                enc[Baton.OPERATOR] = o.operator

            return enc

        if isinstance(o, Timestamp):
            enc = {Baton.OPERATOR: o.operator}

            dt = o.value
            if dt.tzinfo:
                dt = dt.astimezone(timezone.utc)
            ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            match o.event:
                case Timestamp.Event.CREATED:
                    enc[Baton.CREATED] = ts
                case Timestamp.Event.MODIFIED:
                    enc[Baton.MODIFIED] = ts
            return enc

        if isinstance(o, Permission):
            return o.name.lower()

        if isinstance(o, AC):
            return {
                Baton.OWNER: o.user,
                Baton.ZONE: o.zone,
                Baton.LEVEL: o.perm,
            }

        if isinstance(o, Path | PurePath):
            return o.as_posix()

        return super().default(o)


def _make_decoder_hook(pool: BatonPool | None = default_pool):
    def hook(item: dict) -> Any:
        def _populate(x):
            if Baton.AVUS in item:
                x.add_metadata(*item[Baton.AVUS])
            if Baton.ACCESS in item:
                x.add_permissions(*item[Baton.ACCESS])
            return x

        match item:
            case {Baton.COLL: c, Baton.OBJ: o}:
                return _populate(DataObject(PurePath(c, o), pool=pool))
            case {Baton.COLL: c}:
                return _populate(Collection(PurePath(c), pool=pool))
            case {Baton.ATTRIBUTE: attr, Baton.VALUE: value, Baton.UNITS: units}:
                return AVU(attr, value, units)
            case {Baton.ATTRIBUTE: attr, Baton.VALUE: value}:
                return AVU(attr, value)
            case {Baton.OWNER: user, Baton.ZONE: zone, Baton.LEVEL: level}:
                return AC(user, Permission[level.upper()], zone=zone)
            case {Baton.OWNER: user, Baton.LEVEL: level}:
                return AC(user, Permission[level.upper()])
            case _:
                raise BatonError(f"Failed to decode '{item}'")

    return hook


"""JSON decoder for connected baton objects."""
CONNECTED_JSON_DECODER = _make_decoder_hook(pool=default_pool)

"""JSON decoder for disconnected baton objects."""
DISCONNECTED_JSON_DECODER = _make_decoder_hook(pool=None)


def _make_rods_item(item: dict, pool: BatonPool) -> Collection | DataObject:
    """Create a new Collection or DataObject as appropriate for a dictionary
    returned by a Baton.

    Returns: Collection | DataObject
    """
    match item:
        case {Baton.COLL: c, Baton.OBJ: o}:
            return DataObject(PurePath(c, o), pool=pool)
        case {Baton.COLL: c}:
            return Collection(PurePath(c), pool=pool)
        case _:
            raise BatonError(f"{Baton.COLL} key missing from '{item}'")


def _calculate_file_checksum(path: Path | str) -> str:
    """Calculate the MD5 checksum of a local file.

    Args:
        path: A local file path.

    Returns: The checksum of the file.
    """
    # Can swap out for hashlib.file_digest (Python 3.11) if/when stop supporting Python 3.10
    # External md5sum binary not reliably available
    h = hashlib.md5()
    chunk_size = 2**20  # 1MB
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _local_file_checksum(path: Path | str, checksum_source) -> str:
    if checksum_source is None:
        checksum = _calculate_file_checksum(path)
        log.info(
            "Calculated checksum from local file data",
            local_checksum=checksum,
            path=path,
        )
        return checksum

    if callable(checksum_source):
        checksum = checksum_source(path)
        log.info(
            "Obtained checksum from supplied callable",
            local_checksum=checksum,
            path=path,
        )
        return checksum

    if isinstance(checksum_source, os.PathLike):
        with open(checksum_source, "r") as f:
            checksum = f.read()
            log.info(
                "Read pre-calculated checksum from a local file",
                local_checksum=checksum,
                path=path,
            )
            return checksum

    if isinstance(checksum_source, str):
        checksum = checksum_source
        log.info("Using provided checksum string", local_checksum=checksum, path=path)
        return checksum

    raise ValueError(
        f"Invalid type for local_checksum: '{type(checksum_source)}' "
        f"for '{path}'; must be a string, a path of a file containing a"
        f" string, or a callable taking a path of a file and "
        f"returning a string"
    )
