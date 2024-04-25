# partisan

[![Unit tests](https://github.com/wtsi-npg/partisan/actions/workflows/run-tests.yml/badge.svg)](https://github.com/wtsi-npg/partisan/actions/workflows/run-tests.yml)

    partisan |ˈpɑː.tɪˌzæn|

    noun, A long-handled spear with a triangular, double-edged blade and 
          lateral projections.

    adjective, Biased in support of a party, group, or cause.

## Summary

`partisan` is a Python client API for iRODS using the
[baton](https://github.com/wtsi-npg/baton) iRODS client. It is an alternative
to
[python-irodsclient](https://github.com/irods/python-irodsclient), biased
towards the needs of the [NPG](https://github.com/wtsi-npg) team.

### Comparison with [python-irodsclient](https://github.com/irods/python-irodsclient)

`partisan` uses [baton](https://github.com/wtsi-npg/baton), a programming
language-agnostic iRODS client with a JSON interface. This means that
`partisan`

- Has guaranteed compatibility with released iRODS versions. `partisan` uses
  the official iRODS C API (via `baton`).

- Supports improved speed for data object `put` operations. `python-irodsclient`
  in our hands is 3-4x slower than iRODS C API `put` for multi-GB files.

- Supports federation fully and transparently. `python-irodsclient` is not yet
  able to do this [for collections](https://github.com/irods/python-irodsclient/issues/173)

- Provides support for timeouts and retries, which `python-irodsclient` does
  not.

- Provides a simpler programming interface than `python-irodsclient`.
  `partisan`'s `Collection` and `DataObject` classes are created from path
  strings, while iRODS connections are managed transparently by the API.

- Offers consistency of behaviour and a shared surface area for bugs with our
  Perl and Go-based pipelines, which also use `baton`.

### Roadmap to `python-irodsclient`

Both `partisan` and `baton` are tightly focussed on the iRODS features that we
use, while `python-irodsclient` provides much broader functionality. Transition
to `python-irodsclient` may become desirable because of this. In that the
case `partisan` will retain its API and will switch to an
`python-irodsclient` backend, so that applications written against it will
continue to function without change.

## API

The entry points for `partisan`'s API are the `Collection` and `DataObject`
classes, which are constructed from an absolute iRODS path, either as a string
or a `pathlib.PurePath`.

`partisan` uses the `baton` client which must available to Python. Normally
this is achieved by making sure the `baton-do` executable is on your
`PATH`.

Like the iRODS client `icommands`, `baton` uses the environment variable
`IRODS_ENVIRONMENT_FILE` to locate an iRODS configuration file. You should
ensure that this set before attempting to connect to iRODS.

These are the examples use some dummy data from the package data directory.
The assertions are correct for a test system where the zone is named
`testZone`, the user is `irods`, the iRODS resource is a replication resource
with 2 replicas, and the server is configured to use MD5 checksums. A 
runnable copy of these examples is present in `tests/test_examples.py`

Given some collections and data objects already in iRODS ...

### Collections

To make an object representing a collection, pass a string or `os.PathLike`
to the `Collection` constructor. iRODS paths are virtual, existing only in
the iRODS IES database. A `Collection` is a Python `os.PathLike` object.

     coll = Collection(ont_gridion)
     assert coll.exists(), "The collection exists"

 We can examine the collection permissions. Note that you may get a
 different result on your iRODS, depending on your user and zone.
 
    assert coll.permissions() == [AC("irods", Permission.OWN, zone="testZone")]

We can list the collection, which returns a new instance.

    assert coll.list() == coll

We can examine the collection's immediate contents.

        assert coll.contents() == [
            Collection(ont_gridion / "66")
        ], "The collection contains one sub-collection"

We can examine the collection's contents recursively. If you print the
contents, you'll see that collections sort before data objects.

        assert len(coll.contents(recurse=True)) == 26

We can examine the collection's metadata.

        assert coll.metadata() == [], "The collection has no metadata"

We can examine the collection's permissions. Note that you may get a
different result on your iRODS, depending on your user and zone.

        assert coll.permissions() == [AC("irods", Permission.OWN, zone="testZone")]

### DataObjects

To make an object representing a data object, pass a string or `os.PathLike`
to the `DataObject` constructor. iRODS paths are virtual, existing only in
the iRODS catalog database. A `DataObject` is a Python `os.PathLike` object.

        obj = DataObject(
            ont_gridion
            / "66"
            / "DN585561I_A1"
            / "20190904_1514_GA20000_FAL01979_43578c8f"
            / "report.md"
        )
        assert obj.exists(), "The data object exists"

We can examine the data object's permissions. Note that you may get a
different result on your iRODS, depending on your user and zone.

        assert obj.permissions() == [AC("irods", Permission.OWN, zone="testZone")]

We can compare the remote object's checksum with its expected value.

        assert obj.checksum() == "c462fb84625c26ba15ecdf62e15a9560"

We can list the object, which returns a new instance.

        assert obj.list() == obj

We can examine any replicas of the object. Note that you may get a
different result on your iRODS, depending on whether you are using
replication, or not.

        assert len(obj.replicas()) == 2
        for r in obj.replicas():
            assert r.checksum == "c462fb84625c26ba15ecdf62e15a9560"
            assert r.valid

We can see the expected size, according to the iRODS IES database, in bytes.

        assert obj.size() == 1369

We can get the data object to a local file and confirm that the data within
is the expected size and matches the expected checksum.

        with tempfile.TemporaryDirectory() as d:
            local_path = Path(d, obj.name)
            obj.get(local_path)

            with open(local_path, "rb") as f:
                m = hashlib.md5()
                n = 0
                for b in iter(lambda: f.read(1024), b""):
                    n += len(b)
                    m.update(b)

                assert n == 1369
                assert m.hexdigest() == "c462fb84625c26ba15ecdf62e15a9560"


For small text files, we can also read a data object directly into memory.

        assert (
            hashlib.md5(obj.read().encode("utf-8")).hexdigest()
            == "c462fb84625c26ba15ecdf62e15a9560"
        )

We can examine the data object's metadata.

        assert obj.metadata() == [], "The data object has no metadata"

We can add some metadata to the data object. Let's add two `AVU`s (Attribute,
Value, Unit tuples).

        obj.add_metadata(
            AVU("sample_id", "sample_1"), AVU("experiment_id", "experiment_1")
        )

Note that metadata returned by this method are sorted by `AVU` attribute and
value.

        assert obj.metadata() == [
            AVU("experiment_id", "experiment_1"),
            AVU("sample_id", "sample_1"),
        ]

Now that there are metadata on the data object, we can find it by query.
Here we query on experiment_id alone.

        assert DataObject.query_metadata(AVU("experiment_id", "experiment_1")) == [obj]

Note that in iRODS, collection and data object metadata are separate and
must be queried independently. In the `partisan` API, a query using
`DataObject.query_metadata()` searches only data object metadata, while
`Collection.query_metadata()` searches only collection metadata.

        assert Collection.query_metadata(AVU("experiment_id", "experiment_1")) == []

The function `partisan.irods.query_metadata()` is available to search both
collection and data object metadata to return combined results.

        assert irods.query_metadata(AVU("experiment_id", "experiment_1")) == [obj]

The default query operator is `=`. You can specify a different operator for each
AVU used to define a query by setting its `operator` attribute in any of the query
methods.

        assert DataObject.query_metadata(AVU("experiment_id", "experiment%", operator="like")) == [obj]

        assert irods.query_metadata(AVU("experiment_id", "experiment%", operator="like")) == [obj]

### Pools

`partisan` uses a small pool (the default is 4) of `BatonClient` instances to
serve requests. This pool is created automatically and is passed to the
constructors of `Collection`s and `DataObject`s by default. If you would like
an alternative client management strategy, you can do this by creating your
own pool.

Let's create a `Collection` using the default pool.

        coll = Collection(ont_gridion)

Now let's create a `Collection` using a pool which contains only one
`BatonClient`. `partisan` provides a context manager to help with this. When a
pool goes out of scope it will automatically be closed, stopping any
`BatonClient`s within it and terminating any `baton-do` processes they may
have started.

        with client_pool(maxsize=1) as pool:
            coll = Collection(ont_gridion, pool=pool)

### Timeouts

In its API methods that communicate with the iRODS server, `partisan`
provides the two keyword arguments `timeout` and `tries` to control the
timeout for the operation (in seconds) and how many attempts should be made
to carry it out before raising an exception. The default values for 
these are `None` (meaning do not time out) and `1` (meaning make a single
attempt).

Let's check if a collection exists, timing out if a response takes longer
than 10 seconds and allowing 3 tries at that (each timing out after 10 
seconds).

        Collection(ont_gridion).exists(timeout=10, tries=3)

A timeout may occur for one of two reasons. Firstly, if no `BatonClient` is
available before the timeout expires and secondly, if the iRODS server takes 
longer than the timeout to respond with a result. partisan does not
distinguish between these circumstances.

## Requirements

- The `baton-do` executable from the [baton](https://github.com/wtsi-npg/baton)
  iRODS client distribution.
- The unit tests use the
  [iRODS client icommands](https://github.com/irods/irods_client_icommands)
  clients. These are not required during normal operation.

These tools should be present on the `PATH`, when required.

## Testing with Docker

An iRODS server and clients are available as Docker images which may be used
with Docker Compose to set up a standard test environment. The test 
environment consists of an `irods-server` container and an `irods-clients` 
container.

Before running the tests, start the containers and supporting network:

```commandline
    docker-compose up -d
```

The environment variables `IRODS_VERSION` (defaults to `4.2.11`) and 
`DOCKER_TAG` (defaults to `latest`) may be used to choose particular 
Docker images.


```commandline
    IRODS_VERSION="4.2.11" DOCKER_TAG="latest" docker-compose up -d
```

The `./tests/bin` directory contains a universal iRODS proxy script to be used
instead of native iRODS clients. It forwards any client operations to the 
real iRODS clients inside the `irods-clients` container. This directory 
should be on your `PATH` while running the tests. The iRODS authentication 
file can then be created using `iinit`: 

```commandline
    export PATH="${PWD}/tests/bin:$PATH"
    iinit
```

The tests should be run in the root of the repository, with `tmp` redirected 
to a destination in a shared volume:

```commandline
    pytest --basetemp=./tests/tmp
```

Finally, to destroy the test containers and network:

````commandline
     docker-compose down
````


### Test troubleshooting

When starting the containers, you may see an error similar to:

```
    invalid interpolation format for services.irods-clients.environment.CLIENT_USER_ID:
    "required variable UID is missing a value:  \nERROR: The UID environment 
    variable is unset". You may need to escape any $ with another $
```

which is caused by the `UID` shell variable being unset or not exported. See
this [Docker Compose issue](https://github.com/docker/compose/issues/2380) for
more details.


You can work around this by exporting the relevant variable(s):

```commandline
    export UID
    docker-compose up -d
```

or:

```commandline
    UID=$(id -u) docker-compose up -d
```
