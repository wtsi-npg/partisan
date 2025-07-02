# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For more recent changes, see the comprehensive automatically generated changelogs at
https://github.com/wtsi-npg/partisan/releases

## [2.10.1]

### Fixed
 - Add test skip for a bug fixed in iRODS 4.3.2

## [2.10.0]

### Added
 - Support for AVU query operators
 - *_human iRODS groups to the test fixtures

## [2.9.4]

### Fixed
 - Raise supersede operations to info level logging 

## [2.9.3]

### Fixed
 - Fix logger configuration by not caching the logger

## [2.9.2]

### Changed
 - Use structlog.get_logger() with no arguments

## [2.9.1]

### Changed
 - Raise logging from supersede_ methods to info level

## [2.9.0]

### Added
 - Support for iRODS native ("irods::") AVUs

### Changed
 - Bump structlog from 23.1.0 to 23.2.0

## [2.8.1]

### Fixed
 - Incorrect return value when using the user_type keyword for
   RodsItem::acl/permissions 

## [2.8.0]

### Added
 - New rods_users function to list all users
 - New user_type keyword for the existing RodsItem::acl/permissions

### Changed
 - Bump black from 23.7.0 to 23.9.1
 - Bump click from 8.1.6 to 8.1.7
 - Bump pytest from 7.4.0 to 7.4.2
 - Bump setuptools from 68.1.2 to 68.2.2

## [2.7.0]

### Added
- Support for iRODS 4.2.12 iquest "no results found" exit code

### Changed
- Bump setuptools from 68.1.0 to 68.1.2

## [2.6.0]

### Added

 - namespace keyword to RodsItem::avu method
 - attribute argument stringification to RodsItem::avu method
 - pyproject.toml

### Removed
 - setup.py

### Changed
 - Bump click from 8.1.3 to 8.1.6
 - Bump setuptools from 68.0.0 to 68.1.0
 - Bump setuptools-git-versioning from 1.13.3 to 1.13.5
 - Bump black from 23.3.0 to 23.7.0
 - Bump pytest from 7.3.1 to 7.4.0

### Fixed
 - Appending multiple ACLs internal to RodsItem::acl method
 - Docstring errors

## [2.5.1]

### Fixed
 - Incorrect recursive behaviour of RodsItem::add/remove/supersede_permissions methods

## [2.5.0]

### Added
 - attr parameter to RodsItem::metadata method
 - iRODS 4.2.12 to CI
 
## [2.4.0]

### Added
 - RodsItem::avu method
 - Singularity container support for iRODS clients

### Removed
 - Use of Conda-based iRODS clients in CI.
 - Remove use of Conda-based Python in CI.

## [2.3.2]

### Fixed
 - Handle inconsistent own/read permissions in iRODS

## [2.3.1]

### Fixed
 - Handle iuserinfo results for apparently non-existent users, such as those on remote zones

## [2.3.0]

### Added
 - User class, rods_user and current_user functions, iuserinfo icommand

### Changed
 - Bump black from 23.1.0 to 23.3.0
 - Bump pytest from 7.2.2 to 7.3.1
 - Bump structlog from 22.3.0 to 23.1.0

## [2.2.0]

### Added
 - Method for getting a list of the values in an AsValueEnum

### Changed
 - Move to github hosted docker images for ci and docker-compose
 - Update docker-compose setup to match current images and client locations

 - Bump pytest from 7.2.1 to 7.2.2

## [2.1.0]

### Added
 - Replica trimming for DataObjects
 - icp to the supported icommands
 
### Changed
 - Update to ubuntu-latest in github actions
 - Move from checkout@v2 to @v3
 - Move from cache@v2 to @v3
 
 - Bump pytest from 7.2.0 to 7.2.1
 - Bump black from 22.6.0 to 23.1.0
 - Bump setuptools-scm from 7.0.5 to 7.1.0
 - Bump structlog from 21.5.0 to 22.3.0

### Fixed
 - Do not cache rods_type, but set when first known

## [2.0.0]

### Added
 - Metadata utility methods
 - pls command line client with bash completion
 - Cached rods_type attribute to iRODS objects
 - rods_path_type and rods_path_exists support functions

### Changed
 - Recurse keyword added to Collection.iter_contents
 - Sort the result of Collection.iter_contents
 - Bump black from 22.8.0 to 22.10.0
 - Bump pytest from 7.1.3 to 7.2.0

### Fixed
 - Missing __hash__ implementation on iRODS objects

## [1.2.0]

### Added
 -  Support for created/modified timestamps on DataObjects and Replicas

### Changed
 - Require Python >= 3.10
 - Update pytest from 7.1.2 to 7.1.3

## [1.1.0]

### Added
 - Structlog default configuration
 - Custom stringification for AsValueEnum
 - Namespace for DublinCore enum stringification

### Removed
 - Remove iRODS 4.2.10 from GitHub actions workflow

### Changed
 - Update baton version in GitHub actions workflow from 3.3.0 to 4.0.0
 - Update structlog from 21.5.0 to 22.1.0

### Fixed
 - Typehint for add_metadata implied an explicit tuple could be passed
 - Debug log spam when structlog configuration was absent

## [1.0.0]
