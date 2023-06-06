# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.1]

### Fixed
 - Incorrect recursive behaviour of RodsItem::add/remove/supersede_permissions methods

## [2.5.0]

### Added
 - attr parameter to RodsItem::metadata method
 - iRODS 4.2.12 to CI.
 
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
