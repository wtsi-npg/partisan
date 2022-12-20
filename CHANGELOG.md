# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

 - Replace setup.py and requirements files with pyproject.toml

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
