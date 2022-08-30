# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
