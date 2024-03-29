§# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2023-02-01

## Added
- Added lock to StatusManager singleton __new__() method to prevent race conditions.

## [0.2.1] - 2022-05-10

## Fixed
- Removed debugging statement from root of package.
- Removed newline character from version string.

## [0.2.0] - 2022-02-16

### Added
- MLModelDecorator base class.
- Ability to add a decorator to a model in the ModelManager.

## [0.1.1] - 2021-01-17

### Fixed
- Double initialization issue in ModelManager singleton class

## [0.1.0] - 2020-11-04

### Added
- Base class "MLModel" for holding machine learning model prediction functionality
- Class "ModelManager" for managing MLModel objects
