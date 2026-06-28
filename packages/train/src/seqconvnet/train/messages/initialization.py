"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from virid.core import EventMessage

class CreateTransformerMessage(EventMessage): ...


class CreateRnnMessage(EventMessage): ...


class CreateDatasetMessage(EventMessage): ...


class CreateEvnMessage(EventMessage): ...


class CreateLoggerAndCheckpointMessage(EventMessage): ...

