"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from dataclasses import dataclass
from virid.core import EventMessage


class StartTrainingMessage(EventMessage): ...


@dataclass
class OneEpochMessage(EventMessage):
    epoch: int


class CheckPointMessage(EventMessage): ...


@dataclass
class EvalMessage(EventMessage):
    epoch: int
