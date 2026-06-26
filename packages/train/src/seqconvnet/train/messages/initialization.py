"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from dataclasses import dataclass
from virid.core import EventMessage
from ..params import ModelParameters, DatasetParameters, EnvParameters


class TimerMessage(EventMessage): ...


@dataclass
class StartUpMessage(EventMessage):
    dataset_params: DatasetParameters
    model_params: ModelParameters
    env_params: EnvParameters


@dataclass
class CreateTransformerMessage(TimerMessage):
    dataset_params: DatasetParameters
    model_params: ModelParameters
    env_params: EnvParameters


@dataclass
class CreateRnnMessage(TimerMessage):
    dataset_params: DatasetParameters
    model_params: ModelParameters
    env_params: EnvParameters


@dataclass
class CreateDatasetMessage(TimerMessage):
    dataset_params: DatasetParameters
    model_params: ModelParameters
    env_params: EnvParameters


@dataclass
class CreateEvnMessage(TimerMessage):
    dataset_params: DatasetParameters
    model_params: ModelParameters
    env_params: EnvParameters


@dataclass
class CreateLoggerAndCheckpointMessage(TimerMessage): ...
