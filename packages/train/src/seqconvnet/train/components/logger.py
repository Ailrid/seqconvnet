"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from virid.core import component, ViridApp
from dataclasses import dataclass


@component()
@dataclass()
class Logger:
    log_writer = None


def bind_logger_components(app: ViridApp):
    app.bind(Logger)
