"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from virid.core import system, InfoMessage, WarnMessage, ErrorMessage, ViridApp


@system()
def info(message: InfoMessage) -> None:
    print(f"Info: {message.context}")


@system()
def warn(message: WarnMessage) -> None:
    print(f"Warn: {message.context}")


@system()
def error(message: ErrorMessage) -> None:
    print(f"Error: {message.error}")
    print(f"Error: {message.context}")


def register_logger_systems(app: ViridApp) -> None:
    app.register(info)
    app.register(warn)
    app.register(error)
