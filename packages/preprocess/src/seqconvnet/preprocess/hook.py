"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

from .message import TimerMessage
from virid.core import ExecuteHookContext, ViridApp
import time


def timer_start_hook(
    _message: TimerMessage, context: ExecuteHookContext, _success: bool
):
    context.payload["start_time"] = time.time()


def timer_end_hook(message: TimerMessage, context: ExecuteHookContext, _success: bool):
    print(
        f"{message.__class__.__name__} cost: "
        + f"{time.time() - context.payload['start_time']:.2f}s"
    )


def activate_hook(app: ViridApp):
    app.on_before_execute(TimerMessage, timer_start_hook)
    app.on_after_execute(TimerMessage, timer_end_hook)
