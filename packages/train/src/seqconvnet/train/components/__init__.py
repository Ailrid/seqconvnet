from .initialization import *
from .logger import *
from .training import *
from virid.core import ViridApp


def bind_components(app: ViridApp) -> None:
    bind_initialization_components(app)
    bind_training_components(app)
    bind_logger_components(app)
