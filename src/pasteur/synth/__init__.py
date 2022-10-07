from ..utils import find_subclasses
from .base import *
from .synth import *
from .extern import *


def get_synth():
    return find_subclasses(Synth)
