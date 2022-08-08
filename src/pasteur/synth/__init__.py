from ..utils import find_subclasses
from .base import *
from .privbayes import *

# from .sdv import *


def get_synth():
    return find_subclasses(Synth)
