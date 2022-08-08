from ..utils import find_subclasses
from .adult import *
from .base import *
from .mimic import *


def get_datasets():
    return find_subclasses(Dataset)
