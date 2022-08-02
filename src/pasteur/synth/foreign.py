from mimetypes import init
from .base import Synth
from tempfile import NamedTemporaryFile


class ForeignSynth(Synth):
    """Loads Synthesizer code that runs in a foreign code environment."""


class PrivBayes(Synth):
    pass
