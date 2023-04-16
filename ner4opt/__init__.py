"""
:Version: 1.0.0 of April 9, 2023
This module defines the public interface of the
**Ner4Opt Library** providing access to the following modules:
    - ``Ner4Opt``
"""

from ner4opt._version import __version__
from ner4opt.ner4opt import Ner4Opt

from . import setup_nltk