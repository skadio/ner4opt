"""
:Version: 1.0.0 of April 9, 2023
This module defines the public interface of the
**Ner4Opt Library** providing access to the following modules:
    - ``Ner4Opt``

Example Usage
~~~~~~~~~~~~~

>>> from ner4opt import Ner4Opt
>>> example_problem = "How much should the firm allocate in each asset so as to maximize its average return ?"
>>> ner4opt = Ner4Opt('hybrid')  # select a model
>>> entities = ner4opt.get_entities(example_problem)  # get entities

Help
~~~~
>>> from ner4opt import Ner4Opt
>>> Ner4Opt.__doc__
"""

from ner4opt._version import __version__
from ner4opt.ner4opt import Ner4Opt

from . import setup_nltk
