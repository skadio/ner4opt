import logging

import spacy
from spacy.cli.download import download as spacy_download

from .constants import Constants

logger = logging.getLogger(__name__)

# download required models if not available

# SpaCy small model
try:
    spacy_model = spacy.load(Constants.SPACY_SMALL_MODEL)
except OSError:
    logger.warning(f"Spacy models '{Constants.SPACY_SMALL_MODEL}' not found.  Downloading and installing.")
    model_url = f"{Constants.SPACY_SMALL_MODEL}-{Constants.SPACY_MODEL_VERSION}"
    spacy_download(model_url, direct=True)

# SpaCy transformers model
try:
    spacy_model = spacy.load(Constants.SPACY_TRANSFORMERS_MODEL)
except OSError:
    logger.warning(f"Spacy models '{Constants.SPACY_TRANSFORMERS_MODEL}' not found.  Downloading and installing.")
    model_url = f"{Constants.SPACY_TRANSFORMERS_MODEL}-{Constants.SPACY_MODEL_VERSION}"
    spacy_download(model_url, direct=True)
