import pickle
from typing import List
from .exceptions import InvalidModelError

from sentence_transformers import SentenceTransformer


class Ner4Opt(object):

    CLASSICAL = 'classical'
    CLASSICAL_PLUS = 'classical_plus'
    DEEP = 'deep'
    HYBRID = 'hybrid'

    def __init__(self, model_name: str = 'hybrid'):
        """Init Model."""

        self._model_name = model_name

        if model_name == Ner4Opt.CLASSICAL:
            self._model = pickle.load(open('models/classical.pkl', 'rb'))

        elif model_name == Ner4Opt.CLASSICAL_PLUS:
            self._model = pickle.load(open('models/classical_plus.pkl', 'rb'))

        elif model_name == Ner4Opt.DEEP:
            self._model = SentenceTransformer("skadio/optimization_finetuned_roberta")

        elif model_name == Ner4Opt.HYBRID:
            self._model = pickle.load(open('models/hybrid_crf_model.pkl', 'rb'))

        else:
            raise InvalidModelError(
                "Invalid Model {} passed. Model name should be one of the following classical, classical_plus, deep or hybrid"
                .format(model_name))

    def _featurize_text(self, text: str) -> List[dict]:
        """Featurize Text."""
        pass

    def _validate_and_format_entities(self, entities) -> List[dict]:
        """Validates if the entities are in huggingface format and formats if not."""
        pass

    def get_entities(self, text: str) -> List[dict]:
        """Extract Entities."""
        if self._model_name == Ner4Opt.DEEP:
            predicted_entities = self._model.predict(text)
        else:
            features = self._featurize_text(text)
            predicted_entities = self._model.predict(features)
        entities = self._validate_and_format_entities(predicted_entities)
        return entities


if __name__ == '__main__':

    # !pip install ner4opt

    ner4opt = Ner4Opt()
    entities = ner4opt.get_entities("I want to maximize my profit")

    # Result
    """
    [
      {
        "entity_group": "O",
        "score": 1,
        "word": "I",
        "start": 0,
        "end": 1
      },
      {
        "entity_group": "O",
        "score": 1,
        "word": "want",
        "start": 2,
        "end": 5
      },
      {
        "entity_group": "O",
        "score": 1,
        "word": "to",
        "start": 6,
        "end": 8
      },
      {
        "entity_group": "OBJ_DIR",
        "score": 0.999536395072937,
        "word": " maximize",
        "start": 10,
        "end": 18
      },
      {
        "entity_group": "O",
        "score": 1,
        "word": "my",
        "start": 18,
        "end": 20
      },
      {
        "entity_group": "OBJ_NAME",
        "score": 0.999974250793457,
        "word": " profit",
        "start": 22,
        "end": 28
      }
    ]
    """
