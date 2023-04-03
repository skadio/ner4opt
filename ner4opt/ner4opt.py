import os
import pathlib
from typing import List

import joblib

from .constants import Constants
from .exceptions import InvalidModelError
from .features import (get_features, get_objective_name_spans,
                       get_var_name_spans, mark_keywords)
from .utils import (get_feature_dict, l2_augment_sentence, load_torch_model,
                    validate_and_format_entities)


class Ner4Opt(object):

    def __init__(self, model_name: str = Constants.HYBRID, use_gpu: bool = False):
        """Init Model."""
        _root_directory = pathlib.Path(__file__).parent.parent

        self._classical_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY,
                                                  Constants.CLASSICAL_MODEL_NAME)
        self._classical_plus_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY,
                                                       Constants.CLASSICAL_PLUS_MODEL_NAME)
        self._hybrid_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY, Constants.HYBRID_MODEL_NAME)

        self._model_name = model_name

        self._crf_model = None
        self._deep_model = None

        if model_name == Constants.CLASSICAL:
            self._crf_model = joblib.load(self._classical_model_path)

        elif model_name == Constants.CLASSICAL_PLUS:
            self._crf_model = joblib.load(self._classical_plus_model_path)

        elif model_name == Constants.SEMANTIC:
            self._deep_model = load_torch_model(Constants.SEMANTIC_DEEP_MODEL, use_gpu=use_gpu)

        elif model_name == Constants.HYBRID:
            self._deep_model = load_torch_model(Constants.HYBRID_DEEP_MODEL, use_gpu=use_gpu)
            self._crf_model = joblib.load(self._hybrid_model_path)

        else:
            raise InvalidModelError(
                "Invalid Model {} passed. Model name should be one of the following classical, classical_plus, deep or hybrid"
                .format(model_name))


    def get_entities(self, text: str) -> List[dict]:
        """Extract Entities."""
        augmented_sentence, augmentation = l2_augment_sentence(text)

        if self._model_name in [Constants.CLASSICAL, Constants.CLASSICAL_PLUS, Constants.HYBRID]:

            linguistic_features = get_features(augmented_sentence)

            if self._model_name == Constants.HYBRID:
                semantic_features, _ = self._deep_model.predict([augmented_sentence])
                semantic_features = [list(item.values())[0] for item in semantic_features[0]]

                # Gazetter Features
                obj_dir_gazetter_features = mark_keywords(augmented_sentence, Constants.OBJ_DIR_KEYWORDS, "OBJ_DIR")
                const_dir_gazetter_features = mark_keywords(augmented_sentence, Constants.CONST_DIR_KEYWORDS,
                                                            "CONST_DIR")

                # Objective name automaton
                obj_name_features = get_objective_name_spans(augmented_sentence)
                # Var name features
                var_name_features = get_var_name_spans(augmented_sentence)

                # Concatenate features
                concatenated_features = []
                for token_index, token_features in enumerate(linguistic_features):
                    concatenated_features.append(token_features + (
                        obj_dir_gazetter_features[token_index], const_dir_gazetter_features[token_index],
                        obj_name_features[token_index], var_name_features[token_index], semantic_features[token_index],
                    ))
                features_dict = []

                for feature in concatenated_features:
                        features_dict.append(get_feature_dict(concatenated_features, token_index, self._model_name))

            else:
                features_dict = []
                for token_index, token in enumerate(linguistic_features):
                    features_dict.append(get_feature_dict(linguistic_features, token_index, self._model_name))

            predicted_entities = self._crf_model.predict([features_dict])
            predicted_entities_probabilities = self._crf_model.best_estimator_.predict_marginals([features_dict])
            probabilities = []
            for entity_index, entity in enumerate(predicted_entities[0]):
                probabilities.append(predicted_entities_probabilities[0][entity_index][entity])
            predicted_entities = predicted_entities[0][len(augmentation.split())::]
            tokens = text.split()
            predicted_entities = [{tokens[item_index]: item} for item_index, item in enumerate(predicted_entities)]
            probabilities = probabilities[len(augmentation.split())::]
            predicted_entities = validate_and_format_entities(text, predicted_entities, probabilities)
        elif self._model_name == Constants.SEMANTIC:
            predicted_entities, probabilities = self._deep_model.predict([augmented_sentence])
            predicted_entities = predicted_entities[0][len(augmentation.split())::]
            probabilities = probabilities[0][len(augmentation.split())::]
            predicted_entities = validate_and_format_entities(text, predicted_entities, probabilities, crf=False)
        return predicted_entities


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
