import os
import pathlib
from typing import List

import joblib

from .constants import Constants
from .exceptions import InvalidModelError
from .features import Featurizer
from .utils import format_entities, generate_feature_dictionary, l2_augment_sentence, load_torch_model


class Ner4Opt(object):
    """Extract named entities for a given optimization problem.

    The main objective of the Ner4Opt library is to generate named entities for optimization problems.
    There are six entities the current library extracts; CONST_DIR (constraint direction), LIMIT (limit),
    OBJ_DIR (objective direction), OBJ_NAME (objective name), PARAM (parameter) and VAR (variable).

    This library provides access to four main models for extracting these entities.
    `lexical` model built using lexical features alone,
    `lexical_plus` model which in addition to lexical features uses optimization scenario specific features,
    `semantic` model which uses a roberta based transformers model and
    `hybrid` model which combines all the best features of both lexical and semantic models to achieve best results.

    Attributes
    -------
    model_name (str) : Specifies the model to use for extracting the entities.
                       Options include `lexical`, `lexical_plus`, `semantic` and `hybrid`
                       Default is `hybrid`
    use_gpu (bool) :   Specifies the model to use gpu while calculating transformer based features.
                       Default is False

    Methods
    -------
    get_entities()

    Attributes
    -------
    text (str) : Sample optimization problem

    Returns
    -------

    Returns named entities of a given optimization problem. A list of dictionaries for each entity

    Keys:
        Name: start, dtype: int: Starting character index of the entity. Range is [0, len(text)]
        Name: end, dtype: int: Ending character index of the entity. Range is [0, len(text)]
        Name: word, dtype: str: Entity phrase
        Name: entity_group, dtype: str: Type of the entity
        Name: score, dtype: float: Defines the confidence of the prediction. Range is [0.0, 100.0]
    """

    def __init__(self, model_name: str = Constants.HYBRID, use_gpu: bool = False):
        """Init Ner4Opt class."""

        _root_directory = pathlib.Path(__file__).parent.parent

        self._lexical_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY,
                                                Constants.LEXICAL_MODEL_NAME)
        self._lexical_plus_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY,
                                                     Constants.LEXICAL_PLUS_MODEL_NAME)
        self._hybrid_model_path = os.path.join(_root_directory, Constants.MODELS_DIRECTORY, Constants.HYBRID_MODEL_NAME)

        self._model_name = model_name
        self._crf_model = None
        self._deep_model = None

        # Load model
        if model_name == Constants.LEXICAL:
            self._crf_model = joblib.load(self._lexical_model_path)

        elif model_name == Constants.LEXICAL_PLUS:
            self._crf_model = joblib.load(self._lexical_plus_model_path)

        elif model_name == Constants.SEMANTIC:
            self._deep_model = load_torch_model(Constants.SEMANTIC_DEEP_MODEL, use_gpu=use_gpu)

        elif model_name == Constants.HYBRID:
            self._deep_model = load_torch_model(Constants.HYBRID_DEEP_MODEL, use_gpu=use_gpu)
            self._crf_model = joblib.load(self._hybrid_model_path)

        else:
            raise InvalidModelError(
                "Invalid Model {} passed. Model name should be one of the following lexical, lexical_plus, semantic or hybrid"
                .format(model_name))

    def get_entities(self, text: str) -> List[dict]:
        """Extract Entities."""

        predicted_entities = []

        if not text:
            return predicted_entities

        augmented_sentence, augmentation = l2_augment_sentence(text)
        featurizer = Featurizer(augmented_sentence, self._model_name)
        tokens_to_ignore_from_augmentation = len(augmentation.split())

        if self._model_name == Constants.SEMANTIC:
            predicted_entities, probabilities = self._deep_model.predict([augmented_sentence], split_on_space=True)
            # ignore augmentation tokens
            predicted_entities = predicted_entities[0][tokens_to_ignore_from_augmentation::]
            probabilities = probabilities[0][tokens_to_ignore_from_augmentation::]
            # format predicted entities
            predicted_entities = format_entities(text, predicted_entities, probabilities, crf_flag=False)
        else:
            features = featurizer.get_features()
            if self._model_name == Constants.HYBRID:
                # add transformer predictions as an additional feature
                semantic_features, _ = self._deep_model.predict([augmented_sentence], split_on_space=True)
                semantic_features = [list(item.values())[0] for item in semantic_features[0]]

                for token_index, token_features in enumerate(features):
                    tmp_features = token_features + (semantic_features[token_index],)
                    features[token_index] = tmp_features

            # generate feature dictionary, an input required for the CRF model
            crf_model_input = []
            for feature_index in range(len(features)):
                crf_model_input.append(generate_feature_dictionary(features, feature_index, self._model_name))

            # predict and format
            predicted_entities = self._crf_model.predict([crf_model_input])
            predicted_entities_probabilities = self._crf_model.best_estimator_.predict_marginals([crf_model_input])

            probabilities = []
            for entity_index, entity in enumerate(predicted_entities[0]):
                probabilities.append(predicted_entities_probabilities[0][entity_index][entity])
            predicted_entities = predicted_entities[0][tokens_to_ignore_from_augmentation::]

            tokens = text.split()
            predicted_entities = [{tokens[item_index]: item} for item_index, item in enumerate(predicted_entities)]
            probabilities = probabilities[tokens_to_ignore_from_augmentation::]
            predicted_entities = format_entities(text, predicted_entities, probabilities)

        return predicted_entities
