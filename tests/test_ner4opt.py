#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
import json
import math
import os
import unittest
from pathlib import Path

from ner4opt import Ner4Opt


class Ner4OptTest(unittest.TestCase):

    @staticmethod
    def load_pickled_object(filename):
        with open(filename, 'r') as f:
            result = json.load(f)
        return result

    @staticmethod
    def read_example_sentence(filename):
        with open(filename) as file:
            lines = [line.rstrip() for line in file]
        return lines[0]

    @staticmethod
    def assert_list_of_dicts_equal(list_one, list_two):
        assert len(list_one) == len(list_two)

        def are_dicts_close(d1, d2, score_tol=1e-6):
            return (d1['start'] == d2['start'] and d1['end'] == d2['end'] and d1['word'] == d2['word'] and
                    d1['entity_group'] == d2['entity_group'] and
                    math.isclose(d1['score'], d2['score'], rel_tol=score_tol))

        for dict_one, dict_two in zip(list_one, list_two):
            assert are_dicts_close(dict_one, dict_two)

    def setUp(self):

        _current_dir = Path(__file__).parent

        # Example Problem Description
        self._example = self.read_example_sentence(os.path.join(_current_dir, "fixtures/example_problem.txt"))

        # Expected Results for each model for the above problem description
        self._expected_lexical_result = self.load_pickled_object(
            os.path.join(_current_dir, "fixtures/lexical_result.json"))
        self._expected_lexical_plus_result = self.load_pickled_object(
            os.path.join(_current_dir, "fixtures/lexical_plus_result.json"))
        self._expected_semantic_result = self.load_pickled_object(
            os.path.join(_current_dir, "fixtures/semantic_result.json"))
        self._expected_hybrid_result = self.load_pickled_object(
            os.path.join(_current_dir, "fixtures/hybrid_result.json"))

        # Ner4opt Objects for each model
        self._lexical_ner_model = Ner4Opt("lexical")
        self._lexical_plus_ner_model = Ner4Opt("lexical_plus")
        self._semantic_ner_model = Ner4Opt("semantic")
        self._hybrid_ner_model = Ner4Opt("hybrid")

    # Usage tests for all models
    def test_get_entities_lexical(self):
        result = self._lexical_ner_model.get_entities(self._example)
        self.assert_list_of_dicts_equal(result, self._expected_lexical_result)

    def test_get_entities_lexical_plus(self):
        result = self._lexical_plus_ner_model.get_entities(self._example)
        self.assert_list_of_dicts_equal(result, self._expected_lexical_plus_result)

    def test_get_entities_semantic(self):
        result = self._semantic_ner_model.get_entities(self._example)
        self.assert_list_of_dicts_equal(result, self._expected_semantic_result)

    def test_get_entities_hybrid(self):
        result = self._hybrid_ner_model.get_entities(self._example)
        self.assert_list_of_dicts_equal(result, self._expected_hybrid_result)

    # Invalid usage test
    def test_get_entities_invalid_usage(self):

        with self.assertRaises(ValueError) as context:
            invalid_ner_model = Ner4Opt("invalid")
