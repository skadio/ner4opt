#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
import unittest

from ner4opt.featurizer import Featurizer


class FeaturesTest(unittest.TestCase):

    def setUp(self):

        self._example_sentence = "I want to maximize the number of batches of cookies and time spent"
        self._gazetteer = ["maximize", "minimize"]
        self._featurizer = Featurizer(self._example_sentence, "hybrid")

    def test_get_linguistic_features(self):
        result = self._featurizer.get_linguistic_features()
        self.assertEqual(
            result[0],
            ('I', 'i', True, True, 'I', 'PRON', 'PRP', 'nsubj', 'X', True, True, True, False, 'B-NOUN_CHUNK', 'O'))
        self.assertEqual(
            result[-1],
            ('spent', 'spent', False, False, 'spend', 'VERB', 'VBN', 'acl', 'xxxx', True, False, True, False, 'O', 'O'))

    def test_get_gazetteer_features(self):
        result = self._featurizer.get_gazetteer_features(self._gazetteer, "OBJ_DIR")
        self.assertEqual(result, ['O', 'O', 'O', 'B-OBJ_DIR', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])

    def test_get_prepositional_chunk_spans(self):
        result = self._featurizer.get_prepositional_chunk_spans()
        self.assertEqual(result, [
            'O', 'O', 'O', 'O', 'B-PREP_CHUNK', 'I-PREP_CHUNK', 'I-PREP_CHUNK', 'B-PREP_CHUNK', 'I-PREP_CHUNK',
            'I-PREP_CHUNK', 'I-PREP_CHUNK', 'I-PREP_CHUNK', 'O'
        ])

    def test_get_objective_name_spans(self):
        result = self._featurizer.get_objective_name_spans()
        self.assertEqual(result, [
            'O', 'O', 'O', 'O', 'B-OBJ_NAME', 'I-OBJ_NAME', 'I-OBJ_NAME', 'I-OBJ_NAME', 'I-OBJ_NAME', 'I-OBJ_NAME',
            'I-OBJ_NAME', 'I-OBJ_NAME', 'O'
        ])

    def test_get_variable_spans(self):
        result = self._featurizer.get_variable_spans()
        self.assertEqual(result, ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VAR', 'O', 'B-VAR', 'O', 'B-VAR', 'O'])

    def test_get_features(self):
        result = self._featurizer.get_features()
        self.assertEqual(result[0], ('I', 'i', True, True, 'I', 'PRON', 'PRP', 'nsubj', 'X', True, True, True, False,
                                     'B-NOUN_CHUNK', 'O', 'O', 'O', 'O', 'O', 'O'))
        self.assertEqual(result[-1], ('spent', 'spent', False, False, 'spend', 'VERB', 'VBN', 'acl', 'xxxx', True,
                                      False, True, False, 'O', 'O', 'O', 'O', 'O', 'O', 'O'))
