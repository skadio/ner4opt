#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
import unittest

import spacy

from ner4opt.utils import (format_entities, generate_connected_components, generate_singulars_and_plurals,
                           get_phrases_with_hyphens_and_quotes, l2_augment_sentence, spacy_tokenize_sentence)

nlp = spacy.load("en_core_web_sm")


class UtilsTest(unittest.TestCase):

    def setUp(self):

        self._example = "I like optimization. It is a good subject to learn. I would love to maximize my time I spend reading about it."
        self._example_for_spacy_tokenization = "Company A has a profit margin of 10% and B has 20%."
        self._words_and_phrases = ["batch", "coconut", "actor", "wheel", "blood pressure medicine"]
        self._spacy_sentence = nlp('Let us learn about " puff of air test " and a sewing - machine')

    def test_spacy_tokenize_sentence(self):
        result = spacy_tokenize_sentence(self._example_for_spacy_tokenization)
        self.assertEqual(result,
                         "Company A has a profit margin of 10 % and B has 20 % .")

    def test_l2_augment_sentence(self):
        augmented_sentence, augmentation = l2_augment_sentence(self._example)
        self.assertEqual(augmentation,
                         "It is a good subject to learn. I would love to maximize my time I spend reading about it.")

    def test_generate_singulars_and_plurals(self):
        result = generate_singulars_and_plurals(self._words_and_phrases)
        self.assertEqual(set(result), {'coconuts', 'batches', 'blood pressure medicines', 'wheels', 'actors'})

    def test_get_phrases_with_hyphens_and_quotes(self):
        result = get_phrases_with_hyphens_and_quotes(self._spacy_sentence)
        self.assertEqual([item.text for item in result], ['" puff of air test "', "sewing - machine"])

    def test_generate_connected_components(self):
        result = generate_connected_components(self._spacy_sentence)
        expected = {
            0: [0, 0],
            1: [1, 1],
            2: [2, 2],
            3: [3, 3],
            4: [4, 10],
            5: [4, 10],
            6: [4, 10],
            7: [4, 10],
            8: [4, 10],
            9: [4, 10],
            10: [10, 10],
            11: [11, 15],
            12: [11, 15],
            13: [11, 15],
            14: [11, 15]
        }
        self.assertEqual(result, expected)

    def test_format_entities(self):
        result = format_entities("I want to maximize my return", [{
            "I": "O"
        }, {
            "want": "O"
        }, {
            "to": "O"
        }, {
            "maximize": "B-OBJ_DIR"
        }, {
            "my": "O"
        }, {
            "return": "B-OBJ_NAME"
        }], [1.0, 1.0, 1.0, 0.99, 1.0, 0.98])
        self.assertEqual(result, [{
            'start': 10,
            'end': 18,
            'word': 'maximize',
            'entity_group': 'OBJ_DIR',
            'score': 0.99
        }, {
            'start': 22,
            'end': 28,
            'word': 'return',
            'entity_group': 'OBJ_NAME',
            'score': 0.98
        }])
