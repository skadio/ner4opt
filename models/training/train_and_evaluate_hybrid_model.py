#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import os
import re
import sys
import timeit
from pathlib import Path

import joblib
import pandas as pd
import scipy
import sklearn_crfsuite
import spacy
from nl4opt_utils.metric import SpanF1
from nl4opt_utils.reader_utils import extract_spans
from simpletransformers.ner import NERArgs, NERModel
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from ner4opt.features import Featurizer
from ner4opt.utils import generate_feature_dictionary

root_directory = Path(os.path.abspath(''))
sys.path.append(str(root_directory))

# transformer spacy model for better pos, dep, ner accuracy
nlp = spacy.load("en_core_web_trf")
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

nlp_small = spacy.load("en_core_web_sm")
nlp_small.tokenizer = Tokenizer(nlp_small.vocab, token_match=re.compile(r'\S+').match)


def read_file_to_list(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def iob_to_sentences_and_labels(data):
    sentences = []
    labels = []
    tmp_sent = ''
    tmp_label = []

    for item_index, item in tqdm(enumerate(data)):
        if item == '' or item_index == len(data) - 1:
            sentences.append(tmp_sent.lstrip())
            labels.append(tmp_label)
            tmp_sent = ''
            tmp_label = []
        else:
            tmp_sent = tmp_sent + " " + item.split("\t")[0]
            tmp_label.append(item.split('\t')[-1])
    return sentences, labels


def augment_sentences_and_labels(sentences, labels):
    concatenated_sentences = []
    concatenated_labels = []
    number_of_tokens_to_ignore = []
    last_two_sentences_tuple = []

    for sent_index, sent in enumerate(sentences):

        nlp_sent = nlp_small(sent)

        if len(list(nlp_sent.sents)) >= 2:
            last_two_sentences = ' '.join([item.text for item in list(nlp_sent.sents)[-2::]])
        else:
            last_two_sentences = ' '.join([item.text for item in list(nlp_sent.sents)[-1::]])

        count = len(last_two_sentences.split())
        tmp_labels = labels[sent_index][-count::]

        concatenated_sentences.append(last_two_sentences + " " + sent)
        concatenated_labels.append(tmp_labels + labels[sent_index])
        number_of_tokens_to_ignore.append(len(last_two_sentences.split()))
        last_two_sentences_tuple.append((last_two_sentences, tmp_labels))

    return concatenated_sentences, concatenated_labels, number_of_tokens_to_ignore, last_two_sentences_tuple


def prepare_data(train_sentences, train_labels, dev_sentences, dev_labels):
    # ignore '-DOCSTART-'
    train_sentences = train_sentences[1::]
    train_labels = train_labels[1::]

    # ignore '-DOCSTART-'
    dev_sentences = dev_sentences[1::]
    dev_labels = dev_labels[1::]

    (concatenated_train_sentences, concatenated_train_labels, tokens_to_ignore_train,
     last_two_sentences_tuple_train) = augment_sentences_and_labels(train_sentences, train_labels)
    (concatenated_dev_sentences, concatenated_dev_labels, tokens_to_ignore_dev,
     last_two_sentences_tuple_dev) = augment_sentences_and_labels(dev_sentences, dev_labels)

    print("Number of training examples: ", len(concatenated_train_sentences))
    print("Number of dev examples: ", len(concatenated_dev_sentences))

    print("Number of training labels: ", len(concatenated_train_labels))
    print("Number of dev labels: ", len(concatenated_dev_labels))

    return (concatenated_train_sentences, concatenated_dev_sentences, concatenated_train_labels,
            concatenated_dev_labels, last_two_sentences_tuple_train, tokens_to_ignore_dev)


def train_transformers_model(last_two_sentences_tuple_train, train_data):
    # Transformer Model Training
    custom_labels = [
        'O',
        'B-CONST_DIR',
        'I-CONST_DIR',
        'B-LIMIT',
        'I-LIMIT',
        'B-VAR',
        'I-VAR',
        'B-OBJ_DIR',
        'B-OBJ_NAME',
        'I-OBJ_NAME',
        'B-PARAM',
        'I-PARAM',
    ]

    # create model
    model_args = NERArgs()
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "eval_loss"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 2000
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.num_train_epochs = 11
    model_args.adafactor_beta1 = 0.9
    model_args.weight_decay = 0.01
    model_args.max_seq_length = 512
    model_args.learning_rate = 4e-5
    model_args.train_batch_size = 1
    model_args.eval_batch_size = 1
    model_args.manual_seed = 123456789
    model_args.output_dir = "trained_transformer_model"
    model_args.use_cuda = True
    model = NERModel("roberta", "roberta-large", labels=custom_labels, use_cuda=True, args=model_args)

    modified_train_data = []
    example_index = 0

    words = last_two_sentences_tuple_train[0][0].split()
    word_labels = last_two_sentences_tuple_train[0][1]
    for index, item in enumerate(words):
        modified_train_data.append([example_index, item, word_labels[index]])

    for line in train_data[2::]:
        if line == '':
            example_index += 1
            words = last_two_sentences_tuple_train[example_index][0].split()
            word_labels = last_two_sentences_tuple_train[example_index][1]
            for index, item in enumerate(words):
                modified_train_data.append([example_index, item, word_labels[index]])
            continue
        else:
            vals = line.split('\t')
            label = vals[-1]
            modified_train_data.append([example_index, vals[0], label])

    train_df = pd.DataFrame(modified_train_data, columns=["sentence_id", "words", "labels"])
    # train model using only train data
    model.train_model(train_df)
    return model


def get_features(sentences, transformer_predictions):
    final_features = []

    for sentence_index, sentence in tqdm(enumerate(sentences)):

        featurizer = Featurizer(sentence, 'hybrid')
        features = featurizer.get_features()
        semantic_features = transformer_predictions[sentence_index]

        for token_index, token_features in enumerate(features):
            tmp_features = token_features + (semantic_features[token_index],)
            features[token_index] = tmp_features

        features_dict = []
        for token_index, token in enumerate(features):
            features_dict.append(generate_feature_dictionary(features, token_index, "hybrid"))

        final_features.append(features_dict)

    return final_features


def train_and_evaluate(train_filename, test_filename):
    train_filename = root_directory / train_filename
    dev_filename = root_directory / test_filename
    train_data = read_file_to_list(train_filename)
    train_sentences, train_labels = iob_to_sentences_and_labels(train_data)
    dev_data = read_file_to_list(dev_filename)
    dev_sentences, dev_labels = iob_to_sentences_and_labels(dev_data)

    (concatenated_train_sentences, concatenated_dev_sentences, concatenated_train_labels, concatenated_dev_labels,
     last_two_sentences_tuple_train, tokens_to_ignore_dev) = prepare_data(train_sentences, train_labels, dev_sentences,
                                                                          dev_labels)

    model = train_transformers_model(last_two_sentences_tuple_train, train_data)

    predictions_train, raw_outputs_train = model.predict(concatenated_train_sentences, split_on_space=True)

    transformer_predictions_train = []
    for item in tqdm(predictions_train):
        tmp_predictions = [list(val.values())[0] for val in item]
        transformer_predictions_train.append(tmp_predictions)

    predictions_dev, raw_outputs_dev = model.predict(concatenated_dev_sentences, split_on_space=True)

    transformer_predictions_dev = []
    for item in tqdm(predictions_dev):
        tmp_predictions = [list(val.values())[0] for val in item]
        transformer_predictions_dev.append(tmp_predictions)

    # Generate Train and Dev features
    X_train = get_features(concatenated_train_sentences, transformer_predictions_train)
    y_train = concatenated_train_labels
    X_dev = get_features(concatenated_dev_sentences, transformer_predictions_dev)

    # CRF with Randomized search
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        # maximum number of iterations for optimization algorithms
        max_iterations=500,
        all_possible_transitions=True)

    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

    # Search
    rs = RandomizedSearchCV(crf, params_space, cv=3, verbose=1, n_jobs=-1, n_iter=40, scoring=f1_scorer)

    rs.fit(X_train, y_train)

    # Save model
    now = datetime.date.today().strftime("_%d%m%Y")
    model_2save = 'hybrid' + now + '.pkl'
    full_path = model_2save
    joblib.dump(rs, full_path)

    rs_be = rs.best_estimator_
    print("Best Estimator: ", rs_be)

    # NL4OPT Custom Metric on dev set
    y_predictions = rs.predict(X_dev)

    # Truncate predictions to ignore augmented tokens
    truncated_y_dev = []
    truncated_y_predictions_dev = []

    for index, item in enumerate(y_predictions):
        truncated_y_predictions_dev.append(item[tokens_to_ignore_dev[index]::])
        truncated_y_dev.append(concatenated_dev_labels[index][tokens_to_ignore_dev[index]::])

    prediction = [extract_spans(prediction_value) for prediction_value in truncated_y_predictions_dev]
    ground_truth = [extract_spans(gt_value) for gt_value in truncated_y_dev]

    span_f1 = SpanF1()
    span_f1.reset()
    span_f1(prediction, ground_truth)

    return span_f1.get_metric()


if __name__ == '__main__':
    start = timeit.default_timer()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Path to the test data.', default='data/train.txt')
    parser.add_argument('--test', type=str, help='Path to the test data.', default='data/dev.txt')

    args = parser.parse_args()

    result = train_and_evaluate(args.train, args.test)
    print(result)

    with open('results.out', 'w') as filename:
        filename.write('%0.4f ' % result['micro@F1'])
    stop = timeit.default_timer()

    print('Total Time taken: ', stop - start)
