import csv
import os
import pickle
import string
from typing import List

import nltk
import numpy as np
import scipy
import tensorflow as tf
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.framework import ops

ops.reset_default_graph()
nltk.download('punkt')


def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words


def get_stopwords(path_to_file: str) -> List:
    stopwords = list()
    if os.path.isfile(path_to_file):
        with open(path_to_file, 'rb') as f:
            stopwords = pickle.load(f)
    return stopwords


class TFIDF:

    def __init__(self, csv_file: str, stopwords_list_file: str, max_features: int = 1000, **kwargs):
        # Start a graph session or init new
        self.max_features = max_features
        self.csv_file = csv_file
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)
        self.tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=get_stopwords(stopwords_list_file),
                                     max_features=self.max_features)
        self.P = None

    @staticmethod
    def read_train_data(csv_file):
        if os.path.isfile(csv_file):
            text_data = []
            with open(csv_file, 'r', encoding="utf8") as temp_output_file:
                reader = csv.reader(temp_output_file)
                for row in reader:
                    text_data.append(row)
            targets, texts = zip(*text_data)
            return targets, texts

    @staticmethod
    def lemmatize_texts(texts):
        mystem = Mystem()
        lemmatized_texts = []
        for text in texts:
            lemmatized_texts.append(''.join(mystem.lemmatize(text)).rstrip())
        return lemmatized_texts

    def __normalize_text(self, targets, texts):
        targets = [1. if x == '1' else 0. for x in targets]
        # Lower case
        texts = [x.lower() for x in texts]
        # Remove punctuation
        texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
        # Remove numbers
        texts = [''.join(c for c in x if c not in string.digits) for x in texts]
        # Trim extra whitespace
        texts = [' '.join(x.split()) for x in texts]
        # Lemmatize texts with mystem
        texts = TFIDF.lemmatize_texts(texts)
        return targets, texts

    def __random_walk(self, targets, texts):
        H_w = self.tfidf.fit_transform(texts, targets)
        H = np.zeros(H_w.shape, dtype='float64')
        cx = scipy.sparse.coo_matrix(H_w)
        for i, j, v in zip(cx.row, cx.col, cx.data):
            H[i][j] = 1. if v > 0 else 0.

        with tf.compat.v1.Session() as sess:
            Dv = tf.compat.v1.diag(np.array(sum(map(np.array, H))))
            Dvw = tf.compat.v1.diag(np.array([sum(x) for x in H_w.toarray()]))
            W = tf.compat.v1.diag(np.array([1. for x in H]))
            # Lines below are equal to
            # self.P = np.linalg.inv(Dv) @ np.transpose(H) @ W @ np.linalg.inv(Dvw) @ H_w
            current = tf.compat.v1.matmul(tf.compat.v1.matrix_inverse(Dv), H, transpose_b=True).eval()
            current = tf.compat.v1.matmul(current, W).eval()
            current = tf.compat.v1.matmul(current, Dvw).eval()
            self.P = tf.compat.v1.matmul(current, H_w.toarray()).eval()

    def fit(self):
        targets, texts = TFIDF.read_train_data(self.csv_file)
        targets, texts = self.__normalize_text(targets, texts)
        self.__random_walk(targets, texts)
        return targets, texts
