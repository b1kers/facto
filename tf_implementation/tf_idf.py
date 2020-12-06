import csv
import os
import string

import nltk
import pickle
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


class TFIDF:

    def __init__(self, csv_file='common_train.csv'):
        # Start a graph session or init new
        self.max_features = 1000
        self.text_data = None
        self.texts = None
        self.target = None
        self.tfidf = None
        self.sparse_tfidf_texts = None
        self.P = None
        self.read_file(csv_file)
        self.normalize_text()
        self.tf_idf()
        self.random_walk()

    def read_file(self, csv_file='common_train.csv'):
        if os.path.isfile(csv_file):
            self.text_data = []
            with open(csv_file, 'r', encoding="utf8") as temp_output_file:
                reader = csv.reader(temp_output_file)
                for row in reader:
                    self.text_data.append(row)
        self.texts = [x[1] for x in self.text_data]
        self.target = [x[0] for x in self.text_data]

    def lemmatize_texts(self):
        mystem = Mystem()
        lemmatized_texts = []
        for text in self.texts:
            lemmatized_texts.append(''.join(mystem.lemmatize(text)).rstrip())
        return lemmatized_texts

    def normalize_text(self):
        self.target = [1. if x == '1' else 0. for x in self.target]
        # Lower case
        self.texts = [x.lower() for x in self.texts]
        # Remove punctuation
        self.texts = [''.join(c for c in x if c not in string.punctuation) for x in self.texts]
        # Remove numbers
        self.texts = [''.join(c for c in x if c not in string.digits) for x in self.texts]
        # Trim extra whitespace
        self.texts = [' '.join(x.split()) for x in self.texts]
        # Lemmatize texts with mystem
        self.texts = self.lemmatize_texts()

    def tf_idf(self):
        # Create TF-IDF of texts
        with open('tf_implementation/stopwords.pickle', 'rb') as f:
            stopwords = pickle.load(f)
        self.tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords,
                                     max_features=self.max_features)
        texts = self.texts
        self.sparse_tfidf_texts = self.tfidf.fit_transform(texts)

    def random_walk(self):
        H_w = self.sparse_tfidf_texts
        H = np.zeros(H_w.shape, dtype='float64')
        cx = scipy.sparse.coo_matrix(H_w)
        for i, j, v in zip(cx.row, cx.col, cx.data):
            H[i][j] = 1. if v > 0 else 0.

        with tf.Session() as sess:
            Dv = tf.diag(np.array(sum(map(np.array, H))))
            Dvw = tf.diag(np.array([sum(x) for x in H_w.toarray()]))
            W = tf.diag(np.array([1. for x in H]))
            # Lines below are equal to
            # self.P = np.linalg.inv(Dv) @ np.transpose(H) @ W @ np.linalg.inv(Dvw) @ H_w
            current = tf.matmul(tf.matrix_inverse(Dv), H, transpose_b=True).eval()
            current = tf.matmul(current, W).eval()
            current = tf.matmul(current, Dvw).eval()
            self.P = tf.matmul(current, H_w.toarray()).eval()
