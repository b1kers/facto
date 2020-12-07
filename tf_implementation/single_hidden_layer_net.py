import csv
import os
from difflib import get_close_matches
from typing import List, Dict

import numpy as np
import pymorphy2
import tensorflow as tf

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from tf_idf import TFIDF


def get_first(collection: List):
    try:
        return collection[0]
    except IndexError:
        return None


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


class SingleHiddenLayerNetData:

    def __init__(self, cities_csv: str = None, common_csv: str = None,
                 stopwords_list_file: str = None, cutoff: float = 0.8,
                 context: dict = None, **kwargs):
        self.tf_idf = TFIDF(common_csv, stopwords_list_file)
        self.cutoff = cutoff
        self.context = context
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)
        self.cities = SingleHiddenLayerNetData.get_cities(cities_csv)
        self.targets, self.texts = self.tf_idf.fit()
        self.P_matrix = self.tf_idf.P
        self.vocabulary = self.tf_idf.tfidf.vocabulary_
        self.morph = pymorphy2.MorphAnalyzer()
        self.dv = DictVectorizer()

    def __get_morpho_features(self, word_in_context: str, context_idx: int):
        word_data = get_first(self.morph.parse(word_in_context))
        morpho_features = dict()
        for word_feature in dir(word_data):
            if word_feature in ['count', 'index', 'normal_form', 'tag', 'score']:
                morpho_features[f'{word_feature}_{context_idx}'] = getattr(word_data, word_feature)
            morpho_features[f'target_{context_idx}'] = 1
        return morpho_features

    def fit_transform(self):
        x_values = list()
        y_values = list()
        for idx, doc in enumerate(self.texts):
            splitted_docs = doc.split()
            target_label = self.targets[idx]
            for inner_idx, word in enumerate(splitted_docs):
                features = dict()
                word_context = splitted_docs[:inner_idx][-self.context['before']:]
                word_context += splitted_docs[inner_idx + 1:][:self.context['after']]
                for context_idx, word_in_context in enumerate(word_context):
                    feature_value = 0
                    feature_key = f'P_{context_idx}'
                    features.update(self.__get_morpho_features(word_in_context, context_idx))
                    if word_in_context in self.vocabulary:
                        feature_value_range = self.P_matrix[self.vocabulary[word_in_context]]
                        if target_label:
                            closest_match = get_first(
                                get_close_matches(word, self.cities, cutoff=self.cutoff))
                            try:
                                feature_value = feature_value_range[self.vocabulary[closest_match]]
                            except KeyError:
                                pass
                    feature_value = max(feature_value, min(feature_value_range))
                    features[feature_key] = feature_value
                if context_idx == sum(self.context.values()) - 1:
                    x_values.append(features)
                    y_values.append(float(target_label))
        return x_values, y_values

    @staticmethod
    def get_cities(cities_csv: str) -> List:
        cities = list()
        if os.path.isfile(cities_csv):
            with open(cities_csv, 'r', encoding='utf-8') as temp_output_file:
                try:
                    reader = csv.reader(temp_output_file)
                    for i, row in enumerate(reader):
                        try:
                            cities.append(row[-1].lower().strip())
                        except csv.Error:
                            raise csv.Error(f'Invalid line №{i}')
                except csv.Error:
                    raise csv.Error('Broken file')
        return cities


class SingleHiddenLayerNet:

    def __init__(self, **kwargs):
        self.context = None
        self.cities_csv = None
        self.batch_size = None
        self.hidden_layer_nodes = None
        self.train_data = None
        self.epochs = None
        self.stopwords_list_file = None
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)
        # all the above values should be defined
        # to avoid pylint warning
        self.data = SingleHiddenLayerNetData(self.cities_csv, self.train_data,
                                             self.stopwords_list_file)
        x_vals, y_vals = self.data.fit_transform()
        self.x_vals = self.data.dv.fit_transform(x_vals).toarray()
        self.y_vals = np.array(y_vals)
        # Create graph session
        self.sess = tf.compat.v1.Session()

        # Initialize placeholders
        self.x_data = tf.compat.v1.placeholder(shape=[None, 5], dtype=tf.float32)
        self.y_target = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)

        # Create variables for both NN layers
        inputs = tf.Variable(tf.compat.v1.random_normal(
            shape=[5, self.hidden_layer_nodes]))  # inputs -> hidden nodes
        biases = tf.Variable(
            tf.compat.v1.random_normal(
                shape=[self.hidden_layer_nodes]))  # one biases for each hidden node
        hidden = tf.Variable(
            tf.compat.v1.random_normal(
                shape=[self.hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
        outputs = tf.Variable(tf.compat.v1.random_normal(shape=[1]))  # 1 bias for the output

        # Declare model operations
        hidden_output = tf.nn.relu(tf.add(tf.matmul(self.x_data, inputs), biases))
        self.final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, hidden), outputs))

        # Declare loss function (MSE)
        self.loss = tf.reduce_mean(tf.square(self.y_target - self.final_output))

        # Declare optimizers
        self.opts = list()
        # opts.append(tf.train.GradientDescentOptimizer(0.005))
        self.opts.append(tf.compat.v1.train.AdamOptimizer(0.005))
        self.opts.append(tf.compat.v1.train.RMSPropOptimizer(0.005))

    def train(self, ):
        for optimizer in self.opts:
            print(str(optimizer))
            my_opt = optimizer
            train_step = my_opt.minimize(self.loss)
            # Initialize variables
            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)
            # Training loop
            loss_vec = []
            test_loss = []
            for epoch in range(self.epochs):
                # Split data into train/test = 80%/20%
                x_vals_train, x_vals_test, y_vals_train, y_vals_test = train_test_split(
                    self.x_vals, self.y_vals, test_size=0.2, random_state=42)
                x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
                x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

                rand_index = np.random.choice(len(x_vals_train), size=self.batch_size)
                rand_x = x_vals_train[rand_index]
                rand_y = np.transpose([y_vals_train[rand_index]])
                self.sess.run(train_step, feed_dict={self.x_data: rand_x, self.y_target: rand_y})

                temp_loss = self.sess.run(self.loss,
                                          feed_dict={self.x_data: rand_x, self.y_target: rand_y})
                loss_vec.append(np.sqrt(temp_loss))

                test_temp_loss = self.sess.run(self.loss, feed_dict={self.x_data: x_vals_test,
                                                                     self.y_target: np.transpose(
                                                                         [y_vals_test])})
                test_loss.append(np.sqrt(test_temp_loss))
                if epoch % self.batch_size == 0:
                    print(f'Generation: {epoch}. Loss = {temp_loss}')
                    predictions = self.final_output.eval(feed_dict={self.x_data: x_vals_test},
                                                         session=self.sess)
                    print(predictions)
