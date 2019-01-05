import os
import csv
from random import randint
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from difflib import get_close_matches
from tf_idf import TFIDF
ops.reset_default_graph()


# init data
tf_idf = TFIDF()
cities = list()
if os.path.isfile('kaz_cities.csv'):
    with open('kaz_cities.csv', 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            cities.append(row[-1].lower().strip())
tf_idf.random_walk()
P_matrix = tf_idf.P
vocabulary = tf_idf.tfidf.vocabulary_
x_vals = []
y_vals = []
for i, doc in enumerate(tf_idf.texts):
    splitted = doc.split()

    row = list()
    if tf_idf.target[i] > 0:
        for j, word in enumerate(splitted):
            close_matches = get_close_matches(word, cities, cutoff=0.9)
            # close_matches = get_close_matches(word, cities, cutoff=0.9)
            if close_matches:
                for x in splitted[:j][-3:] + splitted[j + 1:][:2]:
                    if x in vocabulary:
                        try:
                            row.append(P_matrix[vocabulary[x]][vocabulary[close_matches[-1]]])
                        except KeyError:
                            row.append(max(P_matrix[vocabulary[x]]))
                    else:
                        row.append(0.0)
                break
        if len(row) == 5:
            x_vals.append(row)
            y_vals.append(1.0)
    else:
        length = len(splitted)
        context = 5
        start_ind = randint(0, length - context)
        for x in splitted[start_ind:start_ind+context]:
            if x in vocabulary:
                row.append(max(P_matrix[vocabulary[x]]))
            else:
                row.append(0.0)
        if len(row) == 5:
            x_vals.append(row)
            y_vals.append(0.0)

x_vals = np.array(x_vals)
y_vals = np.array(y_vals)


# Create graph session
sess = tf.Session()
# make results reproducible
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)
# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)



# Declare batch size
batch_size = 50

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 5], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for both NN layers
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[5,hidden_layer_nodes])) # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # one biases for each hidden node
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1])) # hidden inputs -> 1 output
b2 = tf.Variable(tf.random_normal(shape=[1]))   # 1 bias for the output

# Declare model operations
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

# Declare loss function (MSE)
loss = tf.reduce_mean(tf.square(y_target - final_output))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []

for i in range(500):
    # Split data into train/test = 80%/20%
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))
    
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)%50==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))
        b = randint(0, 10)
        y = tf.nn.softmax(tf.matmul(x, W) + b))
        classification = sess.run(tf.argmax(y, 1), feed_dict={x: x_vals_test})
        print(classification)