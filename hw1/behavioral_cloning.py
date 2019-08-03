import os
import pickle
import numpy as np
import tensorflow as tf
import argparse
import tf_util
import sklearn
from sklearn.model_selection import train_test_split

def load_data(data_file):
    data = pickle.load(open(data_file, 'rb'))

    assert data['observations'].shape[0] == data['actions'].shape[0]
    input_size = data['observations'].shape[-1]
    output_size = data['actions'].shape[-1]

    return data, input_size, output_size

class Model(object):
    def __init__(self, input_size, output_size, hidden_sizes, dropout_rate, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.input = tf.placeholder(tf.float32, shape=[None, input_size], name="input")
        self.label = tf.placeholder(tf.float32, [None, output_size], name="label")
        self.is_training_phase = tf.placeholder_with_default(False, shape=(), name="is_training_phase_input")

        self._build_graph()
        self._build_train_ops()

    def _build_graph(self):
        self.output = self.input
        if self.hidden_sizes:
            for i, hidden_size in enumerate(self.hidden_sizes):
                self.output = tf_util.dense(self.output, hidden_size, "hidden_{}".format(i))
                self.output = tf.nn.relu(self.output)
                if self.dropout_rate > 0:
                    self.output = tf_util.dropout(self.output, 1.0 - self.dropout_rate, self.is_training_phase)
        self.output = tf_util.dense(self.output, self.output_size, "last_layer")

    def _build_train_ops(self):
        self.loss = tf_util.mean(tf.square(self.output - self.label))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

def train(args):
    data_file = os.path.join('expert_data', args.env_name + ".pkl")
    data, input_size, output_size = load_data(data_file)
    print("For environment {}, input size: {}, output size: {}".format(args.env_name, input_size, output_size))
    
    X, y = data['observations'], data['actions']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print("Train size: {}, test size: {}".format(len(X_train), len(X_test)))

    learning_rate = 1e-3
    epochs = 10
    batch_size = 32

    model = Model(input_size, output_size, hidden_sizes=[128,64,32], dropout_rate=0, learning_rate=learning_rate)

    saver = tf.train.Saver()
    saver_path = os.path.join('models', args.env_name)

    steps = len(X_train) // batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=0)
            losses = []
            for j in range(steps):
                X_batch = X_train[j*batch_size:(j+1)*batch_size]
                y_batch = np.squeeze(y_train[j*batch_size:(j+1)*batch_size])

                _, loss = sess.run([model.train_op, model.loss], feed_dict={model.input: X_batch, model.label: y_batch, model.is_training_phase: True})
                losses.append(loss)
                if j % 100 == 0:
                    print("Epoch {}, step {}, loss: {}".format(i, j, np.mean(losses)))
                    losses = []
            if batch_size * steps != len(X_train):
                # train on remainder
                X_batch = X_train[steps * batch_size:]
                y_batch = np.squeeze(y_train[steps * batch_size:])
                _, loss = sess.run([model.train_op, model.loss], feed_dict={model.input: X_batch, model.label: y_batch, model.is_training_phase: True})

        test_loss = sess.run(model.loss, feed_dict={model.input: X_test, model.label: np.squeeze(y_test), model.is_training_phase: False})
        print("Test loss is: {}".format(test_loss))

        saver.save(sess, saver_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        pass

if __name__ == "__main__":
    main()
