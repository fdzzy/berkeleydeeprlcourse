import os
import pickle
import numpy as np
import tensorflow as tf
import argparse
import tf_util
import gym_util
import sklearn
from sklearn.model_selection import train_test_split

class TrainConfig(object):
    def __init__(self, hidden_sizes=[128,64,32], dropout_rate=0, learning_rate=1e-3, epochs=10, batch_size=32):
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

configs = {
    "Ant-v2": TrainConfig(),
    "HalfCheetah-v2": TrainConfig(),
    "Hopper-v2": TrainConfig(),
    "Humanoid-v2": TrainConfig(hidden_sizes=[256,128,64,32],epochs=100),
    "Reacher-v2": TrainConfig(),
    "Walker2d-v2": TrainConfig(),
}

def load_data(data_file):
    data = pickle.load(open(data_file, 'rb'))

    assert data['observations'].shape[0] == data['actions'].shape[0]
    input_size = data['observations'].shape[-1]
    output_size = data['actions'].shape[-1]

    return data, input_size, output_size

class Model(object):
    def __init__(self, input_size, output_size, train_config):
        self.input_size = input_size
        self.output_size = output_size
        self.train_config = train_config

        self.input = tf.placeholder(tf.float32, shape=[None, input_size], name="input")
        self.label = tf.placeholder(tf.float32, [None, output_size], name="label")
        self.is_training_phase = tf.placeholder_with_default(False, shape=(), name="is_training_phase_input")

        self._build_graph()
        self._build_train_ops()

    def _build_graph(self):
        self.output = self.input
        if self.train_config.hidden_sizes:
            for i, hidden_size in enumerate(self.train_config.hidden_sizes):
                self.output = tf_util.dense(self.output, hidden_size, "hidden_{}".format(i))
                self.output = tf.nn.relu(self.output)
                if self.train_config.dropout_rate > 0:
                    self.output = tf_util.dropout(self.output, 1.0 - self.train_config.dropout_rate, self.is_training_phase)
        self.output = tf_util.dense(self.output, self.output_size, "last_layer")

    def _build_train_ops(self):
        self.loss = tf_util.mean(tf.square(self.output - self.label))
        optimizer = tf.train.AdamOptimizer(self.train_config.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

def train_model(sess, model, X, y, test_size, train_config, verbose=True):
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    else:
        X_train, y_train, X_test, y_test = X, y, None, None
    print("Train size: {}, test size: {}".format(len(X_train), 0 if X_test is None else len(X_test)))

    batch_size = train_config.batch_size
    steps = len(X_train) // batch_size
    for i in range(train_config.epochs):
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        losses = []
        for j in range(steps):
            X_batch = X_train[j*batch_size:(j+1)*batch_size]
            y_batch = np.squeeze(y_train[j*batch_size:(j+1)*batch_size])

            _, loss = sess.run([model.train_op, model.loss], feed_dict={model.input: X_batch, model.label: y_batch, model.is_training_phase: True})
            losses.append(loss)
            if j % 100 == 0 and verbose:
                print("Epoch {}, step {}, loss: {}".format(i, j, np.mean(losses)))
                losses = []
        if batch_size * steps != len(X_train):
            # train on remainder
            X_batch = X_train[steps * batch_size:]
            y_batch = np.squeeze(y_train[steps * batch_size:])
            _, loss = sess.run([model.train_op, model.loss], feed_dict={model.input: X_batch, model.label: y_batch, model.is_training_phase: True})
    
    if X_test is not None:
        test_loss = sess.run(model.loss, feed_dict={model.input: X_test, model.label: np.squeeze(y_test), model.is_training_phase: False})
        print("Test loss is: {}".format(test_loss))

def train(args):
    data_file = os.path.join('expert_data', args.env_name + ".pkl")
    data, input_size, output_size = load_data(data_file)
    print("For environment {}, input size: {}, output size: {}".format(args.env_name, input_size, output_size))
    
    X, y = data['observations'], data['actions']
    train_config = configs[args.env_name]
    model = Model(input_size, output_size, train_config)
    saver = tf.train.Saver()
    saver_path = os.path.join('models', args.env_name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess, model, X, y, test_size=0.1, train_config=train_config, verbose=True)
        saver.save(sess, saver_path)

def run(args):
    data_file = os.path.join('expert_data', args.env_name + ".pkl")
    data, input_size, output_size = load_data(data_file)
    model = Model(input_size, output_size, configs[args.env_name])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, os.path.join('models', args.env_name))
        policy_fn = tf_util.function([model.input], model.output)

        #test_in = data['observations'][0]
        #action = policy_fn(test_in[None, :])
        gym_util.run_gym(args.env_name, policy_fn, num_rollouts=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'run':
        run(args)

if __name__ == "__main__":
    main()
