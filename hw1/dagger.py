import os
import pickle
import numpy as np
import tensorflow as tf
import sklearn
import argparse
import load_policy
import tf_util
import gym_util
from behavioral_cloning import configs, Model
from behavioral_cloning import load_data, train_model

def dagger(args):
    data_file = os.path.join('expert_data', args.env_name + ".pkl")
    data, input_size, output_size = load_data(data_file)
    train_config = configs[args.env_name]
    model = Model(input_size, output_size, train_config)
    saver = tf.train.Saver()

    expert_policy_file = os.path.join('experts', args.env_name + ".pkl")
    expert_policy_fn = load_policy.load_policy(expert_policy_file)

    num_rollouts=10
    max_data = 80000

    X, y = data['observations'], data['actions']
    with tf.Session() as sess:
        saver.restore(sess, os.path.join('models', args.env_name))
        learner_policy_fn = tf_util.function([model.input], model.output)

        #train_config.epochs = 15
        for i in range(20):
            print("\nIteration {}".format(i))
            print("Running rollouts...")
            observations, actions = gym_util.run_gym(args.env_name, learner_policy_fn, expert_policy_fn=expert_policy_fn, num_rollouts=num_rollouts)
            print("Retrain model...")
            X = np.concatenate([X, observations])
            y = np.concatenate([y, actions])
            if len(X) > max_data:
                X, y = sklearn.utils.resample(X, y)
                X, y = X[:max_data], y[:max_data]
            train_model(sess, model, X, y, test_size=0.1, train_config=train_config, verbose=False)

        print("\nFinal run...")
        gym_util.run_gym(args.env_name, learner_policy_fn, expert_policy_fn=None, num_rollouts=10, verbose=False)
        saver.save(sess, os.path.join('models_dagger', args.env_name))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    args = parser.parse_args()

    dagger(args)

if __name__ == "__main__":
    main()