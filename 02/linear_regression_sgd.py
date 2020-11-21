#!/usr/bin/env python3

# Team:
# 2f67b427-a885-11e7-a937-00505601122b
# b030d249-e9cb-11e9-9ce9-00505601122b
# 3351ff04-3f62-11e9-b0fd-00505601122b

import argparse
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int,
                    help="Number of SGD iterations over the data")
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True,
                    nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_regression(
        n_samples=args.data_size, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    data = np.c_[data, np.ones(data.shape[0])]

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size,
                                                                        random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1])
    #weights = np.zeros(train_data.shape[1])

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # A gradient for example (x_i, t_i) is `(x_i^T weights - t_i) * x_i`,
        # and the SGD update is `weights = weights - args.learning_rate * gradient`.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        permuted_data = train_data[permutation, :]
        permuted_target = train_target[permutation]

        n_batches = int(train_data.shape[0] / args.batch_size)
        data_batches = np.split(permuted_data, n_batches)
        target_batches = np.split(permuted_target, n_batches)
        n_weights = train_data.shape[1]

        averages = np.empty((n_batches, n_weights))
        for i in range(n_batches):
            gradients = np.empty((data_batches[i].shape[0], n_weights))
            for j in range(data_batches[i].shape[0]):
                gradients[j, :] = (
                    data_batches[i][j, :].T @ weights - target_batches[i][j]) * data_batches[i][j, :]
            averages[i, :] = np.mean(gradients, axis=0)
            weights = weights - args.learning_rate * averages[i, :]

        # TODO: Append current RMSE on train/test to train_rmses/test_rmses.
        train_rmses.append(np.sqrt(sklearn.metrics.mean_squared_error(
            train_target, np.dot(train_data, weights))))
        test_rmses.append(np.sqrt(sklearn.metrics.mean_squared_error(
            test_target, np.dot(test_data, weights))))

    # TODO: Compute into `explicit_rmse` test data RMSE when
    # fitting `sklearn.linear_model.LinearRegression` on train_data.
    reg = LinearRegression()
    reg.fit(train_data, train_target)
    explicit_rmse = np.sqrt(sklearn.metrics.mean_squared_error(
        reg.predict(test_data), test_target))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        if args.plot is True:
            plt.show()
        else:
            plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return test_rmses[-1], explicit_rmse
