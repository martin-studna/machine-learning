#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import math

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int,
                    help="Number of classes to use")
parser.add_argument("--iterations", default=10, type=int,
                    help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help="Learning rate")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def softmax(x, i):
    maximum = np.max(x)
    x -= maximum
    print(np.exp(x[i]) / (np.sum(np.exp(x))))
    return (np.exp(x[i]) / (np.sum(np.exp(x))))


def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(
        n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(
        size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

        permuted_data = train_data[permutation, :]
        permuted_target = train_target[permutation]

        n_batches = int(train_data.shape[0] / args.batch_size)
        data_batches = np.split(permuted_data, n_batches)
        target_batches = np.split(permuted_target, n_batches)
        print(weights)

        for i in range(n_batches):
            gradients = np.empty((args.batch_size, train_data.shape[1]))
            for j in range(args.batch_size):
                gradients[j, :] = (softmax(
                    np.dot(data_batches[i][j, :].T, weights[i][j]), i) - target_batches[i][j]) * data_batches[i][j, :]
            average = np.mean(gradients, axis=0)

            weights[i] -= args.learning_rate * average

        # TODO: After the SGD iteration, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_accuracy, train_loss, test_accuracy, test_loss = None, None, None, None

        train_loss = 0
        for i in range(train_data.shape[0]):
            train_loss = train_loss + (train_target[i] * np.log(softmax(np.dot(train_data[i].T, weights))) +
                                       (1 - train_target[i]) * np.log(1 - softmax(np.dot(train_data[i].T, weights))))
        train_loss = -1 * train_loss / train_data.shape[0]

        test_loss = 0
        for i in range(test_data.shape[0]):
            test_loss = test_loss + (test_target[i] * np.log(softmax(np.dot(test_data[i].T, weights))) +
                                     (1 - test_target[i]) * np.log(1 - softmax(np.dot(test_data[i].T, weights))))
        test_loss = -1 * test_loss / test_data.shape[0]

        train_loss = sklearn.metrics.log_loss(
            train_target, np.dot(train_data, weights))

        train_accuracy = 0
        for i in range(train_data.shape[0]):
            p_C_1 = softmax(np.dot(train_data[i].T, weights))
            p_C_0 = 1 - p_C_1
            train_accuracy = train_accuracy + np.log(p_C_1 / p_C_0)

        test_accuracy = 0
        for i in range(train_data.shape[0]):
            p_C_1 = softmax(np.dot(test_data[i].T, weights))
            p_C_0 = 1 - p_C_1
            test_accuracy = np.log(p_C_1 / p_C_0)

        print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            iteration + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(args)
    print("Learned weights:", *(" ".join([" "] + ["{:.2f}".format(w)
                                                  for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
