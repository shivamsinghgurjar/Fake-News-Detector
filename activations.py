import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


def plot_activation_functions(save_path=None):
    x = np.linspace(-10, 10, 100)

    plt.figure(figsize=(10, 6))
    plt.plot(x, sigmoid(x), label="Sigmoid")
    plt.plot(x, relu(x), label="ReLU")
    plt.plot(x, tanh(x), label="Tanh")

    plt.title("Activation Functions")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    x = np.array([-2, -1, 0, 1, 2])
    print("Sigmoid:", sigmoid(x))
    print("ReLU:", relu(x))
    print("Tanh:", tanh(x))
    print("Softmax:", softmax(np.array([2.0, 1.0, 0.1])))
    plot_activation_functions()
