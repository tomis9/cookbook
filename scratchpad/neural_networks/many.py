import matplotlib.pyplot as plt
import numpy as np


def generate_sample_dataset(K, N, M, n, act_H, act_O):
    X = np.random.rand(n, K)
    W_t = np.random.randint(-5, 5, (K, N))
    W1_t = np.random.randint(-5, 5, (N, M))
    T = []
    for i in range(n):
        h = act_H(np.matmul(W_t.transpose(), X[i].reshape(2, 1)))
        t = act_O(np.matmul(W1_t.transpose(), h))
        T.append(t.ravel())
    T = np.array(T)
    T = T / max(T)
    return X, T, W_t, W1_t


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feed_forward(x, t, W, W1, act_H, act_O):
    K = len(x)
    x = x.reshape(K, 1)
    M = len(t)
    t = t.reshape(M, 1)
    u = np.matmul(W.transpose(), x)
    h = act_H(u)
    u1 = np.matmul(W1.transpose(), h)
    y = act_O(u1)
    return y, h


def backprop(y, t, h, x, W1):
    K = len(x)
    x = x.reshape(K, 1)
    M = len(t)
    t = t.reshape(M, 1)
    EI1 = (y - t) * y * (1 - y)
    dW1 = np.matmul(h, EI1.transpose())
    EI = np.matmul(W1, EI1) * (h * (1 - h))
    dW = np.matmul(x, EI.transpose())
    return dW, dW1


def train(X, T, N, eta, epochs):
    K = len(X[0])
    M = len(T[0])
    n = len(X)
    W = np.random.rand(K, N)
    W1 = np.random.rand(N, M)

    for i in range(epochs):
        dW_sum, dW1_sum = 0, 0
        ys = []
        for i in range(n):
            y, h = feed_forward(x=X[i], t=T[i], W=W, W1=W1)
            ys.append(y.ravel())
            dW, dW1 = backprop(y=y, t=T[i], h=h, x=X[i], W1=W1)
            dW_sum += dW
            dW1_sum += dW1

        W -= dW_sum * eta
        W1 -= dW1_sum * eta
        Y = np.array(ys)
        e = sum(sum((Y - T) ** 2))
        print(e)
    return W, W1


def correlation(X, T, W, W1):
    if len(T[0]) > 1:
        raise ValueError("correlation coefficient is valid only for M=1")
    pred = []
    for i in range(len(X)):
        y, h = feed_forward(X[i], T[i], W, W1)
        pred.append(y.ravel())

    t = T.ravel()
    y = np.array(pred).ravel()
    print(np.corrcoef(y, t))

    plt.plot(y, t, 'ro')
    plt.ylabel('some numbers')
    plt.show()


# meta
eta = 0.01
epochs = 400
N = 10  # number of neurons in hidden layer

X, T, W_t, W1_t = generate_sample_dataset(K=2, N=N, M=1, n=1000)
W, W1 = train(X=X, T=T, N=N, eta=eta, epochs=epochs)

correlation(X, T, W, W1)

# https://medium.com/@curiousily/tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b
