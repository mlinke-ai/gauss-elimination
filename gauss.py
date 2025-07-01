import numpy as np


def solve(a, b):
    # get size of the system
    n = a.shape[0]
    # create a copy for in-place computations
    m = a.copy()
    # create a pivot lookup table
    p = np.arange(n)
    for i in range(n):
        # find index of pivot element and swap with current in lookup table
        s = np.argmax(np.abs(m[i:, i])) + i
        p[[i, s]] = p[[s, i]]
        if m[p[i], i] == 0.0:
            raise ValueError("Pivoting failed")
        else:
            for j in range(i + 1, n):
                # compute columns of L
                m[p[j], i] /= m[p[i], i]
                for k in range(i + 1, n):
                    # compute rows of R
                    m[p[j], k] -= m[p[j], i] * m[p[i], k]
    # forward substitution
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[p[i]] - np.sum(m[p[i], :i] * y[:i])
    # backward substitution
    x = np.zeros_like(b)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.sum(m[p[i], i + 1:] * x[i + 1:])) / m[p[i], i]
    return x
