import numpy as np
from pgauss import solve as p_solve
from cgauss import solve as c_solve
import sys


def main():
    if len(sys.argv) > 1:
        s = int(sys.argv[1])
    else:
        s = 3
    rng = np.random.default_rng()
    a = rng.integers(-10, 10, size=(s, s)).astype(np.float32)
    x = rng.integers(-10, 10, size=s).astype(np.float32)
    b = a @ x
    print("Base")
    print("A =", a, sep="\n")
    print("x =", x, sep="\n")
    print("A @ x = b =", b, sep="\n")
    print("Python implementation")
    try:
        x_p = p_solve(a, b)
    except ValueError as e:
        print(str(e))
    else:
        print(np.isclose(x_p, x))
        print(x_p)
    print("C implementation")
    try:
        x_c = c_solve(a, b)
    except ValueError as e:
        print(str(e))
    else:
        print(np.isclose(x_c, x))
        print(x_c)
    print("NumPy implementation")
    x_n = np.linalg.solve(a, b)
    print(np.isclose(x_n, x))
    print(x_n)


if __name__ == "__main__":
    main()
