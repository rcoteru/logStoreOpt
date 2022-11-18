import numpy as np

rng = np.random.default_rng(seed=0)

N = 5                                   # n ubicaciones
t = np.arange(N)                        # tiempo acceso
c = 7*np.ones(N)                        # capacidad

M = 3                                   # n objetos pedido
a = 3*np.ones(M) + np.arange(M)         # areas

def first_fit(N: int, M: int):

    S = np.zeros((N, M), dtype=int)
    f = np.zeros(N, dtype=float)

    for j in np.argsort(a)[::-1]:   # prioritize larger items

        np.where(c - f > a[j])

        i = np.argmax(c - f >= a[j])
        S[i, j] = 1
        f[i] += a[j]

    return S

S = first_fit(N, M)
cost = np.matmul(t.transpose(), S).sum()
valid = np.matmul(S, a) <= c

print(t)
print(cost)
print(valid)
print(S)