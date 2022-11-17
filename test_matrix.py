from scipy.special import factorial
import numpy as np

rng = np.random.default_rng(seed=0)


N = 5   # ubicaciones
L = 2   # tipos de ubicacion
M = 2   # objetos en el pedido
K = 2   # tipos de objeto (productos)

t0 = 1 # tiempo de acceso extra por nivel
p0 = 1 # limite de peso para las baldas


tn = np.zeros(N) # tiempos de acceso base por ubicacion
sn = np.zeros(N) # capacidad maxima por ubicación

pk = np.zeros(K) # peso por producto
sk = np.zeros(K) # superficie por producto
ak = np.zeros(K) # stack limit por producto

Gmk = np.zeros((M,K)) # asignacion de productos (one-hot)
Unm = np.zeros((N,M)) # asignación de ubicaciones por objeto (one-hot)

qnk = np.zeros((N,K)) # stacks por producto y ubicación
pnk = np.zeros((N,K)) # stack incompleto por producto y ubicacion

Tnl = np.zeros((N,L)) # denota el tipo de cada almacenamiento (one-hot)

# checks

def storage_limit_check():
    return np.all(np.multiply(np.ceil(np.divide(np.matmul(Unm, Gmk) + pnk, ak, axis=1)) + qnk, sk).sum(axis=1) < sn)

# goal functions

def storage_cost():
   return np.multiply(np.ceil(np.divide(np.matmul(Unm, Gmk) + pnk, ak, axis=1)) + qnk, sk).sum()

def access_cost():
   return np.multiply(np.multiply(np.floor(np.divide(np.matmul(Unm, Gmk) + pnk, ak, axis=1)) + qnk, factorial(ak)) + \
        factorial(np.mod(np.matmul(Unm, Gmk) + pnk, ak, axis=1)), tn, axis=0)*t0