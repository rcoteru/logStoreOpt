from optimizer import StorageOptimizer
import numpy as np

# TODO LIST
# ===========================
# - treat costs as vectors
# ===========================

N = 7 # ubicaciones
K = 3 # productos
strategy = "explore" # "greedy" / "explore"

# crea el almacén
A = StorageOptimizer(t0=1)   

# añade las ubicaciones
for n in range(N):
   sn = 10 if n > N/2 else 20
   tn = np.arange(n).sum() + 1
   ln = True if n > N/2 else False
   A.add_location(sn=sn, tn=tn, ln=ln)

# añade los productos
for k in range(K):
   sk = 1
   ak = 2
   pk = False if k == 2 else True
   A.add_product(sk=sk, ak=ak, pk=pk)

# llenado inicial
objs = [1,1,2,0,0,0]
locs = [0,0,1,1,2,2]
for item, loc in zip(objs, locs):
   A.add_item(loc, item)

# pedido a optimizar
order = [1,1,0,1,2,1,0,2,1,0]

# ================================================================= #

print("\n" + "#"*60) # muestra un resumen del almacén

print("\nStorage:\n", A)
print("\nStorage summary:\n", A.location_summary())
print("\nProduct summary:\n", A.product_summary())

print("\nStor. Surface cost:", A.calc_surface().sum())
print("Stor. Access cost:", A.calc_access().sum())

print("\nOrder:\n", order)

sol = A.optimize_order(order, n_walkers=3, 
   strategy="greedy", random_state=0)
print("\nSolution:\n", sol)