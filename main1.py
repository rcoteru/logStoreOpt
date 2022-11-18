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

print("\n" + "#"*60) # muestra un resumen del pedido y el modelo generado

print("\nOrder:\n", order)
model = A.create_model(order)
print("\nModel:\n", model)

print("\n" + "#"*60) # muestra la primera solucion generada aleatoriamente

sol = A.create_initial_solution(order, model, random_state=1)

print("\nRandom initial solution (array):\n", sol)
print("\nRandom initial solution (human):\n", A.display_solution(sol, model))

print("\nStor. Surface cost:", A.calc_surface().sum())
print("Diff. Surface cost:", A.calc_solution_surface(sol, model))
print(" New  Surface cost:", A.add_solution(sol, model).calc_surface().sum())

print("\nStor. Access cost:", A.calc_access().sum())
print("Diff. Access cost:", A.calc_solution_access(sol, model))
print(" New  Access cost:", A.add_solution(sol, model).calc_access().sum())

print("\n" + "#"*60) # muestra el proceso de optimización

bsol, cost, hist = A.local_search(sol, model, strategy=strategy)
print("\nTraining history:\n", hist)

print("\n" + "#"*60) # muestra la solución optimizada

print("\nOptimized solution (array):\n", bsol)
print("\nOptimized solution (human):\n", A.display_solution(bsol, model))

print("\nStor. Surface cost:", A.calc_surface().sum())
print("Diff. Surface cost:", A.calc_solution_surface(bsol, model))
print(" New  Surface cost:", A.add_solution(bsol, model).calc_surface().sum())

print("\nStor. Access cost:", A.calc_access().sum())
print("Diff. Access cost:", A.calc_solution_access(bsol, model))
print(" New  Access cost:", A.add_solution(bsol, model).calc_access().sum())

# sols, costs, hists, dists = A.optimize_order(order, nreps=10)
# print(sols)
# print(costs)
# print(dists)