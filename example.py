"""
This script provides an example use case for the optimizer.

@author Raúl Coterillo
@version November 2022
"""

from optimizer import StorageOptimizer
import numpy as np
import time

N = 7                   # locations
K = 3                   # products
strategy = "explore"    # "greedy" / "explore"
priority = [1,1]        # [surface priority, time priority]

# start the optimizer
A = StorageOptimizer(t0=5)   

# add the locations
for n in range(N):
   sn = 10 if n > N/2 else 20
   tn = 1 + np.arange(n).sum()
   ln = True if n > N/2 else False
   A.add_location(sn=sn, tn=tn, ln=ln)

# add the products
for k in range(K):
   sk = 1
   ak = 2
   pk = False if k == 2 else True
   A.add_product(sk=sk, ak=ak, pk=pk)

# add some items
objs = [1,1,2,0,0,0]
locs = [0,0,1,1,2,2]
for item, loc in zip(objs, locs):
   A.add_item(loc, item)

# order to optimize
order = [1,1,0,1,2,1,0,2,1,0]

# ================================================================= #

print("\nStorage:\n", A)
print("\nStorage summary:\n", A.location_summary())
print("\nProduct summary:\n", A.product_summary())

print("\nOrder:\n", order)

stime = time.perf_counter()
sol, model, hist = A.optimize_order(order, n_walkers=10, 
   strategy=strategy, random_state=0, weights=priority)
etime = time.perf_counter()
print("\nSolution:\n", A.display_solution(sol, model))

print("\nPrevious costs:", A.calc_cost())
print("Solution cost:", A.calc_solution_cost(sol, model))
print(" Updated cost:", A.add_solution(sol, model).calc_cost())

print("\nTotal neighbors visited:", hist["neigh_size"].sum())
print(f"\nTime elapsed: {etime-stime} s\n")

# print("Training history:\n", hist)