"""
This script provides an example use case for the optimizer.

@author Raúl Coterillo
@version December 2022
"""

from optimizer import StorageOptimizer
import numpy as np
import time

N = 7                   # locations
K = 4                   # products
strategy = "explore"    # "greedy" / "explore"
goal_prios = [1,1]      # surface priority, time priority
prod_prios = [1,1,1,1]  # priorities for the different products

PLOT = True             # plot the optimization procedure?
SEED = 3424             # seed for the random number generators

# start the optimizer
A = StorageOptimizer(t0=5) 

# add the locations
for n in range(N):
   sn = 10 if n > N/2 else 20
   tn = 1 + np.arange(n).sum()
   ln = (n > N/2)
   A.add_location(sn=sn, tn=tn, ln=ln)

# add the products
for k in range(K):
   sk = 1
   ak = 2
   pk = (k != 2)
   A.add_product(sk=sk, ak=ak, pk=pk)

# add some items
objs = [1,1,2,0,0,0]
locs = [0,0,1,1,2,2]
for item, loc in zip(objs, locs):
   A.add_item(loc, item)

# order to optimize
order = [1,1,0,1,2,1,0,2,1,0,3,3]

# ================================================================= #

print("\nStorage:\n", A)
print("\nStorage summary:\n", A.location_summary())
print("\nProduct summary:\n", A.product_summary())

print("\nOrder:\n", order)

stime = time.perf_counter()
sol, model, hist = A.optimize_order(order, n_walkers=10, 
   strategy=strategy, random_state=SEED, goal_weights=goal_prios, 
   prod_weights=prod_prios)
etime = time.perf_counter()
print("\nSolution:\n", A.display_solution(sol, model))

b_stats = A.calc_stats()
s_stats = A.calc_solution_stats(sol, model)
new_A = A.add_solution(sol, model)
nb_stats = new_A.calc_stats()

spacing = 20
print("")
print(f"{'':{spacing}}{'surface (m2)':>{spacing}}{'access time (min)':>{spacing}}")
pgoals = A.calc_goal_functions(b_stats)
print(f"{'Previous goals:':>{spacing}}{pgoals[0]:>{spacing}.3f}{pgoals[1]:>{spacing}.3f}")
sgoals = A.calc_goal_functions(s_stats)
print(f"{'Solution goals:':>{spacing}}{sgoals[0]:>{spacing}.3f}{sgoals[1]:>{spacing}.3f}")
nbgoals = A.calc_goal_functions(nb_stats)
print(f"{'New goals:':>{spacing}}{nbgoals[0]:>{spacing}.3f}{nbgoals[1]:>{spacing}.3f}")
print("")

# average access time 
print(f"\n{'':>{spacing}}{'~~~~~~ mean access time (min) ~~~~~~':^{spacing*2}}\n")
print(f"{'product':>{spacing}}{'solution':>{spacing}}{'total':>{spacing}}")
for k in range(nb_stats.shape[0]):
   sol_avg_time = 0
   tot_avg_time = 0
   if nb_stats[k,2] != 0:
      sol_avg_time = s_stats[k,1]/s_stats[k,2]
      tot_avg_time = nb_stats[k,1]/nb_stats[k,2]
   print(f"{k:{spacing}}{sol_avg_time:>{spacing}.3f}{tot_avg_time:>{spacing}.3f}")

print("\nTotal neighbors visited:", hist["neigh_size"].sum())
print(f"\nTime elapsed: {etime-stime} s\n")

# print("Training history:\n", hist)

if PLOT:
   import matplotlib.pyplot as plt
   plt.rcParams['figure.constrained_layout.use'] = True
   plt.figure(figsize=[6,6], )
   for rep, rep_df in hist.reset_index().groupby("rep"):
      plt.plot(rep_df["iteration"], rep_df["scalar_cost"], label=f"rep_{rep}")
   plt.xlabel("LS Iteration")
   plt.ylabel("Cost Function")
   plt.legend()
   plt.grid(True)
   plt.show()
