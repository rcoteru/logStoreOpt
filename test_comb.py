from copy import deepcopy
import numpy as np

N = 5   # ubicaciones
L = 2   # tipos de ubicacion
M = 5   # objetos en el pedido
K = 2   # tipos de objeto (productos)

t0 = 1 # tiempo de acceso extra por nivel
p0 = 2 # limite de peso para las baldas

tn = np.ones(N) # tiempos de acceso base por ubicacion
sn = np.zeros(N) # capacidad maxima por ubicación

pk = np.ones(K) # peso por producto
pk[1] = 3
sk = np.ones(K)   # superficie por producto
ak = np.ones(K)*2 # stack limit por producto

G = np.eye(M,K) # asignacion de productos (one-hot)
G[2,0] = 1

T = np.zeros(N, dtype=bool) # denota si es balda (True) o suelo (False)
T[2:] = True

A = np.zeros((N,K), dtype=object)
for n in range(N):
   for k in range(K):
      A[n,k] = []

A[0,0].append(2)
A[0,0].append(1)
A[2,1].append(1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def count_stacks(A: np.ndarray):   
   """ Count stacks in each location. """
   def _sum_lenghts(x):
      return np.array([len(q) for q in x])
   return np.apply_along_axis(_sum_lenghts, 1, A)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def used_surface(A: np.ndarray, sk: np.ndarray):   
   """ Calculate used surface in each location. """
   def _sum_areas(x):
      return np.multiply(np.array([len(q) for q in x]), sk)
   return np.apply_along_axis(_sum_areas, 1, A)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# NOTA: el tiempo de acceso y el de extracción son equivalentes, 
# puesto que siempre se accede al ultimo piso del stack

def access_time(A: np.ndarray, tn: np.ndarray, t0: float):   
   """ Calculate access time in each location. """

   base_times = np.repeat(np.expand_dims(tn, axis=1), K, axis=1)
   def _sum_items(x):
      return np.array([len(q) for q in x])
   base_times = np.multiply(np.apply_along_axis(_sum_items, 1, A), base_times)

   def _sum_times(x):
      return np.array([t0*np.arange(len(q)).sum() for q in x])
   
   return np.apply_along_axis(_sum_times, 1, A) + base_times

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def stack_state(A: np.ndarray, ak: np.ndarray, full=False):
   """ Returns location of full or empty stacks stacks."""
   def _sum_lenghts(x):
      result = np.zeros_like(x, dtype=object)
      for i, q in enumerate(x):
         result[i] = []         
         for j, s in enumerate(q):
            if full and s == ak[i]:
               result[i].append(j)
            if not full and s < ak[i]:
               result[i].append(j)
      return result
   return np.apply_along_axis(_sum_lenghts, 1, A)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def create_problem(
      A: np.ndarray, 
      G: np.ndarray, 
      ak: np.ndarray, 
      T: np.ndarray,
      pk: np.ndarray,
      p0: float
      ):

   """ Creates three arrays with all the possible locations for new items. """

   qnk = stack_state(A, ak, full=False)       # unoccupied stacks
   Qnk = count_stacks(A)  # total stacks
   prods = G.sum(axis=0).astype(int)

   problem = []
   for k, m_k in enumerate(prods):
      for n in range(N):

         # check de peso para las baldas
         if (not T[n]) or (pk[k] < p0):

            # add new stack
            for item in range(m_k):
               s = int(Qnk[n,k])+item
               sl = ak[k]
               problem.append([n, k, s, sl, item])

            # add one possible location for each already existing stack
            for s in qnk[n,k]:
               sl = ak[k] - A[n,k][s]
               problem.append([n, k, s, sl, 0])

   return np.array(problem, dtype=int)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def update_storage(A: np.ndarray, prob:np.ndarray, sol: np.ndarray) -> np.ndarray:

   """ Returns a copy of the storage updated with the given solution. """

   As = deepcopy(A)
   updates = np.argwhere(sol != 0)
   for u in updates:
      n = int(prob[u,0])
      k = int(prob[u,1])
      q = int(prob[u,2])
      if q >= len(As[n,k]):
         As[n,k].append(1)
      else:
         print(As[n,k])
         As[n,k][q] += 1

   return As

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def find_available_positions(k: int, A: np.ndarray, 
         prob: np.ndarray, sol: np.ndarray) -> int:

   """ Finds available positions for adding a new object to the solution. """

   As = update_storage(A, prob, sol)

   # check stack constraints
   stack_check = sol < prob[:,3]

   # check location capacity constraints
   capacity_check = True 

   # TODO: check duplicity (equivalent solutions) constraints
   # tl;dr: ensure only one non-existent stack is accesible
   duplicity_check = True

   # compute valid positions
   return np.logical_and(capacity_check, duplicity_check)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def create_initial_solution(
      A: np.ndarray,
      G: np.ndarray,
      prob: np.ndarray,
      random_state: int = 0
      ) -> np.ndarray:

   """ Creates a viable solution for the problem. """

   rng = np.random.default_rng(seed=random_state)

   sol = np.zeros(prob.shape[0], dtype=int)
   prods = G.sum(axis=0).astype(int)

   for k, mk in enumerate(prods):
      for _ in range(mk):
         vpos = find_available_positions(k, A, prob, sol)
         cpos = rng.choice(np.argwhere(vpos))
         sol[cpos] += 1

   return sol

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def generate_neighborhood(
      A: np.ndarray,
      G: np.ndarray,
      prob: np.ndarray,
      sol: np.ndarray,
      ) -> np.ndarray:

   """ Generate an array with the first neighbors of the given solution. """

   spots = np.argwhere(sol > 0)

   # TODO
   prods = G.sum(axis=0).astype(int)
   for k, mk in enumerate(prods):
      for _ in range(mk):
         pass

   sols = np.zeros(1)

   return sols

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def local_search(
      A: np.ndarray,
      G: np.ndarray,
      prob: np.ndarray,
      init_sol: np.ndarray,
      max_its: int = 10,
      ) -> np.ndarray:

   """ Find a locally optimal solution using local search. """

   og_surf = used_surface(A, sk).sum()
   og_time = access_time(A, tn, t0).sum()
  
   def _costs(sol: np.ndarray):
      scost = og_surf - used_surface(update_storage(A, prob, sol), sk).sum()
      tcost = og_time - access_time(update_storage(A, prob, sol), tn, t0).sum()
      return scost, tcost

   curr_sol = init_sol
   curr_cost = _costs(init_sol).sum()
   
   for it in range(1, max_its):

      # TODO: check if neighbor has already been explored
      neighs = generate_neighborhood(A, G, prob, curr_sol)

      costs = np.apply_along_axis(_costs, neighs, axis=1)
      cost = costs.sum(axis=1)

      id_min = np.argmin(cost)[0]
      bcost, bsol = cost[id_min], neighs[id_min]

      if bcost < curr_cost:
         curr_cost = bcost
         curr_sol = bsol
      else:
         break
   
   return curr_sol, curr_cost

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("pedido:\n", G.sum(axis=1))
print("almacen:\n",A)
print("stacks por ubi / producto:\n", count_stacks(A))
print("superf por ubi / producto:\n", used_surface(A, sk))
print("acc_tm por ubi / producto:\n", access_time(A, tn, t0))
print("stacks completos por ubi / producto:\n", stack_state(A, ak, full=True))
print("stacks incompletos por ubi / producto:\n", stack_state(A, ak, full=False))
print("------------------------------------")
prob = create_problem(A,G,ak,T,pk,p0)
print("problem:\n", prob)
init_sol = create_initial_solution(A,G,prob,0)
print("\ninitial solution:", init_sol)
og_surface = used_surface(A, sk).sum()
nw_surface = used_surface(update_storage(A, prob, init_sol), sk).sum()
print("surface cost:", nw_surface - og_surface)
og_time = access_time(A, tn, t0).sum()
nw_time = access_time(update_storage(A, prob, init_sol), tn, t0).sum()
print("time cost:", nw_time - og_time)
print("------------------------------------")