from __future__ import annotations
from copy import deepcopy

import pandas as pd
import numpy as np
import json

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Storage():

   A: list[dict[int: list]]   # main data structure
 
   sn: list[float]      # Surface of each location
   tn: list[float]      # Access time of each location
   ln: list[bool]       # Is the location a shelve?

   sk: list[float]      # Surface of each product
   ak: list[float]      # Stack limit of each product
   pk: list[bool]       # Can the product be stored in shelves?

   t0: float            # Extra access time per stack level above ground

   def __init__(self, t0: float) -> None:
   
      self.A = []
      self.sn, self.tn, self.ln = [], [], []
      self.sk, self.ak, self.pk = [], [], []
      self.t0 = t0

   def __str__(self) -> str:
      return json.dumps(self.A, sort_keys=True)

   def add_location(self, sn: float, tn: float, ln: bool = False) -> int:
      """ Add a location. """
      idx = len(self.A)
      self.A.append({})
      self.tn.append(tn) 
      self.sn.append(sn)
      self.ln.append(ln)
      return idx

   def add_product(self, sk: float, ak: int, pk: bool) -> int:
      """ Add a product. """
      idx = len(self.sk)
      self.sk.append(sk)
      self.ak.append(ak)
      self.pk.append(pk)
      return idx

   def add_item(self, n: int, k: int, s: int = None):

      """ Add an item to the storage. """
      
      # Check the location exists
      if n not in range(len(self.A)):
         raise ValueError(f"Location {n} does not exist!")
      
      # Check the product exists
      if k not in range(len(self.sk)):
         raise ValueError(f"Product {k} does not exist!")
      
      # Check shelve compatibility
      if self.ln[n] and not self.pk[k]:
         raise ValueError(f"Product {k} can't be stored in shelves!")

      
      if k not in self.A[n]:
         
         # Capacity check
         if self.sk[k] > self.sn[n] - self.calc_location_surface(n):
               raise ValueError("Location {n} already full!")
         
         # Create new stack list for product in location
         self.A[n][k] = [1]
      
      else:

         # Choose a stack automatically
         if s is None:

            # Try to place it in an existing stack.
            placed = False
            for i, s in enumerate(self.A[n][k]):
               if s < self.sk[k]:
                  self.A[n][k][s] += 1
                  placed = True
                  break

            if not placed:

               # Capacity check
               if self.sk[k] > self.sn[n] - self.calc_location_surface(n):
                  raise ValueError("Location {n} already full!")
               
               # Add new stack to the location
               self.A[n][k].append(1)
            
         # Save to chosen stack
         elif s < len(self.A[n][k]):
         
            # Stack height check
            if self.A[n][k][s] >= self.sk[k]:
               raise ValueError("Stack {s} already full!")

            # Add item to the stack
            self.A[n][k][s] += 1

         # Create new stack
         elif s >= len(self.A[n][k]):

            # Capacity check
            if self.sk[k] > self.sn[n] - self.calc_location_surface(n):
               raise ValueError("Location {n} already full!")

            # Add new stack to the location
            self.A[n][k].append(1)

      return n, k, s

   def count_items(self) -> np.ndarray:

      """ Count stacks per location/product. """

      N, K = len(self.A), len(self.sk)
      stacks = np.zeros((N,K), dtype=int)
      for n in range(N):
         for k in range(K):
            if k in self.A[n]:
               stacks[n,k] = np.array(self.A[n][k]).sum()
      return stacks

   def count_stacks(self, full: bool = None) -> np.ndarray:

      """ Count stacks per location/product. """

      N, K = len(self.A), len(self.sk)
      stacks = np.zeros((N,K), dtype=int)
      if full is None:
         for n in range(N):
            for k in range(K):
               if k in self.A[n]:
                  stacks[n,k] = len(self.A[n][k])
      elif full:
         for n in range(N):
            for k in range(K):
               if k in self.A[n]:
                  stacks[n,k] = len([s for s in self.A[n][k] if s == self.sk[k]])
               else: 
                  stacks[n,k] = []
      else: 
         for n in range(N):
            for k in range(K):
               if k in self.A[n]:
                  stacks[n,k] = len([s for s in self.A[n][k] if s < self.sk[k]])
               else: 
                  stacks[n,k] = []
      return stacks

   def locate_stacks(self, full: bool = True) -> np.ndarray:

      """ Return location of full / not full stacks. """

      N, K = len(self.A), len(self.sk)
      stacks = np.zeros((N,K), dtype=object)
      if full:
         for n in range(N):
            for k in range(K):
               if k in self.A[n]:
                  stacks[n,k] = [i for i, s in enumerate(self.A[n][k]) if s == self.sk[k]]
               else: 
                  stacks[n,k] = []
      else: 
         for n in range(N):
            for k in range(K):
               if k in self.A[n]:
                  stacks[n,k] = [i for i, s in enumerate(self.A[n][k]) if s < self.sk[k]]
               else: 
                  stacks[n,k] = []
      return stacks

   def calc_surface(self) -> np.ndarray:

      """ Calculate used surface per location/product. """

      N, K = len(self.A), len(self.sk)
      surf = np.zeros((N,K))
      for n in range(N):
         for k in range(K):
            if k in self.A[n]:
               surf[n,k] = len(self.A[n][k])*self.sk[k]
      return surf

   def calc_location_surface(self, n: int) -> float:

      """ Calculate used surface at location n. """

      surf, K = 0.0, len(self.sk)
      for k in range(K):
         if k in self.A[n]:
            surf += len(self.A[n][k])*self.sk[k]
      return surf

   def calc_solution_surface(self, sol: np.ndarray, model: pd.DataFrame):

      """ Calculate extra surface added by a solution. """

      df = model.copy()
      df.insert(0, "amount", sol)
      df = df[df["amount"] > 0]
      df = df[df["height"] == 0]
      df = df[["product", "height"]].groupby("product").count()

      surf = 0
      for k, mk in df.iterrows():
         surf += mk["height"]*self.sk[k]

      return surf

   def calc_access(self) -> np.ndarray:

      """ Calculate access times per location/product. """

      acc = np.zeros((N,K))
      for n in range(len(self.A)):
         for k in range(len(self.sk)):
            if k in self.A[n]:
               acc[n,k] = np.array(self.A[n][k]).sum()*self.sn[n] + \
                           np.arange(len(self.A[n][k])).sum()*self.t0
      return acc

   def calc_solution_access(self, sol: np.ndarray, model: pd.DataFrame):

      """ Calculate extra access_time added by a solution. """

      df = model.copy()
      df.insert(0, "amount", sol)
      df = df[df["amount"] > 0]

      
      def _loc_time(x: np.array):
         return self.tn[x]

      def _stack_time(x: np.array):
         return np.arange(x[1]).sum()*self.t0
      
      df["location"].apply(_loc_time)*df["height"]
      df[["amount", "height"]].apply(_stack_time)

      #np.apply_along_axis(_custom_func, df["height"].values)
      

      # surf = 0
      # for k, mk in df.iterrows():
      #    surf += mk*self.sk[k]

      return df

   def location_info(self, aggregate: bool = True) -> pd.DataFrame:

      """ Returns a DataFrame with storage summary information. """

      items  = self.count_items()
      stacks = self.count_stacks()
      surf   = self.calc_surface()
      acc    = self.calc_access()

      if aggregate:

         df = pd.DataFrame(index=np.arange(len(self.A)))

         df["items"] = items.sum(axis=1)
         df["stacks"] = stacks.sum(axis=1)
         df["superf_max"] = self.sn
         df["superf_occ"] = surf.sum(axis=1)
         df["superf_pct"] = surf.sum(axis=1)/self.sn*100
         df["tmp_acc_base"] = acc.sum(axis=1)
         df["tmp_acc_total"] = acc.sum(axis=1)
         df["balda"] = self.ln

      else:

         index = pd.MultiIndex.from_product(
               [np.arange(len(self.A)), np.arange(len(self.sk))], 
               names=["loc", "prod"])
         df = pd.DataFrame(index=index)

         df["items"] = items.flatten()
         df["stacks"] = stacks.flatten()
         df["superficie"] = surf.flatten()
         df["tiempo_acc"] = acc.flatten()

      return df

   def create_model(self, pedido: np.ndarray) -> pd.DataFrame:

      """ Possible locations for the items in an order. """

      pedido = np.array(pedido)
      ks, mks = np.unique(pedido, return_counts=True)

      stacks = self.count_stacks()
      nf_stacks = self.locate_stacks(full=False)

      model = pd.DataFrame(columns=["location", "product", "stack", "height", "slots"])

      for n in range(N):
         for k, mk in zip(ks, mks):

            # checkea si el articulo puede ir o no en baldas
            if self.ln[n] and not self.pk[k]:
               continue

            # add one possible location for each already existing stack
            if k in self.A[n]:
               for s in nf_stacks[n,k]:
                  hg = self.A[n][k][s]
                  sl = self.ak[k] - hg
                  model.loc[len(model)] = [n, k, s, hg, sl]

            # add new stack
            for item in range(mk):
               s = int(stacks[n,k])+item
               sl = self.ak[k]
               model.loc[len(model)] = [n, k, s, 0, sl]

      return model

   def add_solution(self, sol: np.ndarray, model: pd.DataFrame):

      A: Storage = deepcopy(self)
      locs = [l[0] for l in np.argwhere(sol != 0)]

      for l in locs:
         ldata = model.iloc[l]
         n = int(ldata["location"])
         k = int(ldata["product"])
         s = int(ldata["stack"])
         for _ in range(sol[l]):
            A.add_item(n, k, s)

      return A

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def find_positions(
      k: int, 
      sol: np.array, 
      model: pd.DataFrame, 
      A: Storage
      ) -> np.ndarray:

   """ Finds available positions for adding a new object to the solution. """

   # NOTE: shelve compatibility is checked at model creation

   # check product goes there
   product_check = model["product"].isin([k]).values

   # check stack height constraints
   stack_check = (sol < model["slots"]).values

   # check location capacity constraints
   A = A.add_solution(sol, model)
   free = [n[0] for n in np.argwhere(A.sn[n] - A.calc_surface().sum(axis=1) >= A.sk[k])]
   capacity_check = model["location"].isin(free).values
   
   # check duplicity (equivalent solutions) constraints
   new_stacks = model[model["height"]==0].sort_values(["location", "product", "stack"])\
      .drop(columns=["slots"])
   new_stacks["index"] = new_stacks.index
   first_stacks = new_stacks.groupby(["location", "product"]).first()
   
   duplicity_check = np.zeros(len(model), dtype=bool)
   duplicity_check[model[~(model["height"]==0)].index.values] = True
   duplicity_check[first_stacks["index"].values] = True
   # TL;DR: ensure only one non-existent stack is accesible

   return product_check & stack_check & capacity_check & duplicity_check

def create_initial_solution(
      pedido: np.ndarray,
      model: pd.DataFrame, 
      A: Storage,
      random_state: int = 0
      ) -> np.ndarray:

   """ Creates a viable solution for the problem. """

   rng = np.random.default_rng(seed=random_state)
   sol = np.zeros(len(model), dtype=int)

   for k in pedido:
      vpos = find_positions(k, sol, model, A)
      # TODO change to a more reasonable heuristic
      cpos = rng.choice(np.argwhere(vpos))
      sol[cpos] += 1

   return sol

def display_solution(sol: np.ndarray, model: pd.DataFrame):
   
   """ Returns the solution in a human-readable format."""

   df = model.copy()
   df.insert(0, "amount", sol)
   df = df[df["amount"] > 0]
   return df

def neighbor_generator(
      sol: np.array, 
      model: pd.DataFrame, 
      A: Storage
      ) -> np.ndarray:

   "Generates neighbors of a given solution."

   locs = [l[0] for l in np.argwhere(sol != 0)]

   for l in locs:
      ldata = model.iloc[l]
      k = ldata["product"]
      
      lsol = sol.copy()
      lsol[l] -= 1
      npos = find_positions(k, sol, model, A)
      choices = np.argwhere(npos)
      for c in choices:
         neigh = lsol.copy()
         neigh[c] += 1
         yield neigh

def local_search(
      sol: np.array, 
      model: pd.DataFrame, 
      A: Storage,
      strategy: str = "greedy", # TODO
      max_neighbors: int = 100,
      max_iterations: int = 10
      ) -> np.ndarray:
   

   def _cost_func(scost, tcost):
      return scost + tcost

   c_sol = sol
   c_scost = A.calc_solution_surface(c_sol, model)
   c_tcost = 0#A.calc_solution_access(c_sol)
   c_cost = _cost_func(c_scost, c_tcost)

   for it in range(max_iterations):

      # greedy approach
      for i, neigh in enumerate(neighbor_generator(c_sol, model, A)):
         
         n_scost = A.calc_solution_surface(neigh, model)
         n_tcost = 0#A.calc_solution_access(neigh)
         n_cost = _cost_func(n_scost, n_tcost)

         if n_cost < c_cost or i == max_neighbors:
            c_sol = neigh
            c_cost = n_cost
            print(n_scost, n_tcost)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

   N = 7 # ubicaciones
   K = 3 # productos

   # crea el almacén
   A = Storage(t0=1)   

   # añade las ubicaciones
   for n in range(N):
      sn = 2 if n%2==0 else 4
      tn = 1 if n%2==1 else 1.5
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

   pedido = [1,1,0,1,2]

   print("Storage:\n", A)
   print("\nStorage_status:\n", A.location_info())
   print("\nPedido:\n", pedido)
   model = A.create_model(pedido)
   print("\nModelo:\n", model)
   sol = create_initial_solution(pedido, model, A, random_state=0)
   print("\nRandom initial solution (array):\n", sol)
   print("\nRandom initial solution (human):\n", display_solution(sol, model))
   print("\nStorage cost: ", A.calc_solution_surface(sol, model))
   #print("Access cost: ", A.calc_solution_access(sol, model))
   local_search(sol, model, A)