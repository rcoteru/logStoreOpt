from __future__ import annotations
from copy import deepcopy

import pandas as pd
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Storage():

   A: list[dict[int: list]]   
 
   sn: list[float] 
   tn: list[float] 
   ln: list[bool] 

   sk: list[float]
   ak: list[float]
   pk: list[bool]

   t0: float

   def __init__(self, t0: float) -> None:
   
      self.A = []
      self.sn, self.tn, self.ln = [], [], []
      self.sk, self.ak, self.pk = [], [], []
      self.t0 = t0

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
      """ Add an item. """

      # TODO capacity check
      # TODO location type check
      
      # sanity checks
      assert(n in range(len(self.A)))
      assert(k in range(len(self.sk)))

      if k not in self.A[n]:
         self.A[n][k] = [1]
      else:

         # automatic stack choice
         if s is None:
            placed = False
            for i, s in enumerate(self.A[n][k]):
               if s < self.sk[k]:
                  self.A[n][k][s] += 1
                  placed = True
                  break
            if not placed:
               self.A[n][k].append(1)
            
         # if stack exists
         elif s < len(self.A[n][k]):

            if not (self.A[n][k] < self.sk[k]):
               raise ValueError("Stack {s} already full!")
            else:
               self.A[n][k][s] += 1

         # if stcks does not exist
         elif s >= len(self.A[n][k]):
            self.A[n][k].append(1)

      return n, k, s

   def count_items(self) -> np.ndarray:

      """ Count stacks per location/product. """

      stacks = np.zeros((N,K), dtype=int)
      for n in range(len(self.A)):
         for k in range(len(self.sk)):
            if k in self.A[n]:
               stacks[n,k] = np.array(self.A[n][k]).sum()
      return stacks

   def count_stacks(self, full: bool = None) -> np.ndarray:

      """ Count stacks per location/product. """

      if full is None:
         stacks = np.zeros((N,K), dtype=int)
         for n in range(len(self.A)):
            for k in range(len(self.sk)):
               if k in self.A[n]:
                  stacks[n,k] = len(self.A[n][k])
      elif full:
         stacks = np.zeros((N,K), dtype=int)
         for n in range(len(self.A)):
            for k in range(len(self.sk)):
               if k in self.A[n]:
                  stacks[n,k] = len([s for s in self.A[n][k] if s == self.sk[k]])
               else: 
                  stacks[n,k] = []
      else: 
         stacks = np.zeros((N,K), dtype=int)
         for n in range(len(self.A)):
            for k in range(len(self.sk)):
               if k in self.A[n]:
                  stacks[n,k] = len([s for s in self.A[n][k] if s < self.sk[k]])
               else: 
                  stacks[n,k] = []
      return stacks

   def locate_stacks(self, full: bool = True) -> np.ndarray:

      """ Return location of full / not full stacks. """

      stacks = np.zeros((N,K), dtype=object)
      if full:
         for n in range(len(self.A)):
            for k in range(len(self.sk)):
               if k in self.A[n]:
                  stacks[n,k] = [i for i, s in enumerate(self.A[n][k]) if s == self.sk[k]]
               else: 
                  stacks[n,k] = []
      else: 
         stacks = np.zeros((N,K), dtype=object)
         for n in range(len(self.A)):
            for k in range(len(self.sk)):
               if k in self.A[n]:
                  stacks[n,k] = [i for i, s in enumerate(self.A[n][k]) if s < self.sk[k]]
               else: 
                  stacks[n,k] = []
      return stacks

   def calc_surface(self) -> np.ndarray:

      """ Calculate used surface per location/product. """

      surf = np.zeros((N,K))
      for n in range(len(self.A)):
         for k in range(len(self.sk)):
            if k in self.A[n]:
               surf[n,k] = len(self.A[n][k])*self.sk[k]
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

   def location_info(self, aggregate: bool = True) -> pd.DataFrame:

      """ Returns a DataFrame with storage. """

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

      model = pd.DataFrame(columns=["ubicacion", "producto", "stack", "huecos", "nuevo"])

      for n in range(N):
         for k, mk in zip(ks, mks):

            # checkea si el articulo puede ir o no en baldas
            if self.ln[n] and not self.pk[k]:
               continue

            # add one possible location for each already existing stack
            if k in self.A[n]:
               for s in nf_stacks[n,k]:
                  sl = self.ak[k] - self.A[n][k][s]
                  model.loc[len(model)] = [n, k, s, sl, False]

            # add new stack
            for item in range(mk):
               s = int(stacks[n,k])+item
               sl = self.ak[k]
               model.loc[len(model)] = [n, k, s, sl, True]

      return model

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def find_positions(
      k: int, 
      sol: np.array, 
      model: pd.DataFrame, 
      A: Storage
      ) -> np.ndarray:

   """  """




   # capacity check
   # if sk[k] > sn[n] - csn[k]:
   #    model[ubicacion].isin(n) = False



   pass

def neighbor_generator(
      k: int, 
      sol: np.array, 
      model: pd.DataFrame, 
      A: Storage
      ) -> np.ndarray:


   yield 


def local_search(
      pedido: pd.DataFrame,
      model: pd.DataFrame, 
      A: Storage,
      max_neighbors: int =100
      ) -> np.ndarray:
   
   
   
   pass


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

   N = 5 # ubicaciones
   K = 3 # productos

   # crea el almacén
   A = Storage(t0=1)   

   # añade las ubicaciones
   for n in range(N):
      sn = 2 if n%2==0 else 3
      tn = 1 if n%2==1 else 1.5
      ln = False if n <= 2 else True
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

   print(A.location_info())
   print(A.create_model(pedido))