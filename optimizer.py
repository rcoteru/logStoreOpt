"""
This module provides the main class for the optimization.

The StorageOptimizer class contains all the data structures
and methods to perform the optimization.

@author Raúl Coterillo
@version December 2022
"""

from __future__ import annotations

from functools import partial
import multiprocessing as mp
from copy import deepcopy
import json

import pandas as pd
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class StorageOptimizer():

    A: list[dict[int: list]]   # main data structure
    
    sn: list[float]      # Surface of each location
    tn: list[float]      # Access time of each location
    ln: list[bool]       # Is the location a shelve?

    sk: list[float]      # Surface of each product
    ak: list[float]      # Stack limit of each product
    pk: list[bool]       # Can the product be stored in shelves?

    t0: float            # Extra access time per stack level above ground

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Class methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, t0: float) -> None:
        """ Class constructor. """
        self.A = []
        self.sn, self.tn, self.ln = [], [], []
        self.sk, self.ak, self.pk = [], [], []
        self.t0 = t0


    def __str__(self) -> str:
        """ String representation of the class. """
        return json.dumps(self.A, sort_keys=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Location-related methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def add_location(self, sn: float, tn: float, ln: bool = False) -> int:
        """ Add a location. """
        idx = len(self.A)
        self.A.append({})
        self.tn.append(tn) 
        self.sn.append(sn)
        self.ln.append(ln)
        return idx


    def location_summary(self) -> pd.DataFrame:

        """ Returns a DataFrame with storage summary information. """

        items  = self.count_items()
        stacks = self.count_stacks()
        surf   = self.calc_surface()
        acc    = self.calc_access()

        df = pd.DataFrame(index=np.arange(len(self.A)))
        df["items"] = items.sum(axis=1)
        df["stacks"] = stacks.sum(axis=1)
        df["superf_max"] = self.sn
        df["superf_occ"] = surf.sum(axis=1)
        df["superf_pct"] = surf.sum(axis=1)/self.sn*100
        df["tmp_acc_base"] = self.tn
        df["tmp_acc_items"] = acc.sum(axis=1)
        df["balda"] = self.ln

        return df

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Product-related methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def add_product(self, sk: float, ak: int, pk: bool) -> int:
        """ Add a product. """
        idx = len(self.sk)
        self.sk.append(sk)
        self.ak.append(ak)
        self.pk.append(pk)
        return idx
    

    def product_summary(self) -> pd.DataFrame:

        """ Returns a DataFrame with storage summary information. """

        items  = self.count_items()
        stacks = self.count_stacks()
        surf   = self.calc_surface()

        df = pd.DataFrame(index=np.arange(len(self.sk)))
        df["total_items"] = items.sum(axis=0)
        df["total_stcks"] = stacks.sum(axis=0)
        df["surface_occ"] = self.sk
        df["total_surf"] = surf.sum(axis=0)
        df["stack_limit"] = self.ak
        df["shelve_comp"] = self.pk

        return df

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Item-related methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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

            # TODO needs fixing, generates too many stacks
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
                if self.A[n][k][s] >= self.ak[k]:
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Stack-related methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
                        stacks[n,k] = [i for i, s in enumerate(self.A[n][k]) if s == self.ak[k]]
                    else: 
                        stacks[n,k] = []
        else: 
            for n in range(N):
                for k in range(K):
                    if k in self.A[n]:
                        stacks[n,k] = [i for i, s in enumerate(self.A[n][k]) if s < self.ak[k]]
                    else: 
                        stacks[n,k] = []
                    
        return stacks

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Surface cost functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Access time cost functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def calc_access(self) -> np.ndarray:

        """ Calculate access times per location/product. """

        N, K = len(self.A), len(self.sk)
        acc = np.zeros((N,K))
        for n in range(len(self.A)):
            for k in range(len(self.sk)):
                if k in self.A[n]:
                    # base access time
                    acc[n,k] += len(self.A[n][k])*self.tn[n]
                    # height access time
                    acc[n,k] += np.array([np.arange(s).sum()*self.t0 for s in self.A[n][k]]).sum()

        return acc

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Total cost functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def calc_stats(self) -> np.ndarray:

        """ Calculate needed stats for the cost function. """

        N, K = len(self.A), len(self.sk)
        
        # surface, total access time, and item count per product
        stats = np.zeros((K, 3))
        
        for n in range(N):
            for k in range(K):
                if k in self.A[n]:
                    # surface
                    stats[k,0] += len(self.A[n][k])*self.sk[k]
                    # base access time
                    stats[k,1] += len(self.A[n][k])*self.tn[n]
                    # height access time
                    stats[k,1] += np.array([np.arange(s).sum()*self.t0 for s in self.A[n][k]]).sum()
                    # item count
                    stats[k,2] += np.array(self.A[n][k]).sum()
        
        return stats

    def calc_solution_stats(self, sol: np.ndarray, model: pd.DataFrame) -> np.ndarray:

        """ Calculate needed partial (solution) stats for the cost function. """

        K = len(self.sk)
        sol_stats = np.zeros((K, 3))
        
        # filter out unused information
        df = model.copy()
        df.insert(0, "amount", sol)
        df = df[df["amount"] > 0]

        # surface
        sdf = df[df["height"] == 0]
        sdf = sdf[["product", "height"]].groupby("product").count().reset_index()
        sdf.rename(columns={"height": "number_of_new_stacks"}, inplace=True)
        def _surf(x: np.array):
            return x["number_of_new_stacks"]*self.sk[x["product"]]
        sdf["surf"] = sdf.apply(_surf, axis=1)
        for k, k_row in sdf.iterrows():
            sol_stats[k, 0] += k_row["surf"]

        # access time
        def _time(x: np.array):
            if x["height"] == 0:
                return self.tn[x["location"]] + (x["amount"]-1)*self.t0
            else:    
                return x["amount"]*self.t0
        
        df["time"] = df[["location", "amount", "height"]].apply(_time, axis=1)
        time = df[["time", "product"]].groupby("product").sum()
        for k, k_row in time.iterrows():
            sol_stats[k, 1] += k_row["time"]

        # item count 
        counts = df[["amount", "product"]].groupby("product").sum()
        for k, k_row in counts.iterrows():
            sol_stats[k, 2] += k_row["amount"]

        return sol_stats

    def calc_goal_functions(self,
            stats: np.ndarray, 
            prod_weights: np.ndarray = None
            ) -> np.ndarray:

        """ Calculate the goal functions from previously gathered stats. """


        base_weights = self.count_items().sum(axis=0)
        base_weights = base_weights + stats[:,2]

        if prod_weights is not None:
            prod_weights = np.multiply(base_weights, prod_weights)
        else:
            prod_weights = base_weights

        gfs = np.zeros(2)

        # surface goal function
        gfs[0] = stats[:,0].sum()

        # average access time 
        prod_weights = np.array(prod_weights)
        for k in range(stats.shape[0]):
            if stats[k,2] != 0:
                gfs[1] += stats[k,1]/stats[k,2]*prod_weights[k]

        return gfs

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Model-related methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def create_model(self, order: np.ndarray) -> pd.DataFrame:

        """ Possible locations for the items in an order. """

        order = np.array(order)
        ks, mks = np.unique(order, return_counts=True)

        stacks = self.count_stacks()
        nf_stacks = self.locate_stacks(full=False)

        model = pd.DataFrame(columns=["location", "product", "stack", "height", "slots"])

        N = len(self.A)
        for n in range(N):
            for k, mk in zip(ks, mks):

                # check if product is shelve-compatible
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


    def find_valid_locations(self, k: int, sol: np.array, model: pd.DataFrame) -> np.ndarray:

        """ Finds available positions for adding a new object to the solution. """

        # NOTE: shelve compatibility is checked at model creation

        # check product goes there
        product_check = model["product"].isin([k]).values

        # check stack height constraints
        stack_check = (sol < model["slots"]).values

        # check location capacity constraints
        B = self.add_solution(sol, model)
        free = [n[0] for n in np.argwhere(self.sn - B.calc_surface().sum(axis=1) >= self.sk[k])]
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


    def create_initial_solution(self,
            order: np.ndarray,
            model: pd.DataFrame, 
            random_state: int = 0
            ) -> np.ndarray:

        """ Creates a viable solution for the problem. """

        rng = np.random.default_rng(seed=random_state)
        sol = np.zeros(len(model), dtype=int)

        for k in rng.permutation(order):
            vpos = self.find_valid_locations(k, sol, model)
            cpos = rng.choice(np.argwhere(vpos))
            sol[cpos] += 1

        return sol


    def add_solution(self, sol: np.ndarray, model: pd.DataFrame) -> StorageOptimizer:

        """ Return a copy of the storage after adding the items of the solution. """

        A: StorageOptimizer = deepcopy(self)
        locs = [l[0] for l in np.argwhere(sol != 0)]

        for l in locs:
            ldata = model.iloc[l]
            n = int(ldata["location"])
            k = int(ldata["product"])
            s = int(ldata["stack"])
            for _ in range(sol[l]):
                A.add_item(n, k, s)

        return A

    @staticmethod
    def display_solution(sol: np.ndarray, model: pd.DataFrame):
   
        """ Returns the solution in a human-readable format."""

        df = model.copy()
        df.insert(0, "amount", sol)
        df = df[df["amount"] > 0]
        return df

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # Optimization-related methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def neighbor_generator(self, sol: np.array, model: pd.DataFrame) -> np.ndarray:

        "Generates neighbors of a given solution."

        locs = [l[0] for l in np.argwhere(sol != 0)]

        for l in locs:
            ldata = model.iloc[l]
            k = ldata["product"]
            
            lsol = sol.copy()
            lsol[l] -= 1
            npos = self.find_valid_locations(k, sol, model)
            npos[l] = False
            choices = np.argwhere(npos)
            for c in choices:
                neigh = lsol.copy()
                neigh[c] += 1
                yield neigh


    def local_search(self,
        sol: np.array, 
        model: pd.DataFrame, 
        strategy: str = "greedy",
        max_neighbors: int = 100,
        max_iterations: int = 10,
        prod_weights: np.array = None,
        goal_weights: np.array = None,
        cost_scaling: list[np.array] = None,
        ) -> np.ndarray:
   
        """ Optimizes a given solution using local search. """

        if strategy not in ["greedy", "explore"]:
            raise NotImplementedError(f"Unknown strategy '{strategy}'.")

        b_stats = self.calc_stats()
        def _scalar_goals(goals: np.array):
            if cost_scaling is None:
                return np.average(goals, weights=goal_weights)
            else:
                means, stds = cost_scaling[0], cost_scaling[1]
                goals = np.divide((goals-means),stds)
                return np.average(goals, weights=goal_weights)
 
        c_sol = sol
        c_stats = self.calc_solution_stats(c_sol, model)
        c_vgoal = self.calc_goal_functions(b_stats + c_stats, prod_weights=prod_weights)
        c_sgoal = _scalar_goals(c_vgoal)

        hist = pd.DataFrame(columns=["iteration", "surf_cost", "time_cost", "scalar_cost", "neigh_size", "improved"])
        hist.loc[len(hist)] = [0, c_vgoal[0], c_vgoal[1], c_sgoal, 0, False]

        for it in range(1, max_iterations+1):

            # greedy approach
            if strategy == "greedy":

                improved = False
                for idx_neigh, neigh in enumerate(self.neighbor_generator(c_sol, model)):
   
                    n_stats = self.calc_solution_stats(neigh, model)
                    n_vgoal = self.calc_goal_functions(b_stats + n_stats, prod_weights=prod_weights)
                    n_sgoal = _scalar_goals(n_vgoal)

                    if n_sgoal < c_sgoal or idx_neigh == max_neighbors:
                        improved = True
                        c_vgoal = n_vgoal
                        c_sgoal = n_sgoal
                        c_sol = neigh
                        break

            # exploration approach
            elif strategy == "explore":
                
                neighs, vgoals, sgoals = [], [], []
                improved = False

                for idx_neigh, neigh in enumerate(self.neighbor_generator(c_sol, model)):

                    n_stats = self.calc_solution_stats(neigh, model)
                    n_vgoal = self.calc_goal_functions(b_stats + n_stats, prod_weights=prod_weights)
                    n_sgoal = _scalar_goals(n_vgoal)

                    neighs.append(neigh)
                    vgoals.append(n_vgoal)
                    sgoals.append(n_sgoal)
                    if idx_neigh == max_neighbors:
                        break
                    
                neighs, vgoals, sgoals = np.array(neighs), np.array(vgoals), np.array(sgoals)
                min_loc = np.argmin(sgoals)
                if sgoals[min_loc] < c_sgoal:
                    improved = True
                    c_sol = neighs[min_loc]
                    c_vgoal = vgoals[min_loc]
                    c_sgoal = sgoals[min_loc]
                    
            hist.loc[len(hist)] = [it, c_vgoal[0], c_vgoal[1], c_sgoal, idx_neigh+1, improved]
            if not improved:
                break

        hist["iteration"] = hist["iteration"].astype(int)
        hist["neigh_size"] = hist["neigh_size"].astype(int)
        hist.set_index("iteration", inplace=True)

        return c_sol, c_sgoal, hist

    def optimize_order(self, 
            order: np.array, 
            n_walkers: int = 10,
            strategy: str = "greedy",
            max_neighbors: int = 100,
            max_iterations: int = 10,
            prod_weights: np.array = None,
            goal_weights: np.array = None,
            nsols_norm: int = 100,
            nprocs: int = mp.cpu_count(),
            random_state: int = 0
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        """ Optimizes a given incoming order with parallel local search. """

        # input checks ~~~~~~~~~~~~~~~~~~~~~~

        if (prod_weights is not None) and (len(prod_weights) != len(self.sk)):
            raise ValueError("Array 'prod_weights' should be of length " + \
                            f"{len(self.sk)}, not {len(prod_weights)}!")
        
        if (goal_weights is not None) and (len(goal_weights) != 2):
            raise ValueError("Array 'goal_weights' should be of length " + \
                            f"2, not {len(goal_weights)}!")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        model = self.create_model(order)

        # create random solutions for normalization
        ss = np.random.SeedSequence(random_state)
        seeds = list(ss.spawn(nsols_norm))

        norm_gf = []
        b_stats = self.calc_stats()
        for s in seeds:
            try:
                nsol = self.create_initial_solution(order, model, random_state=s)
                n_stats = self.calc_solution_stats(nsol, model)
                n_vgoal = self.calc_goal_functions(b_stats + n_stats, 
                        prod_weights=prod_weights)
                norm_gf.append(n_vgoal)
            except:
                pass
        norm_gf = np.array(norm_gf)
        cost_scaling = [norm_gf.mean(axis=0), norm_gf.std(axis=0)]
        
        # create seeds for the different walkers
        ss = np.random.SeedSequence(random_state)
        seeds = list(ss.spawn(n_walkers))

        # create the problem model
        model = self.create_model(order)

        # local search function call
        B = deepcopy(self)
        ls_call = partial(ls, A=B, model=model, order=order, 
            strategy=strategy, cost_scaling=cost_scaling, 
            goal_weights=goal_weights, prod_weights=prod_weights,
            max_iterations=max_iterations, max_neighbors=max_neighbors)

        # run the local search walkers in parallel
        with mp.Pool(processes=nprocs) as pool:
            results = pool.map(ls_call, seeds)

        sols = np.array([x[0] for x in results])
        costs = np.array([x[1] for x in results])
        
        hists = []
        for i, x in enumerate(results):
            x[2]["rep"] = i
            hists.append(x[2].reset_index())
        hists = pd.concat(hists)
        idx = pd.MultiIndex.from_frame(hists[["rep", "iteration"]])
        hists.set_index(idx, inplace=True)
        hists.drop(columns=["rep", "iteration"], inplace=True)

        sols, idx, counts = np.unique(sols, axis=0, return_counts=True, return_index=True)    
        costs = costs[idx]
        bsol_idx = np.argmin(costs)

        return sols[bsol_idx,:], model, hists

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def ls(
    seed: int, 
    A: StorageOptimizer, model: pd.DataFrame, order: pd.DataFrame, 
    strategy: str, cost_scaling: list[np.ndarray], 
    goal_weights: np.ndarray, prod_weights: np.array,
    max_neighbors: int, max_iterations: int):

    """ Function call for parallel exeution. """

    isol = A.create_initial_solution(order, model, random_state=seed)
    bsol, cost, hist = A.local_search(isol, model, strategy=strategy,
            max_iterations=max_iterations, max_neighbors=max_neighbors,
            cost_scaling=cost_scaling, goal_weights=goal_weights, 
            prod_weights=prod_weights)
    return bsol, cost, hist 