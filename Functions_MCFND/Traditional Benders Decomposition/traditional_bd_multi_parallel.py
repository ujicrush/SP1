import numpy as np
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import warnings
import time
import math
from scipy.stats import norm
from types import SimpleNamespace
from multiprocessing import Pool
import os

from general_library import *
from types_file import *

# ===========================================================================
# ========== where you need to substitute for different benchmarks ==========

config = Config()
config.ver_prefix = ""
config.jobid = 00000
config.use_vi_1 = True
config.ntries = 50
config.n_clusters = 1
config.n = 42
config.TIME_LIMIT = 7200
config.demand_lb = 0.99
config.use_z0 = True
config.corr_pcnt = 0
config.n_clusters_ratio = 1.0
config.isBenchmark = False
config.nC = 2
config.randomSeed = 1
config.method = "scp_slim"
config.demand_ub = 1.01
config.use_partial_cuts = True
config.nR = 5
config.useMosek = True
config.k_nn = 6
config.slim_repeats = 4
config.rootCuts = 0
config.method_kelley = "scp_slim"
config.verbose_logging = 0
config.Rx = 6
config.k = 42
config.use_file_demands = False
config.is_magnanti_wong_cut = False
config.R_div = 3
config.round_z0 = True
config.Ry = 5
config.gamma = 1.01
config.m = 7
config.use_vi_2 = True
config.use_avg_scenario = True
config.use_si_vi = True
config.R_div_kelley = 4

cmcmd_prob = SimpleNamespace()

cmcmd_prob.n_nodes = 7
cmcmd_prob.n_edges = 42
cmcmd_prob.n_commodities = 2
cmcmd_prob.n_days = 5
cmcmd_prob.n_clusters = 1
cmcmd_prob.k = 42

cmcmd_prob.A = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
])

cmcmd_prob.b = np.array([
    [-71.0, 22.0], [7.0, 22.0], [13.0, 23.0], [5.0, 8.0], [19.0, -89.0], [20.0, 9.0], [7.0, 5.0],
    [-105.0, 7.0], [21.0, 24.0], [16.0, 25.0], [12.0, 20.0], [25.0, -123.0], [7.0, 25.0], [24.0, 22.0],
    [-86.0, 11.0], [11.0, 11.0], [7.0, 19.0], [14.0, 21.0], [16.0, -84.0], [16.0, 10.0], [22.0, 12.0],
    [-82.0, 7.0], [7.0, 11.0], [16.0, 19.0], [17.0, 15.0], [9.0, -78.0], [12.0, 13.0], [21.0, 13.0],
    [-90.0, 16.0], [15.0, 21.0], [11.0, 18.0], [17.0, 12.0], [12.0, -90.0], [24.0, 9.0], [11.0, 14.0]
]).reshape((5, 7, 2)).transpose(1, 2, 0)  # shape: (n_nodes, n_commodities, n_days)

cmcmd_prob.c = np.array([
    30.116295556199972, 0.0, 27.237110271064804, 0.0, 54.00345441488728, 35.109400381007994, 37.61450361098948, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 42.240858707997255, 0.0, 0.0, 79.05525433522197, 0.0, 0.0, 41.277681081948984, 0.0, 0.0, 0.0, 0.0, 
    25.84988554940333, 48.416591078542304, 0.0, 0.0, 24.06169353262146, 0.0, 0.0, 0.0, 0.0, 44.18269066857184, 34.515052364894565, 
    0.0, 0.0, 0.0, 0.0, 76.2509104040624
])

cmcmd_prob.d = np.array([
    0.09964422662066935, 1.0674774816021837, 1.682070392914671, 0.8432586520373289, 0.9693889616489197, 0.5405677446200995, 
    0.09964422662066935, 1.1379427279833385, 1.7462515756692043, 0.9266592184287601, 0.9690044708003377, 0.48327552777603, 
    1.0674774816021837, 1.1379427279833385, 0.6228827159731242, 1.2629816627706754, 0.806022833085596, 1.6073367461624197, 
    1.682070392914671, 1.7462515756692043, 0.6228827159731242, 1.8306126225729193, 1.1596509421789953, 2.2226048231941413, 
    0.8432586520373289, 0.9266592184287601, 1.2629816627706754, 1.8306126225729193, 1.631192425444467, 1.037178281153327, 
    0.9693889616489197, 0.9690044708003377, 0.806022833085596, 1.1596509421789953, 1.631192425444467, 1.4149250175634531, 
    0.5405677446200995, 0.48327552777603, 1.6073367461624197, 2.2226048231941413, 1.037178281153327, 1.4149250175634531
])

cmcmd_prob.u = np.array([
    55.0, 40.0, 45.0, 56.0, 60.0, 60.0, 49.0, 56.0, 46.0, 55.0, 45.0, 45.0, 57.0, 40.0, 51.0, 39.0, 45.0, 40.0, 52.0, 64.0, 60.0, 
    58.0, 39.0, 64.0, 61.0, 45.0, 60.0, 52.0, 56.0, 56.0, 47.0, 57.0, 47.0, 47.0, 55.0, 63.0, 46.0, 51.0, 57.0, 51.0, 47.0, 59.0
])

cmcmd_prob.lb = np.array([
    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 
    0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0
])

cmcmd_prob.ub = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
])

cmcmd_prob.gamma = 1.01
cmcmd_prob.sampling_rate = 3.0

cmcmd_prob.edge_map = {
    5: (1, 6, 6), 16: (3, 5, 19), 20: (4, 2, 23), 35: (6, 5, 40), 12: (2, 7, 14), 24: (4, 7, 28), 28: (5, 4, 32), 8: (2, 3, 10), 
    17: (3, 6, 20), 30: (5, 7, 35), 1: (1, 2, 2), 19: (4, 1, 22), 22: (4, 5, 26), 23: (4, 6, 27), 6: (1, 7, 7), 32: (6, 2, 37), 
    11: (2, 6, 13), 36: (6, 7, 42), 37: (7, 1, 43), 9: (2, 4, 11), 31: (6, 1, 36), 41: (7, 5, 47), 14: (3, 2, 16), 3: (1, 4, 4), 
    39: (7, 3, 45), 29: (5, 6, 34), 7: (2, 1, 8), 25: (5, 1, 29), 33: (6, 3, 38), 40: (7, 4, 46), 34: (6, 4, 39), 4: (1, 5, 5), 
    13: (3, 1, 15), 15: (3, 4, 18), 2: (1, 3, 3), 10: (2, 5, 12), 18: (3, 7, 21), 21: (4, 3, 24), 26: (5, 2, 30), 27: (5, 3, 31), 
    38: (7, 2, 44), 42: (7, 6, 48)
}

cmcmd_prob.outgoing_edges = {
    5: [25, 26, 27, 28, 29, 30], 4: [19, 20, 21, 22, 23, 24], 6: [31, 32, 33, 34, 35, 36], 7: [37, 38, 39, 40, 41, 42], 
    2: [7, 8, 9, 10, 11, 12], 3: [13, 14, 15, 16, 17, 18], 1: [1, 2, 3, 4, 5, 6]
}

cmcmd_prob.incoming_edges = {
    5: [4, 10, 16, 22, 35, 41], 4: [3, 9, 15, 28, 34, 40], 6: [5, 11, 17, 23, 29, 42], 7: [6, 12, 18, 24, 30, 36], 
    2: [1, 14, 20, 26, 32, 38], 3: [2, 8, 21, 27, 33, 39], 1: [7, 13, 19, 25, 31, 37]
}

cmcmd_prob.old_to_new_map = {
    5: 4, 16: 14, 20: 17, 35: 30, 12: 10, 24: 21, 28: 24, 8: 7, 30: 26, 37: 32, 23: 20, 19: 16, 22: 19, 32: 28, 6: 5, 43: 37, 
    11: 9, 36: 31, 44: 38, 31: 27, 45: 39, 47: 41, 14: 12, 3: 2, 39: 34, 29: 25, 7: 6, 46: 40, 40: 35, 48: 42, 34: 29, 4: 3, 
    13: 11, 15: 13, 2: 1, 10: 8, 18: 15, 21: 18, 26: 22, 27: 23, 38: 33, 42: 36
}

# ===========================================================================
# ===========================================================================

A, b, c, d, u, k, gamma, z0, ub, m, n, nC, nR = cmcmd_prob.A, cmcmd_prob.b, cmcmd_prob.c, cmcmd_prob.d, cmcmd_prob.u, cmcmd_prob.k, cmcmd_prob.gamma, cmcmd_prob.lb,cmcmd_prob.ub, cmcmd_prob.n_nodes, cmcmd_prob.n_edges, cmcmd_prob.n_commodities, cmcmd_prob.n_days

# initialization
lb = []
ub = []

sub_coef = []
sub_sol = []
sub_obj = []

iter = 0
start_time = time.time()

num_processes = min(nR, os.cpu_count())

# subproblem model for each scenario
def solve_subproblem_r(r, z_star, A, b, d, u, gamma, n, m, nC):
    print(f"[debug] start scenario {r}", flush=True)
    try:
        model = gp.Model(f"subproblem_r_{r}")
        model.setParam("OutputFlag", 0)
        model.setParam("FeasibilityTol", 1e-5)
        model.setParam("DualReductions", 0)
        model.setParam("Threads", 1)

        x = model.addVars(n, nC, lb=0.0, name="x")
        y = model.addVars(n, lb=0.0, name="y")

        # y = sum(x)
        for e in range(n):
            model.addConstr(y[e] == gp.quicksum(x[e, cc] for cc in range(nC)))

        # A x = b[:,:,r]
        for i in range(m):
            for cc in range(nC):
                model.addConstr(gp.quicksum(A[i, e] * x[e, cc] for e in range(n)) == b[i, cc, r])

        # y <= u * z
        dual_constr = {}
        for e in range(n):
            dual_constr[e] = model.addConstr(y[e] <= u[e] * z_star[e])

        # objective function
        obj = gp.quicksum(d[e] * y[e] for e in range(n)) + (1 / (2 * gamma)) * gp.quicksum(y[e] * y[e] for e in range(n))
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            duals = {e: u[e] * dual_constr[e].Pi for e in range(n)}
            return model.objVal, duals
        else:
            return None, None

    except Exception as e:
        print(f"Error in scenario {r}: {e}")
        return None, None

def run_benders():
    iter = 0
    lb, ub = [], []

    cut_info = []

    num_processes = min(nR, os.cpu_count())
    start_time = time.time()

    while True:
        # ========== Master Problem ==========
        if iter == 0:
            z_star = z0
            lb.append(0)
            print('Warm Start: Skip the first main problem iteration, directly use z0 for subproblem computation of objective and gradient!')
        else:
            master = gp.Model("masterproblem")
            master.setParam("OutputFlag", 1)
            master.setParam("TimeLimit", 7200)
            master.setParam("FeasibilityTol", 1e-5)

            z = master.addVars(n, vtype=gp.GRB.BINARY, name="z")
            t = master.addVars(nR, lb=0.0, name="t")

            # objective function：sum(t_r)
            master.setObjective(gp.quicksum(c[e] * z[e] for e in range(n)) + gp.quicksum(t[r] for r in range(nR)), GRB.MINIMIZE)

            # add all cut (each cut corresponds to a separate t[r])
            for i, (r, obj_r, dual_r, z_old) in enumerate(cut_info):
                cut = t[r] >= obj_r + gp.quicksum(
                    dual_r[e] * (z[e] - z_old[e]) for e in range(n)
                )
                master.addConstr(cut)

            master.addConstr(gp.quicksum(z[e] for e in range(n)) >= 1, name="at_least_nonzero")

            master.optimize()

            if master.status != GRB.OPTIMAL:
                print(f"[Warning] Main problem did not reach optimality! Status = {master.status}")
                break

            z_star = np.array([z[e].X for e in range(n)])
            lb.append(master.ObjVal)
            print(f"Master Problem Solved!\n\t\tLower Bound: {master.ObjVal}\n\t\tz_star: {z_star}")

        # Parallel subproblem solving for each scenario
        inputs = [(r, z_star, A, b, d, u, gamma, n, m, nC) for r in range(nR)]
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(solve_subproblem_r, inputs)

        total_obj = 0
        for r, (obj_r, dual_r) in enumerate(results):
            if obj_r is None:
                continue
            total_obj += obj_r
            cut_info.append((r, obj_r, dual_r, z_star.copy()))
            print(f"[scenario {r}] Sub-Problem obj = {obj_r:.2f}")

        # upper bound = c·z + ∑ f_r(z)
        ub.append(sum(c[e] * z_star[e] for e in range(n)) + total_obj)
        print(f"Sub-Problem Solved!\n\t\tUpper Bound: {total_obj}")

        iter += 1
        gap = (ub[-1] - lb[-1]) / ub[-1]
        print(f"Iteration {iter} completed, current gap = {gap:.4e}.")

        if gap < 0.0001:
            res_benders = ub[-1]
            sol_z = np.round(z_star).astype(int)
            benders_time = time.time() - start_time
            break

    print("========== Benders Decomposition Completed ==========")
    print(f"Optimal Objective Value: {res_benders:.4f}")
    print(f"Final z (0/1): {sol_z}")
    print(f"Total Time: {benders_time:.2f} seconds")
    print(f"Iterations: {iter}")
    print(f"Gap: {gap}")

if __name__ == "__main__":
    run_benders()