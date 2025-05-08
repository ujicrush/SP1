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

config = Config()
config.ver_prefix = ""
config.jobid = 00000
config.use_vi_1 = True
config.ntries = 50
config.n_clusters = 1
config.n = 20
config.TIME_LIMIT = 7200
config.demand_lb = 0.99
config.use_z0 = True
config.corr_pcnt = 0
config.n_clusters_ratio = 1.0
config.isBenchmark = False
config.nC = 3
config.randomSeed = 1
config.method = "scp_slim"
config.demand_ub = 1.01
config.use_partial_cuts = True
config.nR = 3
config.useMosek = True
config.k_nn = 6
config.slim_repeats = 4
config.rootCuts = 0
config.method_kelley = "scp_slim"
config.verbose_logging = 0
config.Rx = 6
config.k = 20
config.use_file_demands = False
config.is_magnanti_wong_cut = False
config.R_div = 2
config.round_z0 = True
config.Ry = 5
config.gamma = 1.01
config.m = 5
config.use_vi_2 = True
config.use_avg_scenario = True
config.use_si_vi = True
config.R_div_kelley = 4

cmcmd_prob = SimpleNamespace()

cmcmd_prob.n_nodes = 5
cmcmd_prob.n_edges = 20
cmcmd_prob.n_commodities = 3
cmcmd_prob.n_days = 3

cmcmd_prob.k = 20
cmcmd_prob.A = np.array([
    [1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], 
    [-1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], 
    [0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0], 
    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0], 
    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0]
])

cmcmd_prob.b = np.array([
    [6.0, 20.0, -43.0], [-43.0, 7.0, 12.0], [13.0, 22.0, 9.0], [5.0, -72.0, 5.0], [19.0, 23.0, 17.0],
    [21.0, 24.0, -65.0], [-65.0, 7.0, 25.0], [12.0, 24.0, 22.0], [25.0, -75.0, 7.0], [7.0, 20.0, 11.0],
    [7.0, 11.0, -60.0], [-61.0, 11.0, 18.0], [16.0, 19.0, 12.0], [16.0, -50.0, 21.0], [22.0, 9.0, 9.0]
]).reshape((3, 5, 3)).transpose(1, 2, 0)  # shape: (n_nodes, n_commodities, n_days)

cmcmd_prob.c = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

cmcmd_prob.d = np.array([
    1.4407932309895017, 1.7692441614759717, 2.0797308914234005, 2.2373507937299473, 1.4407932309895017, 
    2.094148088838585, 2.8885275443143636, 0.8301228389758245, 1.7692441614759717, 2.094148088838585, 
    0.9840771019041136, 2.420257685825962, 2.0797308914234005, 2.8885275443143636, 0.9840771019041136, 
    3.343418460514712, 2.2373507937299473, 0.8301228389758245, 2.420257685825962, 3.343418460514712
])

cmcmd_prob.u = np.array([
    47.0, 47.0, 50.0, 73.0, 56.0, 49.0, 69.0, 57.0, 73.0, 48.0, 74.0, 60.0, 47.0, 57.0, 54.0, 49.0, 67.0, 69.0, 48.0, 58.0
])

cmcmd_prob.lb = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
cmcmd_prob.ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

cmcmd_prob.gamma = 1.01
cmcmd_prob.sampling_rate = 2.0

cmcmd_prob.edge_map = {
    5: (2, 1, 6), 16: (4, 5, 20), 20: (5, 4, 24), 12: (3, 5, 15), 8: (2, 5, 10), 17: (5, 1, 21), 1: (1, 2, 2), 19: (5, 3, 23), 
    6: (2, 3, 8), 11: (3, 4, 14), 9: (3, 1, 11), 14: (4, 2, 17), 3: (1, 4, 4), 7: (2, 4, 9), 4: (1, 5, 5), 13: (4, 1, 16), 
    15: (4, 3, 18), 2: (1, 3, 3), 10: (3, 2, 12), 18: (5, 2, 22)
}

cmcmd_prob.outgoing_edges = {
    5: [17, 18, 19, 20], 4: [13, 14, 15, 16], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12], 1:[1, 2, 3, 4]
}

cmcmd_prob.incoming_edges = {
    5: [4, 8, 12, 16], 4: [3, 7, 11, 20], 2: [1, 10, 14, 18], 3: [2, 6, 15, 19], 1: [5, 9, 13, 17]
}

cmcmd_prob.old_to_new_map = {
    5: 4, 16: 13, 20: 16, 12: 10, 24: 20, 8: 6, 17: 14, 23: 19, 22: 18, 6: 5, 11: 9, 9: 7, 14: 11, 3: 2, 4: 3, 15: 12, 21: 17, 
    2: 1, 10: 8, 18: 15
}

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

# Subproblem model for each scenario
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
    sub_coef, sub_sol, sub_obj = [], [], []

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
            t = master.addVar(lb=0.0, name="t")

            master.setObjective(gp.quicksum(c[e] * z[e] for e in range(n)) + t, gp.GRB.MINIMIZE)

            for i in range(len(sub_coef)):
                cut = t >= sub_obj[i] + gp.quicksum(sub_coef[i][e] * (z[e] - sub_sol[i][e]) for e in range(n))
                master.addConstr(cut)

            master.addConstr(gp.quicksum(z[e] for e in range(n)) >= 0.01, name="at_least_nonzero")
            master.optimize()

            z_star = np.array([z[e].X for e in range(n)])
            lb.append(master.ObjVal)
            print(f"Master Problem Solved\n\t\tLower Bound: {master.ObjVal}\n\t\tValues of z_star: {z_star}")

        # Parallel subproblem solving for each scenario
        inputs = [(r, z_star, A, b, d, u, gamma, n, m, nC) for r in range(nR)]
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(solve_subproblem_r, inputs)

        total_obj = 0
        dual_agg = {e: 0 for e in range(n)}

        for r, (obj_r, dual_r) in enumerate(results):
            if obj_r is None:
                continue
            total_obj += obj_r
            for e in range(n):
                dual_agg[e] += dual_r[e]

        sub_coef.append(dual_agg)
        sub_sol.append(z_star.copy())
        sub_obj.append(total_obj)
        ub.append(sum(c[e] * z_star[e] for e in range(n)) + total_obj)
        print(f"Evaluation Solved\n\t\tObjective Value: {total_obj}")

        iter += 1
        gap = (ub[-1] - lb[-1]) / ub[-1]
        print(f"Iteration {iter} completed, current gap = {gap:.4e}.")

        if gap < 0.0001:
            res_benders = ub[-1]
            sol_z = z_star
            benders_time = time.time() - start_time
            break

    print("========== Benders Decomposition Completed ==========")
    print(f"Optimal Objective Value: {res_benders:.4f}")
    print(f"Final z: {sol_z}")
    print(f"Total Time: {benders_time:.2f} seconds")
    print(f"Iterations: {iter}")
    print(f"Gap: {gap}")

if __name__ == "__main__":
    run_benders()